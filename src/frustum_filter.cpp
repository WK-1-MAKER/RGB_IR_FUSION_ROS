#include "ros_yolo/frustum_filter.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace ros_yolo
{
namespace
{
const sensor_msgs::PointField& coordinateField(const sensor_msgs::PointCloud2& cloud,
                                               const char* name)
{
    const auto field = std::find_if(cloud.fields.begin(), cloud.fields.end(),
        [name](const sensor_msgs::PointField& candidate) { return candidate.name == name; });
    if (field == cloud.fields.end() || field->count != 1 ||
        (field->datatype != sensor_msgs::PointField::FLOAT32 &&
         field->datatype != sensor_msgs::PointField::FLOAT64))
    {
        throw std::invalid_argument("PointCloud2 requires scalar FLOAT32/FLOAT64 x, y, z fields.");
    }
    const uint32_t size = field->datatype == sensor_msgs::PointField::FLOAT32 ? 4 : 8;
    if (field->offset > cloud.point_step || size > cloud.point_step - field->offset)
    {
        throw std::invalid_argument("PointCloud2 coordinate extends beyond point_step.");
    }
    return *field;
}

template <typename T>
double readScalar(const uint8_t* bytes, bool swap)
{
    uint8_t local[sizeof(T)];
    std::copy(bytes, bytes + sizeof(T), local);
    if (swap) std::reverse(local, local + sizeof(T));
    T value;
    std::memcpy(&value, local, sizeof(T));
    return static_cast<double>(value);
}

double readCoordinate(const uint8_t* record, const sensor_msgs::PointField& field, bool swap)
{
    return field.datatype == sensor_msgs::PointField::FLOAT32
        ? readScalar<float>(record + field.offset, swap)
        : readScalar<double>(record + field.offset, swap);
}

sensor_msgs::PointCloud2 makeEmptyFilteredCloud(const sensor_msgs::PointCloud2& cloud)
{
    sensor_msgs::PointCloud2 filtered;
    filtered.header = cloud.header;
    filtered.height = 1;
    filtered.fields = cloud.fields;
    filtered.is_bigendian = cloud.is_bigendian;
    filtered.point_step = cloud.point_step;
    filtered.is_dense = cloud.is_dense;
    return filtered;
}

void validateCloudLayout(const sensor_msgs::PointCloud2& cloud)
{
    const uint64_t rowBytes = static_cast<uint64_t>(cloud.width) * cloud.point_step;
    const uint64_t totalBytes = static_cast<uint64_t>(cloud.height) * cloud.row_step;
    if (cloud.point_step == 0 || rowBytes > cloud.row_step || totalBytes > cloud.data.size())
    {
        throw std::invalid_argument("PointCloud2 data/row_step/point_step layout is inconsistent.");
    }
}

}  // namespace

sensor_msgs::PointCloud2 filterPointCloud(const sensor_msgs::PointCloud2& cloud,
                                         const BodyFrustum& frustum)
{
    return filterPointCloudBatch(cloud, std::vector<BodyFrustum>{frustum}).filtered_clouds.front();
}

BatchedPointCloudResult filterPointCloudBatch(const sensor_msgs::PointCloud2& cloud,
                                             const std::vector<BodyFrustum>& frustums)
{
    BatchedPointCloudResult result;
    result.target_point_indices.resize(frustums.size());
    result.filtered_clouds.reserve(frustums.size());
    for (std::size_t i = 0; i < frustums.size(); ++i)
    {
        result.filtered_clouds.push_back(makeEmptyFilteredCloud(cloud));
    }
    if (cloud.width == 0 || cloud.height == 0 || frustums.empty()) return result;

    validateCloudLayout(cloud);
    const auto& x = coordinateField(cloud, "x");
    const auto& y = coordinateField(cloud, "y");
    const auto& z = coordinateField(cloud, "z");
    const uint16_t marker = 1;
    const bool hostBigEndian = *reinterpret_cast<const uint8_t*>(&marker) == 0;
    const bool swap = hostBigEndian != cloud.is_bigendian;
    for (auto& filtered : result.filtered_clouds) filtered.data.reserve(cloud.data.size());

    for (uint32_t row = 0; row < cloud.height; ++row)
    {
        for (uint32_t column = 0; column < cloud.width; ++column)
        {
            const std::size_t offset = static_cast<std::size_t>(row) * cloud.row_step +
                                       static_cast<std::size_t>(column) * cloud.point_step;
            const uint8_t* record = cloud.data.data() + offset;
            const cv::Vec3d point(readCoordinate(record, x, swap),
                                  readCoordinate(record, y, swap),
                                  readCoordinate(record, z, swap));
            if (!std::isfinite(point[0]) || !std::isfinite(point[1]) || !std::isfinite(point[2])) continue;

            std::vector<std::size_t> matchedTargets;
            for (std::size_t target = 0; target < frustums.size(); ++target)
            {
                if (frustums[target].contains(point)) matchedTargets.push_back(target);
            }
            if (matchedTargets.empty()) continue;

            const std::size_t candidateIndex = result.candidate_points.size();
            result.candidate_points.push_back(point);
            result.source_offsets.push_back(offset);
            for (const std::size_t target : matchedTargets)
            {
                result.target_point_indices[target].push_back(candidateIndex);
                auto& filtered = result.filtered_clouds[target];
                if (filtered.data.size() > std::numeric_limits<uint32_t>::max() - cloud.point_step)
                {
                    throw std::overflow_error("Filtered PointCloud2 exceeds row_step capacity.");
                }
                filtered.data.insert(filtered.data.end(), record, record + cloud.point_step);
                ++filtered.width;
            }
        }
    }
    for (auto& filtered : result.filtered_clouds)
    {
        filtered.row_step = static_cast<uint32_t>(filtered.data.size());
    }
    return result;
}
}  // namespace ros_yolo
