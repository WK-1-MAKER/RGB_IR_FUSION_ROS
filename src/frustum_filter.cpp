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
}  // namespace

sensor_msgs::PointCloud2 filterPointCloud(const sensor_msgs::PointCloud2& cloud,
                                         const BodyFrustum& frustum)
{
    sensor_msgs::PointCloud2 filtered;
    filtered.header = cloud.header;
    filtered.height = 1;
    filtered.fields = cloud.fields;
    filtered.is_bigendian = cloud.is_bigendian;
    filtered.point_step = cloud.point_step;
    filtered.is_dense = cloud.is_dense;
    if (cloud.width == 0 || cloud.height == 0) return filtered;

    const uint64_t rowBytes = static_cast<uint64_t>(cloud.width) * cloud.point_step;
    const uint64_t totalBytes = static_cast<uint64_t>(cloud.height) * cloud.row_step;
    if (cloud.point_step == 0 || rowBytes > cloud.row_step || totalBytes > cloud.data.size())
    {
        throw std::invalid_argument("PointCloud2 data/row_step/point_step layout is inconsistent.");
    }
    const auto& x = coordinateField(cloud, "x");
    const auto& y = coordinateField(cloud, "y");
    const auto& z = coordinateField(cloud, "z");
    const uint16_t marker = 1;
    const bool hostBigEndian = *reinterpret_cast<const uint8_t*>(&marker) == 0;
    const bool swap = hostBigEndian != cloud.is_bigendian;
    filtered.data.reserve(cloud.data.size());
    for (uint32_t row = 0; row < cloud.height; ++row)
    {
        for (uint32_t column = 0; column < cloud.width; ++column)
        {
            const size_t offset = static_cast<size_t>(row) * cloud.row_step +
                                  static_cast<size_t>(column) * cloud.point_step;
            const uint8_t* record = cloud.data.data() + offset;
            const cv::Vec3d point(readCoordinate(record, x, swap),
                                  readCoordinate(record, y, swap),
                                  readCoordinate(record, z, swap));
            if (!frustum.contains(point)) continue;
            if (filtered.data.size() > std::numeric_limits<uint32_t>::max() - cloud.point_step)
            {
                throw std::overflow_error("Filtered PointCloud2 exceeds row_step capacity.");
            }
            filtered.data.insert(filtered.data.end(), record, record + cloud.point_step);
            ++filtered.width;
        }
    }
    filtered.row_step = static_cast<uint32_t>(filtered.data.size());
    return filtered;
}
}  // namespace ros_yolo
