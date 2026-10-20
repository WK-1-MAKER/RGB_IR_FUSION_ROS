#include "ros_yolo/frustum_filter.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace
{
template <typename T>
void writeValue(uint8_t* output, T value, bool bigEndian)
{
    uint8_t bytes[sizeof(T)];
    std::memcpy(bytes, &value, sizeof(T));
    const uint16_t marker = 1;
    const bool hostBigEndian = *reinterpret_cast<const uint8_t*>(&marker) == 0;
    if (hostBigEndian != bigEndian) std::reverse(bytes, bytes + sizeof(T));
    std::copy(bytes, bytes + sizeof(T), output);
}

sensor_msgs::PointCloud2 makeCloud(bool doubles, bool bigEndian)
{
    sensor_msgs::PointCloud2 cloud;
    cloud.header.frame_id = "body";
    cloud.header.stamp = ros::Time(123, 456);
    cloud.header.seq = 17;
    cloud.width = 3;
    cloud.height = 2;
    cloud.is_bigendian = bigEndian;
    const uint32_t scalarSize = doubles ? 8 : 4;
    cloud.point_step = 3 * scalarSize + 4;
    cloud.row_step = cloud.width * cloud.point_step + 7;
    cloud.data.resize(cloud.row_step * cloud.height, 0xa5);
    for (size_t i = 0; i < 3; ++i)
    {
        sensor_msgs::PointField field;
        field.name = std::string(1, "xyz"[i]);
        field.offset = i * scalarSize;
        field.datatype = doubles ? sensor_msgs::PointField::FLOAT64 : sensor_msgs::PointField::FLOAT32;
        field.count = 1;
        cloud.fields.push_back(field);
    }
    sensor_msgs::PointField intensity;
    intensity.name = "intensity";
    intensity.offset = 3 * scalarSize;
    intensity.datatype = sensor_msgs::PointField::UINT32;
    intensity.count = 1;
    cloud.fields.push_back(intensity);
    const cv::Vec3d points[] = {{0, 0, 5}, {2, 0, 5}, {0, 0, -5},
                              {0.25, -0.25, 5}, {0, 0, std::numeric_limits<double>::quiet_NaN()}, {0, 2, 5}};
    for (size_t i = 0; i < 6; ++i)
    {
        uint8_t* record = cloud.data.data() + (i / 3) * cloud.row_step + (i % 3) * cloud.point_step;
        for (size_t j = 0; j < 3; ++j)
        {
            if (doubles) writeValue(record + j * scalarSize, points[i][j], bigEndian);
            else writeValue(record + j * scalarSize, static_cast<float>(points[i][j]), bigEndian);
        }
        writeValue(record + 3 * scalarSize, static_cast<uint32_t>(100 + i), bigEndian);
    }
    return cloud;
}

ros_yolo::BodyFrustum makeFrustum(const cv::Rect& box = cv::Rect(90, 90, 20, 20))
{
    ros_yolo::BodyFrustum result;
    const cv::Matx33d k(100, 0, 100, 0, 100, 100, 0, 0, 1);
    EXPECT_TRUE(ros_yolo::buildBodyFrustum(box, k, cv::Matx33d::eye(), cv::Vec3d(0, 0, 0), result));
    return result;
}
}

TEST(FrustumFilterTest, PreservesRecordsAndHeaderWithRowPaddingAndBothByteOrders)
{
    for (bool doubles : {false, true}) for (bool bigEndian : {false, true})
    {
        const auto cloud = makeCloud(doubles, bigEndian);
        const auto filtered = ros_yolo::filterPointCloud(cloud, makeFrustum());
        EXPECT_EQ(filtered.width, 2u);
        EXPECT_EQ(filtered.height, 1u);
        EXPECT_EQ(filtered.header, cloud.header);
        EXPECT_EQ(filtered.fields, cloud.fields);
        EXPECT_EQ(filtered.is_bigendian, cloud.is_bigendian);
        EXPECT_EQ(filtered.point_step, cloud.point_step);
        EXPECT_EQ(filtered.row_step, 2 * cloud.point_step);
        std::vector<uint8_t> expected(cloud.data.begin(), cloud.data.begin() + cloud.point_step);
        expected.insert(expected.end(), cloud.data.begin() + cloud.row_step,
                        cloud.data.begin() + cloud.row_step + cloud.point_step);
        EXPECT_EQ(filtered.data, expected);
    }
}

TEST(FrustumFilterTest, FiltersEachBoxIndependently)
{
    const auto cloud = makeCloud(false, false);
    const auto left = ros_yolo::filterPointCloud(cloud, makeFrustum());
    const auto right = ros_yolo::filterPointCloud(cloud, makeFrustum(cv::Rect(130, 90, 20, 20)));
    EXPECT_EQ(left.width, 2u);
    ASSERT_EQ(right.width, 1u);
    EXPECT_TRUE(std::equal(right.data.begin(), right.data.end(), cloud.data.begin() + cloud.point_step));
}

TEST(FrustumFilterTest, HandlesEmptyCloudAndRejectsInvalidLayoutBeforeReading)
{
    auto cloud = makeCloud(false, false);
    cloud.width = 0; cloud.height = 1; cloud.row_step = 0; cloud.data.clear();
    EXPECT_TRUE(ros_yolo::filterPointCloud(cloud, makeFrustum()).data.empty());
    cloud = makeCloud(false, false);
    cloud.data.resize(1);
    EXPECT_THROW(ros_yolo::filterPointCloud(cloud, makeFrustum()), std::invalid_argument);
    cloud = makeCloud(false, false);
    cloud.fields[2].offset = cloud.point_step;
    EXPECT_THROW(ros_yolo::filterPointCloud(cloud, makeFrustum()), std::invalid_argument);
    cloud = makeCloud(false, false);
    cloud.row_step = 1;
    EXPECT_THROW(ros_yolo::filterPointCloud(cloud, makeFrustum()), std::invalid_argument);
}
