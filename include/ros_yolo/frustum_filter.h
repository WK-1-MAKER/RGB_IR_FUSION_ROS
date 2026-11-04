#pragma once

#include "ros_yolo/frustum_geometry.h"
#include <sensor_msgs/PointCloud2.h>

#include <cstddef>
#include <vector>
namespace ros_yolo
{
// 输入点云已位于 body 坐标系；输出保留每个命中点的全部原始字段。
// 无法安全读取的 XYZ 字段或点云布局会抛出 std::invalid_argument。
sensor_msgs::PointCloud2 filterPointCloud(const sensor_msgs::PointCloud2& cloud,
                                         const BodyFrustum& frustum);

struct BatchedPointCloudResult
{
    std::vector<cv::Vec3d> candidate_points;
    std::vector<std::size_t> source_offsets;
    std::vector<std::vector<std::size_t>> target_point_indices;
    std::vector<sensor_msgs::PointCloud2> filtered_clouds;
};

BatchedPointCloudResult filterPointCloudBatch(const sensor_msgs::PointCloud2& cloud,
                                             const std::vector<BodyFrustum>& frustums);
}  // namespace ros_yolo
