#pragma once

#include "ros_yolo/frustum_geometry.h"
#include <sensor_msgs/PointCloud2.h>

namespace ros_yolo
{
// 输入点云已位于 body 坐标系；输出保留每个命中点的全部原始字段。
// 无法安全读取的 XYZ 字段或点云布局会抛出 std::invalid_argument。
sensor_msgs::PointCloud2 filterPointCloud(const sensor_msgs::PointCloud2& cloud,
                                         const BodyFrustum& frustum);
}  // namespace ros_yolo
