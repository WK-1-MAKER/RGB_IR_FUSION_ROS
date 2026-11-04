#pragma once

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>

namespace ros_yolo
{
struct EuclideanClusterConfig
{
    double cluster_tolerance = 0.5;
    std::size_t min_cluster_size = 3;
    std::size_t max_cluster_size = 100000;
};

struct LidarLocalizationResult
{
    bool valid = false;
    cv::Vec3d centroid_body{0.0, 0.0, 0.0};
    std::vector<std::size_t> cluster_point_indices;
};

struct AxisAlignedBounds
{
    cv::Vec3d minimum{0.0, 0.0, 0.0};
    cv::Vec3d maximum{0.0, 0.0, 0.0};
};

// Computes tight bounds along the axes of the points' coordinate frame.
AxisAlignedBounds axisAlignedBounds(const std::vector<cv::Vec3d>& points);

class LidarLocalizer
{
public:
    explicit LidarLocalizer(const EuclideanClusterConfig& config);

    LidarLocalizationResult localize(const std::vector<cv::Vec3d>& candidate_points,
                                     const std::vector<std::size_t>& source_offsets,
                                     const std::vector<std::size_t>& target_point_indices) const;

private:
    EuclideanClusterConfig config_;
};
}  // namespace ros_yolo
