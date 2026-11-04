#include "ros_yolo/lidar_localizer.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ros_yolo
{
AxisAlignedBounds axisAlignedBounds(const std::vector<cv::Vec3d>& points)
{
    if (points.empty()) throw std::invalid_argument("cannot compute bounds of an empty point set");

    AxisAlignedBounds bounds;
    bounds.minimum = cv::Vec3d(std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::infinity());
    bounds.maximum = cv::Vec3d(-std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity());
    for (const cv::Vec3d& point : points)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            if (!std::isfinite(point[axis]))
                throw std::invalid_argument("cannot compute bounds of non-finite points");
            bounds.minimum[axis] = std::min(bounds.minimum[axis], point[axis]);
            bounds.maximum[axis] = std::max(bounds.maximum[axis], point[axis]);
        }
    }
    return bounds;
}

LidarLocalizer::LidarLocalizer(const EuclideanClusterConfig& config) : config_(config)
{
    if (!std::isfinite(config_.cluster_tolerance) || config_.cluster_tolerance <= 0.0 ||
        config_.min_cluster_size == 0 || config_.max_cluster_size < config_.min_cluster_size ||
        config_.max_cluster_size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw std::invalid_argument("invalid Euclidean cluster configuration");
    }
}

LidarLocalizationResult LidarLocalizer::localize(
    const std::vector<cv::Vec3d>& candidate_points,
    const std::vector<std::size_t>& source_offsets,
    const std::vector<std::size_t>& target_point_indices) const
{
    if (source_offsets.size() != candidate_points.size())
    {
        throw std::invalid_argument("candidate points and source offsets must have equal size");
    }
    LidarLocalizationResult result;
    if (target_point_indices.empty()) return result;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    std::vector<std::size_t> localToCandidate;
    cloud->points.reserve(target_point_indices.size());
    localToCandidate.reserve(target_point_indices.size());
    for (const std::size_t candidateIndex : target_point_indices)
    {
        if (candidateIndex >= candidate_points.size())
        {
            throw std::out_of_range("target point index is outside the candidate point array");
        }
        const cv::Vec3d& point = candidate_points[candidateIndex];
        const float x = static_cast<float>(point[0]);
        const float y = static_cast<float>(point[1]);
        const float z = static_cast<float>(point[2]);
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;
        cloud->points.emplace_back(x, y, z);
        localToCandidate.push_back(candidateIndex);
    }
    if (cloud->points.empty()) return result;
    cloud->width = static_cast<std::uint32_t>(cloud->points.size());
    cloud->height = 1;
    cloud->is_dense = true;

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    tree->setInputCloud(cloud);
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> extraction;
    extraction.setClusterTolerance(config_.cluster_tolerance);
    extraction.setMinClusterSize(static_cast<int>(config_.min_cluster_size));
    extraction.setMaxClusterSize(static_cast<int>(config_.max_cluster_size));
    extraction.setSearchMethod(tree);
    extraction.setInputCloud(cloud);

    std::vector<pcl::PointIndices> clusters;
    extraction.extract(clusters);
    const pcl::PointIndices* selected = nullptr;
    std::size_t selectedFirstOffset = std::numeric_limits<std::size_t>::max();
    for (const auto& cluster : clusters)
    {
        std::size_t firstOffset = std::numeric_limits<std::size_t>::max();
        for (const int localIndex : cluster.indices)
        {
            firstOffset = std::min(firstOffset, source_offsets[localToCandidate.at(localIndex)]);
        }
        if (!selected || cluster.indices.size() > selected->indices.size() ||
            (cluster.indices.size() == selected->indices.size() && firstOffset < selectedFirstOffset))
        {
            selected = &cluster;
            selectedFirstOffset = firstOffset;
        }
    }
    if (!selected) return result;

    result.cluster_point_indices.reserve(selected->indices.size());
    for (const int localIndex : selected->indices)
    {
        result.cluster_point_indices.push_back(localToCandidate.at(localIndex));
    }
    std::sort(result.cluster_point_indices.begin(), result.cluster_point_indices.end());
    for (const std::size_t candidateIndex : result.cluster_point_indices)
    {
        result.centroid_body += candidate_points[candidateIndex];
    }
    result.centroid_body *= 1.0 / static_cast<double>(result.cluster_point_indices.size());
    result.valid = true;
    return result;
}
}  // namespace ros_yolo
