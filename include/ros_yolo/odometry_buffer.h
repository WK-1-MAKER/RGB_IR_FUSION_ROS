#pragma once

#include <nav_msgs/Odometry.h>
#include <opencv2/core.hpp>
#include <ros/ros.h>

#include <cstddef>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>

namespace ros_yolo
{
struct BodyPose
{
    cv::Matx33d rotation = cv::Matx33d::eye();
    cv::Vec3d translation{0.0, 0.0, 0.0};

    cv::Vec3d transform(const cv::Vec3d& point) const
    {
        return rotation * point + translation;
    }
};

class OdometryBuffer
{
public:
    OdometryBuffer(double max_pose_age_seconds,
                   double cache_duration_seconds,
                   std::string world_frame);
    void insert(const nav_msgs::Odometry& message);
    // Wait for odometry to cover stamp, then select the nearest sample without interpolation.
    bool lookup(const ros::Time& stamp, BodyPose& pose, double wait_timeout_seconds) const;
    std::size_t generation() const;

private:
    struct Sample { ros::Time stamp; BodyPose pose; };
    static BodyPose poseFromMessage(const nav_msgs::Odometry& message);
    void pruneLocked();

    double max_pose_age_seconds_;
    double cache_duration_seconds_;
    std::string world_frame_;
    mutable std::mutex mutex_;
    mutable std::condition_variable condition_;
    std::deque<Sample> samples_;
    std::size_t generation_{0};
};
}  // namespace ros_yolo
