#include "ros_yolo/odometry_buffer.h"

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <utility>

namespace ros_yolo
{
OdometryBuffer::OdometryBuffer(double max_pose_age_seconds,
                               double cache_duration_seconds,
                               std::string world_frame)
    : max_pose_age_seconds_(max_pose_age_seconds),
      cache_duration_seconds_(cache_duration_seconds),
      world_frame_(std::move(world_frame))
{
    if (!std::isfinite(max_pose_age_seconds_) || max_pose_age_seconds_ <= 0.0 ||
        !std::isfinite(cache_duration_seconds_) || cache_duration_seconds_ < max_pose_age_seconds_ ||
        world_frame_.empty())
        throw std::invalid_argument("invalid odometry buffer configuration");
}

BodyPose OdometryBuffer::poseFromMessage(const nav_msgs::Odometry& message)
{
    const auto& p = message.pose.pose.position;
    const auto& o = message.pose.pose.orientation;
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
        !std::isfinite(o.x) || !std::isfinite(o.y) || !std::isfinite(o.z) || !std::isfinite(o.w))
        throw std::invalid_argument("odometry pose contains a non-finite value");
    Eigen::Quaterniond q(o.w, o.x, o.y, o.z);
    if (!std::isfinite(q.norm()) || q.norm() <= 1e-12)
        throw std::invalid_argument("odometry orientation has zero norm");
    q.normalize();
    const Eigen::Matrix3d matrix = q.toRotationMatrix();
    BodyPose pose;
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col) pose.rotation(row, col) = matrix(row, col);
    pose.translation = cv::Vec3d(p.x, p.y, p.z);
    return pose;
}

void OdometryBuffer::insert(const nav_msgs::Odometry& message)
{
    if (message.header.stamp.isZero()) throw std::invalid_argument("odometry timestamp must be nonzero");
    const BodyPose pose = poseFromMessage(message);
    std::lock_guard<std::mutex> lock(mutex_);
    if (message.header.frame_id != world_frame_)
    {
        samples_.clear();
        ++generation_;
        return;
    }
    if (!samples_.empty() && message.header.stamp < samples_.back().stamp)
    {
        samples_.clear();
        ++generation_;
    }
    const auto existing = std::find_if(samples_.begin(), samples_.end(),
        [&message](const Sample& sample) { return sample.stamp == message.header.stamp; });
    if (existing != samples_.end()) existing->pose = pose;
    else samples_.push_back({message.header.stamp, pose});
    pruneLocked();
    condition_.notify_all();
}

void OdometryBuffer::pruneLocked()
{
    if (samples_.empty()) return;
    const ros::Time newest = samples_.back().stamp;
    const ros::Duration horizon(cache_duration_seconds_);
    while (samples_.size() > 1 && newest - samples_[1].stamp > horizon) samples_.pop_front();
}

bool OdometryBuffer::lookup(const ros::Time& stamp, BodyPose& pose, double wait_timeout_seconds) const
{
    if (stamp.isZero()) return false;
    std::unique_lock<std::mutex> lock(mutex_);
    // Do not select the previous scan while this scan's odometry is still arriving.
    if (!condition_.wait_for(lock, std::chrono::duration<double>(wait_timeout_seconds), [this, &stamp]() {
            return !samples_.empty() && samples_.back().stamp >= stamp;
        })) return false;

    auto nearest = std::lower_bound(samples_.begin(), samples_.end(), stamp,
        [](const Sample& sample, const ros::Time& value) { return sample.stamp < value; });
    if (nearest != samples_.begin())
    {
        const auto lower = std::prev(nearest);
        if (stamp - lower->stamp <= nearest->stamp - stamp) nearest = lower;
    }
    if (std::abs((nearest->stamp - stamp).toSec()) > max_pose_age_seconds_) return false;
    pose = nearest->pose;
    return true;
}

std::size_t OdometryBuffer::generation() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return generation_;
}
}  // namespace ros_yolo
