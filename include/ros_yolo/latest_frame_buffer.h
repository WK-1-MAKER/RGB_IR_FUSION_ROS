#pragma once

#include <sensor_msgs/Image.h>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <utility>

namespace ros_yolo
{
struct FramePair
{
    FramePair() = default;

    FramePair(const sensor_msgs::ImageConstPtr& rgbIn,
              const sensor_msgs::ImageConstPtr& irIn)
        : rgb(rgbIn), ir(irIn)
    {
    }

    sensor_msgs::ImageConstPtr rgb;
    sensor_msgs::ImageConstPtr ir;
};

class LatestFrameBuffer
{
public:
    void push(FramePair frame)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopped_)
            {
                return;
            }
            pending_.reset(new FramePair(std::move(frame)));
        }
        condition_.notify_one();
    }

    bool waitAndTake(FramePair& frame)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this]() { return stopped_ || pending_; });
        if (stopped_)
        {
            return false;
        }

        frame = std::move(*pending_);
        pending_.reset();
        return true;
    }

    void stop()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
            pending_.reset();
        }
        condition_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    std::unique_ptr<FramePair> pending_;
    bool stopped_{false};
};
}  // namespace ros_yolo
