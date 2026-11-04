#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

namespace ros_yolo
{

struct ByteTrack3DConfig
{
    double track_high_threshold = 0.60;
    double track_low_threshold = 0.10;
    double new_track_threshold = 0.60;
    int track_buffer = 30;
    // Negative values derive seconds-based expiry from track_buffer. The node
    // supplies explicit values when running from ROS sensor timestamps.
    double track_buffer_seconds = -1.0;
    double world_measurement_timeout_seconds = -1.0;
    double match_threshold = 0.80;
    bool fuse_score = true;
    // Expose allocated IDs for current new tracks before temporal confirmation.
    bool return_unconfirmed_tracks = false;

    double measurement_std = 0.30;
    double acceleration_std = 0.10;
    double initial_velocity_std = 1.00;
    double image_acceleration_std = 0.01;
    double aspect_acceleration_std = 0.1;
    double initial_image_velocity_scale = 0.0625;
    double world_weight = 0.60;
    double world_gate = 9.21034;
};

// Independent of the detector and ROS messages. box is (left, top, width, height)
// in pixels. position is optional fixed-reference world-plane XY in meters;
// either non-finite component makes the entire observation missing.
struct ByteTrack3DDetection
{
    cv::Rect2d box;
    double score = 0.0;
    int class_id = 0;
    cv::Point2d position{std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::quiet_NaN()};
};

struct ByteTrack3DResult
{
    cv::Rect2d box;
    int track_id = 0;
    double score = 0.0;
    int class_id = 0;
    std::size_t detection_index = 0;
    cv::Point2d position{std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::quiet_NaN()};
    // World-plane velocity in meters per second; NaN when world state is unavailable.
    cv::Point2d velocity{std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::quiet_NaN()};

    // Python column order: x1,y1,x2,y2,track_id,score,class_id,detection_index,X,Y.
    std::array<double, 10> toArray() const;
};

class BYTETracker3D
{
public:
    explicit BYTETracker3D(const ByteTrack3DConfig& config = ByteTrack3DConfig());
    ~BYTETracker3D();

    BYTETracker3D(const BYTETracker3D&) = delete;
    BYTETracker3D& operator=(const BYTETracker3D&) = delete;

    // Compatibility wrapper: advances one synthetic second per call.
    std::vector<ByteTrack3DResult> update(const std::vector<ByteTrack3DDetection>& detections);
    // Call once per sensor timestamp, including update({}, timestamp) for empty
    // frames. Velocities are per second. Invalid input throws std::invalid_argument
    // before advancing any state. Non-positive box extents are silently dropped.
    // Currently matched tracks are returned in track-pool order. By default only
    // confirmed tracks are returned; return_unconfirmed_tracks also exposes new tracks.
    // detection_index is the original zero-based row in detections.
    // This stateful object is intended for use from a single thread.
    std::vector<ByteTrack3DResult> update(const std::vector<ByteTrack3DDetection>& detections,
                                          double timestamp_seconds);
    // Clear image/world trajectories and restart this instance's IDs at 1.
    void reset();
    int frameId() const;
    std::vector<int> removedTrackIds() const;  // Removed in the latest update only.

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ros_yolo
