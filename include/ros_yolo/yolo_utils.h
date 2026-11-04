#pragma once

#include <codecvt>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace ros_yolo
{
struct Detection
{
    cv::Rect box;
    float conf{};
    int classId{};
};

struct TrackAnnotation
{
    int track_id = 0;
    bool world_position_valid = false;
    cv::Point2d world_position;
};

namespace utils
{
size_t vectorProduct(const std::vector<int64_t>& vector);
std::wstring charToWstring(const char* str);
std::vector<std::string> loadNames(const std::string& path);

void visualizeDetection(cv::Mat& image,
                        const std::vector<Detection>& detections,
                        const std::vector<std::string>& classNames,
                        const std::vector<cv::Point3f>& targetPoints3d = {},
                        const std::vector<TrackAnnotation>& tracks = {});

std::string formatTrackLabel(const TrackAnnotation& track);

void letterbox(const cv::Mat& image,
               cv::Mat& outImage,
               const cv::Size& newShape,
               const cv::Scalar& color,
               bool auto_,
               bool scaleFill,
               bool scaleUp,
               int stride);

void scaleCoords(const cv::Size& imageShape, cv::Rect& box, const cv::Size& imageOriginalShape);

template <typename T>
T clip(const T& n, const T& lower, const T& upper)
{
    return std::max(lower, std::min(n, upper));
}

}  // namespace utils
}  // namespace ros_yolo
