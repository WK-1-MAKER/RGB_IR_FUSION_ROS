#pragma once

#include <opencv2/opencv.hpp>

#include <cstdint>
#include <vector>

#include "ros_yolo/yolo_utils.h"

namespace ros_yolo
{
std::vector<Detection> decodeYolov8Detections(const std::vector<float>& output,
                                              const std::vector<int64_t>& shape,
                                              const cv::Size& resizedImageShape,
                                              const cv::Size& originalImageShape,
                                              float confThreshold,
                                              float iouThreshold);
}  // namespace ros_yolo
