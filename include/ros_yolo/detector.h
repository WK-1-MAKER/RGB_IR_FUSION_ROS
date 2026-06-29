#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

#include "ros_yolo/yolo_utils.h"

namespace ros_yolo
{
std::vector<Detection> decodeCurrentYolov8Tensor(const Ort::Value& tensor,
                                                 const cv::Size& resizedImageShape,
                                                 const cv::Size& originalImageShape,
                                                 float confThreshold,
                                                 float iouThreshold);

class YoloDetector
{
public:
    explicit YoloDetector(std::nullptr_t) {}
    YoloDetector(const std::string& modelPath, bool isGPU, const cv::Size& inputSize);

    std::vector<Detection> detect(cv::Mat& rgbImage,
                                  cv::Mat& irImage,
                                  float confThreshold,
                                  float iouThreshold);

private:
    void preprocessing(const cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape);
    std::vector<Detection> postprocessing(const cv::Size& resizedImageShape,
                                          const cv::Size& originalImageShape,
                                          std::vector<Ort::Value>& outputTensors,
                                          float confThreshold,
                                          float iouThreshold);

    Ort::Env env_{nullptr};
    Ort::SessionOptions sessionOptions_{nullptr};
    Ort::Session session_{nullptr};

    std::vector<std::string> inputNameStrings_;
    std::vector<std::string> outputNameStrings_;
    std::vector<const char*> inputNames_;
    std::vector<const char*> outputNames_;
    bool isDynamicInputShape_{false};
    cv::Size2f inputImageShape_;
};
}  // namespace ros_yolo
