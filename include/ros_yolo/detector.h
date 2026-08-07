#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <memory>
#include <string>
#include <vector>

#include "3rdparty/tensorrtbuffer/include/buffers.h"
#include "ros_yolo/yolo_utils.h"

namespace ros_yolo
{
enum class InferenceBackend
{
    ONNX,
    TENSORRT
};

InferenceBackend inferBackendFromModelPath(const std::string& modelPath);

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
    ~YoloDetector();

    std::vector<Detection> detect(cv::Mat& rgbImage,
                                  cv::Mat& irImage,
                                  float confThreshold,
                                  float iouThreshold);

private:
    void initializeTensorRt(const std::string& modelPath);
    std::vector<Detection> detectTensorRt(const std::vector<float>& rgbInput,
                                          const std::vector<float>& irInput,
                                          const cv::Size& resizedImageShape,
                                          const cv::Size& originalImageShape,
                                          float confThreshold,
                                          float iouThreshold);
    void preprocessing(const cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape);
    std::vector<Detection> postprocessing(const cv::Size& resizedImageShape,
                                          const cv::Size& originalImageShape,
                                          std::vector<Ort::Value>& outputTensors,
                                          float confThreshold,
                                          float iouThreshold);

    Ort::Env env_{nullptr};
    Ort::SessionOptions sessionOptions_{nullptr};
    Ort::Session session_{nullptr};

    InferenceBackend backend_{InferenceBackend::ONNX};
    // The runtime must outlive the deserialized engine and execution context.
    tensorrt_buffer::TensorRTUniquePtr<nvinfer1::IRuntime> trtRuntime_;
    std::shared_ptr<nvinfer1::ICudaEngine> trtEngine_;
    std::shared_ptr<nvinfer1::IExecutionContext> trtContext_;
    std::unique_ptr<tensorrt_buffer::BufferManager> trtBuffers_;
    cudaStream_t trtStream_{nullptr};
    std::vector<int64_t> trtOutputShape_;
    int trtRgbBinding_{-1};
    int trtIrBinding_{-1};
    int trtOutputBinding_{-1};

    std::vector<std::string> inputNameStrings_;
    std::vector<std::string> outputNameStrings_;
    std::vector<const char*> inputNames_;
    std::vector<const char*> outputNames_;
    bool isDynamicInputShape_{false};
    ONNXTensorElementDataType onnxInputDataType_{ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};
    cv::Size2f inputImageShape_;
};
}  // namespace ros_yolo
