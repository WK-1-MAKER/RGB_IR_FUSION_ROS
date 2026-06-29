#include "ros_yolo/detector.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "ros_yolo/yolov8_decode.h"

namespace ros_yolo
{
std::vector<Detection> decodeCurrentYolov8Tensor(const Ort::Value& tensor,
                                                 const cv::Size& resizedImageShape,
                                                 const cv::Size& originalImageShape,
                                                 float confThreshold,
                                                 float iouThreshold)
{
    const std::vector<int64_t> shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 3)
    {
        throw std::runtime_error("Current YOLOv8 deployment expects a rank-3 output tensor.");
    }

    const size_t count = tensor.GetTensorTypeAndShapeInfo().GetElementCount();
    const float* rawOutput = tensor.GetTensorData<float>();
    std::vector<float> output(rawOutput, rawOutput + count);
    return decodeYolov8Detections(output,
                                  shape,
                                  resizedImageShape,
                                  originalImageShape,
                                  confThreshold,
                                  iouThreshold);
}

YoloDetector::YoloDetector(const std::string& modelPath, bool isGPU, const cv::Size& inputSize)
{
    env_ = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
    sessionOptions_ = Ort::SessionOptions();
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    const auto cudaAvailable = std::find(availableProviders.begin(),
                                         availableProviders.end(),
                                         "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption;

    if (isGPU && cudaAvailable == availableProviders.end())
    {
        std::cout << "GPU is not supported by this ONNX Runtime build. Fallback to CPU." << std::endl;
        std::cout << "Inference device: CPU" << std::endl;
    }
    else if (isGPU)
    {
        std::cout << "Inference device: GPU" << std::endl;
        sessionOptions_.AppendExecutionProvider_CUDA(cudaOption);
    }
    else
    {
        std::cout << "Inference device: CPU" << std::endl;
    }

#ifdef _WIN32
    std::wstring wModelPath = utils::charToWstring(modelPath.c_str());
    session_ = Ort::Session(env_, wModelPath.c_str(), sessionOptions_);
#else
    session_ = Ort::Session(env_, modelPath.c_str(), sessionOptions_);
#endif

    Ort::AllocatorWithDefaultOptions allocator;
    const size_t inputCount = session_.GetInputCount();
    const size_t outputCount = session_.GetOutputCount();
    if (inputCount < 2)
    {
        throw std::runtime_error("Dual-modal model required: ONNX input count must be >= 2.");
    }
    if (outputCount == 0)
    {
        throw std::runtime_error("Invalid ONNX model: no output tensor found.");
    }

    const std::vector<int64_t> inputTensorShape =
        session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (inputTensorShape.size() != 4)
    {
        throw std::runtime_error("Expected 4D input tensor for RGB branch.");
    }

    isDynamicInputShape_ = inputTensorShape[2] == -1 || inputTensorShape[3] == -1;
    for (auto dim : inputTensorShape)
    {
        std::cout << "Input shape: " << dim << std::endl;
    }

    inputNameStrings_.reserve(inputCount);
    outputNameStrings_.reserve(outputCount);
    inputNames_.reserve(inputCount);
    outputNames_.reserve(outputCount);

    for (size_t i = 0; i < inputCount; ++i)
    {
        auto inputName = session_.GetInputNameAllocated(i, allocator);
        inputNameStrings_.emplace_back(inputName.get());
        inputNames_.push_back(inputNameStrings_.back().c_str());
        std::cout << "Input[" << i << "] name: " << inputNames_.back() << std::endl;
    }

    for (size_t i = 0; i < outputCount; ++i)
    {
        auto outputName = session_.GetOutputNameAllocated(i, allocator);
        outputNameStrings_.emplace_back(outputName.get());
        outputNames_.push_back(outputNameStrings_.back().c_str());
        std::cout << "Output[" << i << "] name: " << outputNames_.back() << std::endl;
    }

    inputImageShape_ = cv::Size2f(inputSize);
}

void YoloDetector::preprocessing(const cv::Mat& image,
                                 float*& blob,
                                 std::vector<int64_t>& inputTensorShape)
{
    cv::Mat resizedImage;
    cv::Mat floatImage;
    cv::cvtColor(image, resizedImage, cv::COLOR_BGR2RGB);
    utils::letterbox(resizedImage,
                     resizedImage,
                     inputImageShape_,
                     cv::Scalar(114, 114, 114),
                     isDynamicInputShape_,
                     false,
                     true,
                     32);

    inputTensorShape[2] = resizedImage.rows;
    inputTensorShape[3] = resizedImage.cols;

    resizedImage.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);
    blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];
    const cv::Size floatImageSize{floatImage.cols, floatImage.rows};

    std::vector<cv::Mat> chw(floatImage.channels());
    for (int i = 0; i < floatImage.channels(); ++i)
    {
        chw[i] = cv::Mat(floatImageSize,
                         CV_32FC1,
                         blob + i * floatImageSize.width * floatImageSize.height);
    }
    cv::split(floatImage, chw);
}

std::vector<Detection> YoloDetector::postprocessing(const cv::Size& resizedImageShape,
                                                    const cv::Size& originalImageShape,
                                                    std::vector<Ort::Value>& outputTensors,
                                                    float confThreshold,
                                                    float iouThreshold)
{
    if (outputTensors.size() != 1)
    {
        throw std::runtime_error("Current YOLOv8 deployment expects exactly one output tensor.");
    }

    return decodeCurrentYolov8Tensor(outputTensors.front(),
                                     resizedImageShape,
                                     originalImageShape,
                                     confThreshold,
                                     iouThreshold);
}

std::vector<Detection> YoloDetector::detect(cv::Mat& rgbImage,
                                            cv::Mat& irImage,
                                            float confThreshold,
                                            float iouThreshold)
{
    if (rgbImage.empty() || irImage.empty())
    {
        return {};
    }

    if (rgbImage.size() != irImage.size())
    {
        cv::resize(irImage, irImage, rgbImage.size());
    }

    float* rgbBlob = nullptr;
    float* irBlob = nullptr;
    std::vector<int64_t> rgbInputShape{1, 3, -1, -1};
    std::vector<int64_t> irInputShape{1, 3, -1, -1};
    preprocessing(rgbImage, rgbBlob, rgbInputShape);
    preprocessing(irImage, irBlob, irInputShape);

    const size_t rgbInputTensorSize = utils::vectorProduct(rgbInputShape);
    const size_t irInputTensorSize = utils::vectorProduct(irInputShape);
    std::vector<float> rgbInputTensorValues(rgbBlob, rgbBlob + rgbInputTensorSize);
    std::vector<float> irInputTensorValues(irBlob, irBlob + irInputTensorSize);

    delete[] rgbBlob;
    delete[] irBlob;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> inputTensors;
    inputTensors.reserve(2);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo,
                                                           rgbInputTensorValues.data(),
                                                           rgbInputTensorSize,
                                                           rgbInputShape.data(),
                                                           rgbInputShape.size()));
    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo,
                                                           irInputTensorValues.data(),
                                                           irInputTensorSize,
                                                           irInputShape.data(),
                                                           irInputShape.size()));

    std::vector<Ort::Value> outputTensors =
        session_.Run(Ort::RunOptions{nullptr},
                     inputNames_.data(),
                     inputTensors.data(),
                     inputTensors.size(),
                     outputNames_.data(),
                     outputNames_.size());

    const cv::Size resizedShape(static_cast<int>(rgbInputShape[3]), static_cast<int>(rgbInputShape[2]));
    return postprocessing(resizedShape,
                          rgbImage.size(),
                          outputTensors,
                          confThreshold,
                          iouThreshold);
}
}  // namespace ros_yolo
