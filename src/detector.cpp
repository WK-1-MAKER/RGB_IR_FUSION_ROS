#include "ros_yolo/detector.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "ros_yolo/yolov8_decode.h"

namespace ros_yolo
{
namespace
{
std::string lowerCopy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string dimsToString(const nvinfer1::Dims& dims)
{
    std::ostringstream stream;
    stream << "[";
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (i != 0)
        {
            stream << ",";
        }
        stream << dims.d[i];
    }
    stream << "]";
    return stream.str();
}

float halfBitsToFloat(uint16_t bits)
{
    const bool negative = (bits & 0x8000U) != 0;
    const uint16_t exponent = static_cast<uint16_t>((bits >> 10U) & 0x1FU);
    const uint16_t mantissa = static_cast<uint16_t>(bits & 0x03FFU);
    float value = 0.0F;
    if (exponent == 0)
    {
        value = std::ldexp(static_cast<float>(mantissa), -24);
    }
    else if (exponent == 0x1FU)
    {
        value = mantissa == 0 ? std::numeric_limits<float>::infinity()
                              : std::numeric_limits<float>::quiet_NaN();
    }
    else
    {
        value = std::ldexp(static_cast<float>(1024U + mantissa), static_cast<int>(exponent) - 25);
    }
    return negative ? -value : value;
}
}  // namespace

InferenceBackend inferBackendFromModelPath(const std::string& modelPath)
{
    const std::string lowerPath = lowerCopy(modelPath);
    if (lowerPath.size() >= 5 && lowerPath.compare(lowerPath.size() - 5, 5, ".onnx") == 0)
    {
        return InferenceBackend::ONNX;
    }
    if (lowerPath.size() >= 7 && lowerPath.compare(lowerPath.size() - 7, 7, ".engine") == 0)
    {
        return InferenceBackend::TENSORRT;
    }
    throw std::invalid_argument("Model path must end with .onnx or .engine: " + modelPath);
}

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

    const auto typeInfo = tensor.GetTensorTypeAndShapeInfo();
    const size_t count = typeInfo.GetElementCount();
    std::vector<float> output(count);
    if (typeInfo.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    {
        const float* rawOutput = tensor.GetTensorData<float>();
        std::copy(rawOutput, rawOutput + count, output.begin());
    }
    else if (typeInfo.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
    {
        const uint16_t* rawOutput = tensor.GetTensorData<uint16_t>();
        for (size_t i = 0; i < count; ++i)
        {
            output[i] = halfBitsToFloat(rawOutput[i]);
        }
    }
    else
    {
        throw std::runtime_error("Current YOLOv8 output tensor must be float32 or float16");
    }
    return decodeYolov8Detections(output,
                                  shape,
                                  resizedImageShape,
                                  originalImageShape,
                                  confThreshold,
                                  iouThreshold);
}

YoloDetector::YoloDetector(const std::string& modelPath, bool isGPU, const cv::Size& inputSize)
{
    backend_ = inferBackendFromModelPath(modelPath);
    inputImageShape_ = cv::Size2f(inputSize);
    if (backend_ == InferenceBackend::TENSORRT)
    {
        initializeTensorRt(modelPath);
        std::cout << "Inference backend: TensorRT" << std::endl;
        return;
    }

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
    onnxInputDataType_ = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType();
    if (onnxInputDataType_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
        onnxInputDataType_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
    {
        throw std::runtime_error("ONNX input tensors must be float32 or float16");
    }
    for (size_t i = 1; i < inputCount; ++i)
    {
        const auto inputType = session_.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
        if (inputType != onnxInputDataType_)
        {
            throw std::runtime_error("Both ONNX input tensors must use the same data type");
        }
    }
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

}

YoloDetector::~YoloDetector()
{
    if (trtStream_ != nullptr)
    {
        cudaStreamDestroy(trtStream_);
        trtStream_ = nullptr;
    }
}

void YoloDetector::initializeTensorRt(const std::string& modelPath)
{
    int deviceCount = 0;
    const cudaError_t deviceStatus = cudaGetDeviceCount(&deviceCount);
    if (deviceStatus != cudaSuccess || deviceCount == 0)
    {
        throw std::runtime_error("TensorRT backend requires a CUDA device");
    }

    std::ifstream file(modelPath, std::ios::binary | std::ios::ate);
    if (!file)
    {
        throw std::runtime_error("Cannot open TensorRT engine: " + modelPath);
    }
    const std::streamsize size = file.tellg();
    if (size <= 0)
    {
        throw std::runtime_error("TensorRT engine is empty: " + modelPath);
    }
    file.seekg(0, std::ios::beg);
    std::vector<char> bytes(static_cast<size_t>(size));
    if (!file.read(bytes.data(), size))
    {
        throw std::runtime_error("Cannot read TensorRT engine: " + modelPath);
    }

    trtRuntime_ = tensorrt_buffer::TensorRTUniquePtr<nvinfer1::IRuntime>{
        nvinfer1::createInferRuntime(tensorrt_log::gLogger)};
    if (!trtRuntime_)
    {
        throw std::runtime_error("TensorRT runtime creation failed");
    }

    trtEngine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
        trtRuntime_->deserializeCudaEngine(bytes.data(), bytes.size()));
    if (!trtEngine_)
    {
        throw std::runtime_error("TensorRT engine deserialization failed: " + modelPath);
    }

    trtContext_ = std::shared_ptr<nvinfer1::IExecutionContext>(
        trtEngine_->createExecutionContext());
    if (!trtContext_)
    {
        throw std::runtime_error("TensorRT execution context creation failed");
    }

    trtRgbBinding_ = trtEngine_->getBindingIndex("images_rgb");
    trtIrBinding_ = trtEngine_->getBindingIndex("images_ir");
    trtOutputBinding_ = trtEngine_->getBindingIndex("output");
    if (trtRgbBinding_ < 0 || trtIrBinding_ < 0 || trtOutputBinding_ < 0)
    {
        throw std::runtime_error("TensorRT engine must expose images_rgb, images_ir and output bindings");
    }
    if (!trtEngine_->bindingIsInput(trtRgbBinding_) || !trtEngine_->bindingIsInput(trtIrBinding_) ||
        trtEngine_->bindingIsInput(trtOutputBinding_))
    {
        throw std::runtime_error("TensorRT binding directions do not match the dual-input model");
    }

    auto configureInput = [&](int binding, const char* name) {
        const auto dataType = trtEngine_->getBindingDataType(binding);
        if (dataType != nvinfer1::DataType::kFLOAT && dataType != nvinfer1::DataType::kHALF)
        {
            throw std::runtime_error(std::string("TensorRT input binding must be FP32 or FP16: ") + name);
        }
        nvinfer1::Dims dims = trtEngine_->getBindingDimensions(binding);
        if (dims.nbDims != 4)
        {
            throw std::runtime_error(std::string("TensorRT input binding must be rank-4: ") + name +
                                     " has " + dimsToString(dims));
        }
        if (dims.d[0] == -1 || dims.d[2] == -1 || dims.d[3] == -1)
        {
            if (!trtContext_->setBindingDimensions(
                    binding,
                    nvinfer1::Dims4(1, 3, static_cast<int>(inputImageShape_.height),
                                    static_cast<int>(inputImageShape_.width))))
            {
                throw std::runtime_error(std::string("Cannot set TensorRT input dimensions: ") + name);
            }
            dims = trtContext_->getBindingDimensions(binding);
        }
        if (dims.d[0] != 1 || dims.d[1] != 3 || dims.d[2] <= 0 || dims.d[3] <= 0)
        {
            throw std::runtime_error(std::string("Unsupported TensorRT input dimensions for ") + name +
                                     ": " + dimsToString(dims));
        }
    };
    configureInput(trtRgbBinding_, "images_rgb");
    configureInput(trtIrBinding_, "images_ir");

    const auto outputDataType = trtEngine_->getBindingDataType(trtOutputBinding_);
    if (outputDataType != nvinfer1::DataType::kFLOAT && outputDataType != nvinfer1::DataType::kHALF)
    {
        throw std::runtime_error("TensorRT output binding must be FP32 or FP16");
    }
    const nvinfer1::Dims outputDims = trtContext_->getBindingDimensions(trtOutputBinding_);
    if (outputDims.nbDims != 3)
    {
        throw std::runtime_error("TensorRT output binding must be rank-3: " + dimsToString(outputDims));
    }
    trtOutputShape_.clear();
    for (int i = 0; i < outputDims.nbDims; ++i)
    {
        if (outputDims.d[i] <= 0)
        {
            throw std::runtime_error("TensorRT output binding has unresolved dimensions: " +
                                     dimsToString(outputDims));
        }
        trtOutputShape_.push_back(outputDims.d[i]);
    }

    try
    {
        trtBuffers_ = std::make_unique<tensorrt_buffer::BufferManager>(trtEngine_, 0, trtContext_.get());
    }
    catch (const std::exception& error)
    {
        throw std::runtime_error(std::string("TensorRT buffer allocation failed: ") + error.what());
    }
    if (cudaStreamCreateWithFlags(&trtStream_, cudaStreamNonBlocking) != cudaSuccess)
    {
        throw std::runtime_error("TensorRT CUDA stream creation failed");
    }
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

    if (backend_ == InferenceBackend::TENSORRT)
    {
        const cv::Size resizedShape(static_cast<int>(rgbInputShape[3]),
                                     static_cast<int>(rgbInputShape[2]));
        return detectTensorRt(rgbInputTensorValues,
                              irInputTensorValues,
                              resizedShape,
                              rgbImage.size(),
                              confThreshold,
                              iouThreshold);
    }

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> inputTensors;
    inputTensors.reserve(2);
    if (onnxInputDataType_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    {
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
    }
    else
    {
        std::vector<half_float::half> rgbHalf(rgbInputTensorSize);
        std::vector<half_float::half> irHalf(irInputTensorSize);
        for (size_t i = 0; i < rgbInputTensorSize; ++i)
        {
            rgbHalf[i] = half_float::half(rgbInputTensorValues[i]);
        }
        for (size_t i = 0; i < irInputTensorSize; ++i)
        {
            irHalf[i] = half_float::half(irInputTensorValues[i]);
        }
        inputTensors.push_back(Ort::Value::CreateTensor(memoryInfo,
                                                         rgbHalf.data(),
                                                         rgbHalf.size() * sizeof(half_float::half),
                                                         rgbInputShape.data(),
                                                         rgbInputShape.size(),
                                                         ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
        inputTensors.push_back(Ort::Value::CreateTensor(memoryInfo,
                                                         irHalf.data(),
                                                         irHalf.size() * sizeof(half_float::half),
                                                         irInputShape.data(),
                                                         irInputShape.size(),
                                                         ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
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

std::vector<Detection> YoloDetector::detectTensorRt(const std::vector<float>& rgbInput,
                                                     const std::vector<float>& irInput,
                                                     const cv::Size& resizedImageShape,
                                                     const cv::Size& originalImageShape,
                                                     float confThreshold,
                                                     float iouThreshold)
{
    if (!trtEngine_ || !trtContext_ || !trtBuffers_ || trtStream_ == nullptr)
    {
        throw std::runtime_error("TensorRT backend is not initialized");
    }

    auto copyInput = [&](const std::string& name, const std::vector<float>& values) {
        const int binding = trtEngine_->getBindingIndex(name.c_str());
        const size_t elementCount = values.size();
        const size_t bufferBytes = trtBuffers_->size(name);
        if (bufferBytes == tensorrt_buffer::BufferManager::kINVALID_SIZE_VALUE)
        {
            throw std::runtime_error("TensorRT input buffer is missing: " + name);
        }
        const auto type = trtEngine_->getBindingDataType(binding);
        const size_t elementSize = type == nvinfer1::DataType::kHALF ? sizeof(half_float::half) : sizeof(float);
        if (bufferBytes != elementCount * elementSize)
        {
            throw std::runtime_error("TensorRT input size mismatch for " + name);
        }

        void* hostBuffer = trtBuffers_->getHostBuffer(name);
        if (type == nvinfer1::DataType::kHALF)
        {
            auto* halfValues = static_cast<half_float::half*>(hostBuffer);
            for (size_t i = 0; i < elementCount; ++i)
            {
                halfValues[i] = half_float::half(values[i]);
            }
        }
        else
        {
            std::copy(values.begin(), values.end(), static_cast<float*>(hostBuffer));
        }

        const cudaError_t status = cudaMemcpyAsync(trtBuffers_->getDeviceBuffer(name),
                                                   hostBuffer,
                                                   bufferBytes,
                                                   cudaMemcpyHostToDevice,
                                                   trtStream_);
        if (status != cudaSuccess)
        {
            throw std::runtime_error(std::string("TensorRT input copy failed: ") + cudaGetErrorString(status));
        }
    };

    copyInput("images_rgb", rgbInput);
    copyInput("images_ir", irInput);

    auto& bindings = trtBuffers_->getDeviceBindings();
    if (!trtContext_->enqueueV2(bindings.data(), trtStream_, nullptr))
    {
        throw std::runtime_error("TensorRT enqueueV2 failed");
    }

    const size_t outputBytes = trtBuffers_->size("output");
    const auto outputType = trtEngine_->getBindingDataType(trtOutputBinding_);
    const size_t outputElementSize = outputType == nvinfer1::DataType::kHALF ? sizeof(half_float::half) : sizeof(float);
    const size_t outputElements = utils::vectorProduct(trtOutputShape_);
    if (outputBytes != outputElements * outputElementSize)
    {
        throw std::runtime_error("TensorRT output size does not match the reported output shape");
    }
    void* outputHostBuffer = trtBuffers_->getHostBuffer("output");
    const cudaError_t outputStatus = cudaMemcpyAsync(outputHostBuffer,
                                                      trtBuffers_->getDeviceBuffer("output"),
                                                      outputBytes,
                                                      cudaMemcpyDeviceToHost,
                                                      trtStream_);
    if (outputStatus != cudaSuccess)
    {
        throw std::runtime_error(std::string("TensorRT output copy failed: ") + cudaGetErrorString(outputStatus));
    }
    const cudaError_t syncStatus = cudaStreamSynchronize(trtStream_);
    if (syncStatus != cudaSuccess)
    {
        throw std::runtime_error(std::string("TensorRT stream synchronization failed: ") +
                                 cudaGetErrorString(syncStatus));
    }

    std::vector<float> output(outputElements);
    if (outputType == nvinfer1::DataType::kHALF)
    {
        const auto* halfOutput = static_cast<const half_float::half*>(outputHostBuffer);
        for (size_t i = 0; i < outputElements; ++i)
        {
            output[i] = static_cast<float>(halfOutput[i]);
        }
    }
    else
    {
        std::copy(static_cast<const float*>(outputHostBuffer),
                  static_cast<const float*>(outputHostBuffer) + outputElements,
                  output.begin());
    }

    return decodeYolov8Detections(output,
                                  trtOutputShape_,
                                  resizedImageShape,
                                  originalImageShape,
                                  confThreshold,
                                  iouThreshold);
}
}  // namespace ros_yolo
