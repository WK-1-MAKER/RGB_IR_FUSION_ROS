#include "ros_yolo/detector.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <array>
#include <cmath>

namespace ros_yolo
{
namespace
{
float sigmoidf(const float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

std::array<cv::Size2f, 3> anchorsForStride(const int stride)
{
    if (stride <= 8)
    {
        return {cv::Size2f(10.f, 13.f), cv::Size2f(16.f, 30.f), cv::Size2f(33.f, 23.f)};
    }
    if (stride <= 16)
    {
        return {cv::Size2f(30.f, 61.f), cv::Size2f(62.f, 45.f), cv::Size2f(59.f, 119.f)};
    }
    return {cv::Size2f(116.f, 90.f), cv::Size2f(156.f, 198.f), cv::Size2f(373.f, 326.f)};
}
}  // namespace

YoloDetector::YoloDetector(const std::string& modelPath,
                           const bool& isGPU,
                           const cv::Size& inputSize)
{
    env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
    sessionOptions = Ort::SessionOptions();

    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption;

    if (isGPU && (cudaAvailable == availableProviders.end()))
    {
        std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        std::cout << "Inference device: CPU" << std::endl;
    }
    else if (isGPU && (cudaAvailable != availableProviders.end()))
    {
        std::cout << "Inference device: GPU" << std::endl;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    }
    else
    {
        std::cout << "Inference device: CPU" << std::endl;
    }

#ifdef _WIN32
    std::wstring wModelPath = utils::charToWstring(modelPath.c_str());
    session = Ort::Session(env, wModelPath.c_str(), sessionOptions);
#else
    session = Ort::Session(env, modelPath.c_str(), sessionOptions);
#endif

    Ort::AllocatorWithDefaultOptions allocator;

    const size_t inputCount = session.GetInputCount();
    const size_t outputCount = session.GetOutputCount();
    if (inputCount < 2)
    {
        throw std::runtime_error("Dual-modal model required: ONNX input count must be >= 2.");
    }
    if (outputCount == 0)
    {
        throw std::runtime_error("Invalid ONNX model: no output tensor found.");
    }

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    std::vector<int64_t> inputTensorShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
    isDynamicInputShape = false;

    if (inputTensorShape[2] == -1 && inputTensorShape[3] == -1)
    {
        std::cout << "Dynamic input shape" << std::endl;
        isDynamicInputShape = true;
    }

    for (auto shape : inputTensorShape)
    {
        std::cout << "Input shape: " << shape << std::endl;
    }

    inputNameStrings.clear();
    outputNameStrings.clear();
    inputNames.clear();
    outputNames.clear();
    inputNameStrings.reserve(inputCount);
    outputNameStrings.reserve(outputCount);
    inputNames.reserve(inputCount);
    outputNames.reserve(outputCount);

    for (size_t i = 0; i < inputCount; ++i)
    {
        auto inputName = session.GetInputNameAllocated(i, allocator);
        inputNameStrings.emplace_back(inputName.get());
        inputNames.push_back(inputNameStrings.back().c_str());
        std::cout << "Input[" << i << "] name: " << inputNames.back() << std::endl;
    }

    for (size_t i = 0; i < outputCount; ++i)
    {
        auto outputName = session.GetOutputNameAllocated(i, allocator);
        outputNameStrings.emplace_back(outputName.get());
        outputNames.push_back(outputNameStrings.back().c_str());
        std::cout << "Output[" << i << "] name: " << outputNames.back() << std::endl;
    }

    inputImageShape = cv::Size2f(inputSize);
}

void YoloDetector::getBestClassInfo(std::vector<float>::iterator it,
                                    const int& numClasses,
                                    float& bestConf,
                                    int& bestClassId)
{
    bestClassId = 5;
    bestConf = 0.0f;

    for (int i = 5; i < numClasses + 5; i++)
    {
        if (it[i] > bestConf)
        {
            bestConf = it[i];
            bestClassId = i - 5;
        }
    }
}

void YoloDetector::preprocessing(const cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape)
{
    cv::Mat resizedImage;
    cv::Mat floatImage;
    cv::cvtColor(image, resizedImage, cv::COLOR_BGR2RGB);
    utils::letterbox(resizedImage,
                     resizedImage,
                     inputImageShape,
                     cv::Scalar(114, 114, 114),
                     isDynamicInputShape,
                     false,
                     true,
                     32);

    inputTensorShape[2] = resizedImage.rows;
    inputTensorShape[3] = resizedImage.cols;

    resizedImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);
    blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];
    cv::Size floatImageSize{floatImage.cols, floatImage.rows};

    std::vector<cv::Mat> chw(floatImage.channels());
    for (int i = 0; i < floatImage.channels(); ++i)
    {
        chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
    }
    cv::split(floatImage, chw);
}

std::vector<Detection> YoloDetector::postprocessing(const cv::Size& resizedImageShape,
                                                    const cv::Size& originalImageShape,
                                                    std::vector<Ort::Value>& outputTensors,
                                                    const float& confThreshold,
                                                    const float& iouThreshold)
{
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;

    std::vector<std::vector<int64_t>> allShapes;
    allShapes.reserve(outputTensors.size());
    bool hasDecodedOutput = false;
    for (auto& tensor : outputTensors)
    {
        auto shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
        allShapes.push_back(shape);
        if (shape.size() == 2 || shape.size() == 3)
        {
            hasDecodedOutput = true;
        }
    }

    for (size_t oi = 0; oi < outputTensors.size(); ++oi)
    {
        const std::vector<int64_t>& outputShape = allShapes[oi];
        /* std::cout << "Output[" << oi << "] shape:";
        for (const auto dim : outputShape)
        {
            std::cout << " " << dim;
        }
        std::cout << std::endl; */

        auto* rawOutput = outputTensors[oi].GetTensorData<float>();
        const size_t count = outputTensors[oi].GetTensorTypeAndShapeInfo().GetElementCount();
        std::vector<float> output(rawOutput, rawOutput + count);

        if (outputShape.size() == 5 && hasDecodedOutput)
        {
            continue;
        }

        if (outputShape.size() == 2 || outputShape.size() == 3)
        {
            int rows = 0;
            int cols = 0;
            if (outputShape.size() == 2)
            {
                rows = static_cast<int>(outputShape[0]);
                cols = static_cast<int>(outputShape[1]);
            }
            else
            {
                if (outputShape[0] != 1)
                {
                    throw std::runtime_error("Only batch=1 is supported for ROS postprocessing.");
                }

                const int dim1 = static_cast<int>(outputShape[1]);
                const int dim2 = static_cast<int>(outputShape[2]);
                if (dim2 > 5)
                {
                    rows = dim1;
                    cols = dim2;
                }
                else if (dim1 > 5)
                {
                    rows = dim2;
                    cols = dim1;
                    std::vector<float> transposed(static_cast<size_t>(rows) * static_cast<size_t>(cols));
                    for (int r = 0; r < rows; ++r)
                    {
                        for (int c = 0; c < cols; ++c)
                        {
                            transposed[static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c)] =
                                output[static_cast<size_t>(c) * static_cast<size_t>(rows) + static_cast<size_t>(r)];
                        }
                    }
                    output.swap(transposed);
                }
                else
                {
                    continue;
                }
            }

            if (rows <= 0 || cols <= 5)
            {
                continue;
            }

            const int numClasses = cols - 5;
            const int elementsInBatch = rows * cols;
            for (auto it = output.begin(); it != output.begin() + elementsInBatch; it += cols)
            {
                const float objConf = it[4];
                float clsProb;
                int classId;
                getBestClassInfo(it, numClasses, clsProb, classId);
                const float confidence = objConf * clsProb;
                if (confidence <= confThreshold)
                {
                    continue;
                }

                const int centerX = static_cast<int>(it[0]);
                const int centerY = static_cast<int>(it[1]);
                const int width = static_cast<int>(it[2]);
                const int height = static_cast<int>(it[3]);
                const int left = centerX - width / 2;
                const int top = centerY - height / 2;

                boxes.emplace_back(left, top, width, height);
                confs.emplace_back(confidence);
                classIds.emplace_back(classId);
            }
            continue;
        }

        if (outputShape.size() == 5)
        {
            if (outputShape[0] != 1)
            {
                continue;
            }

            const int d1 = static_cast<int>(outputShape[1]);
            const int d2 = static_cast<int>(outputShape[2]);
            const int d3 = static_cast<int>(outputShape[3]);
            const int d4 = static_cast<int>(outputShape[4]);

            if (d4 > 5)
            {
                const int anchors = d1;
                const int ny = d2;
                const int nx = d3;
                const int no = d4;
                const int numClasses = no - 5;
                if (numClasses <= 0)
                {
                    continue;
                }
                const int stride = std::max(1, resizedImageShape.width / std::max(1, nx));
                const auto anchorSet = anchorsForStride(stride);

                for (int a = 0; a < anchors; ++a)
                {
                    const cv::Size2f anchor = anchorSet[std::min(a, 2)];
                    for (int gy = 0; gy < ny; ++gy)
                    {
                        for (int gx = 0; gx < nx; ++gx)
                        {
                            const size_t base = static_cast<size_t>((((a * ny) + gy) * nx + gx) * no);
                            const float px = sigmoidf(output[base + 0]);
                            const float py = sigmoidf(output[base + 1]);
                            const float pw = sigmoidf(output[base + 2]);
                            const float ph = sigmoidf(output[base + 3]);
                            const float objConf = sigmoidf(output[base + 4]);

                            float clsProb = 0.0f;
                            int classId = 0;
                            for (int c = 0; c < numClasses; ++c)
                            {
                                const float prob = sigmoidf(output[base + 5 + c]);
                                if (prob > clsProb)
                                {
                                    clsProb = prob;
                                    classId = c;
                                }
                            }

                            const float confidence = objConf * clsProb;
                            if (confidence <= confThreshold)
                            {
                                continue;
                            }

                            const float centerX = (px * 2.0f - 0.5f + static_cast<float>(gx)) * static_cast<float>(stride);
                            const float centerY = (py * 2.0f - 0.5f + static_cast<float>(gy)) * static_cast<float>(stride);
                            const float width = (pw * 2.0f) * (pw * 2.0f) * anchor.width;
                            const float height = (ph * 2.0f) * (ph * 2.0f) * anchor.height;
                            const int left = static_cast<int>(centerX - width * 0.5f);
                            const int top = static_cast<int>(centerY - height * 0.5f);

                            boxes.emplace_back(left,
                                               top,
                                               std::max(1, static_cast<int>(width)),
                                               std::max(1, static_cast<int>(height)));
                            confs.emplace_back(confidence);
                            classIds.emplace_back(classId);
                        }
                    }
                }
                continue;
            }

            if (d2 > 5)
            {
                const int anchors = d1;
                const int no = d2;
                const int ny = d3;
                const int nx = d4;
                const int numClasses = no - 5;
                if (numClasses <= 0)
                {
                    continue;
                }
                const int stride = std::max(1, resizedImageShape.width / std::max(1, nx));
                const auto anchorSet = anchorsForStride(stride);

                for (int a = 0; a < anchors; ++a)
                {
                    const cv::Size2f anchor = anchorSet[std::min(a, 2)];
                    for (int gy = 0; gy < ny; ++gy)
                    {
                        for (int gx = 0; gx < nx; ++gx)
                        {
                            auto valueAt = [&](int c) {
                                const size_t idx = static_cast<size_t>((((a * no) + c) * ny + gy) * nx + gx);
                                return output[idx];
                            };

                            const float px = sigmoidf(valueAt(0));
                            const float py = sigmoidf(valueAt(1));
                            const float pw = sigmoidf(valueAt(2));
                            const float ph = sigmoidf(valueAt(3));
                            const float objConf = sigmoidf(valueAt(4));

                            float clsProb = 0.0f;
                            int classId = 0;
                            for (int c = 0; c < numClasses; ++c)
                            {
                                const float prob = sigmoidf(valueAt(5 + c));
                                if (prob > clsProb)
                                {
                                    clsProb = prob;
                                    classId = c;
                                }
                            }

                            const float confidence = objConf * clsProb;
                            if (confidence <= confThreshold)
                            {
                                continue;
                            }

                            const float centerX = (px * 2.0f - 0.5f + static_cast<float>(gx)) * static_cast<float>(stride);
                            const float centerY = (py * 2.0f - 0.5f + static_cast<float>(gy)) * static_cast<float>(stride);
                            const float width = (pw * 2.0f) * (pw * 2.0f) * anchor.width;
                            const float height = (ph * 2.0f) * (ph * 2.0f) * anchor.height;
                            const int left = static_cast<int>(centerX - width * 0.5f);
                            const int top = static_cast<int>(centerY - height * 0.5f);

                            boxes.emplace_back(left,
                                               top,
                                               std::max(1, static_cast<int>(width)),
                                               std::max(1, static_cast<int>(height)));
                            confs.emplace_back(confidence);
                            classIds.emplace_back(classId);
                        }
                    }
                }
            }
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, confThreshold, iouThreshold, indices);

    std::vector<Detection> detections;
    detections.reserve(indices.size());
    for (int idx : indices)
    {
        Detection det;
        det.box = cv::Rect(boxes[idx]);
        utils::scaleCoords(resizedImageShape, det.box, originalImageShape);

        det.conf = confs[idx];
        det.classId = classIds[idx];
        detections.emplace_back(det);
    }

    return detections;
}

std::vector<Detection> YoloDetector::detect(cv::Mat& rgbImage,
                                            cv::Mat& irImage,
                                            const float& confThreshold,
                                            const float& iouThreshold)
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

    std::vector<Ort::Value> inputTensors;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                                             OrtMemType::OrtMemTypeDefault);

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

    std::vector<Ort::Value> outputTensors = session.Run(Ort::RunOptions{nullptr},
                                                        inputNames.data(),
                                                        inputTensors.data(),
                                                        inputTensors.size(),
                                                        outputNames.data(),
                                                        outputNames.size());

    cv::Size resizedShape(static_cast<int>(rgbInputShape[3]), static_cast<int>(rgbInputShape[2]));
    std::vector<Detection> result = postprocessing(resizedShape,
                                                   rgbImage.size(),
                                                   outputTensors,
                                                   confThreshold,
                                                   iouThreshold);

    delete[] rgbBlob;
    delete[] irBlob;
    return result;
}
}  // namespace ros_yolo
