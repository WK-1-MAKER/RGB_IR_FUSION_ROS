#include "ros_yolo/detector.h"

#include <gtest/gtest.h>

#include <stdexcept>

namespace ros_yolo
{
namespace
{
Ort::Value makeTensor(const std::vector<int64_t>& shape, const std::vector<float>& data)
{
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor<float>(memoryInfo,
                                           const_cast<float*>(data.data()),
                                           data.size(),
                                           shape.data(),
                                           shape.size());
}

Ort::Value makeHalfTensor(const std::vector<int64_t>& shape,
                          const std::vector<half_float::half>& data)
{
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor(memoryInfo,
                                    const_cast<half_float::half*>(data.data()),
                                    data.size() * sizeof(half_float::half),
                                    shape.data(),
                                    shape.size(),
                                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
}
}  // namespace

TEST(YoloDetectorBackendTest, SelectsOnnxFromExtension)
{
    EXPECT_EQ(inferBackendFromModelPath("/tmp/aifi-best.onnx"), InferenceBackend::ONNX);
    EXPECT_EQ(inferBackendFromModelPath("/tmp/AIFI-BEST.ONNX"), InferenceBackend::ONNX);
}

TEST(YoloDetectorBackendTest, SelectsTensorRtFromExtension)
{
    EXPECT_EQ(inferBackendFromModelPath("/tmp/aifi-best.engine"), InferenceBackend::TENSORRT);
    EXPECT_EQ(inferBackendFromModelPath("/tmp/AIFI-BEST.ENGINE"), InferenceBackend::TENSORRT);
}

TEST(YoloDetectorBackendTest, RejectsUnsupportedExtension)
{
    EXPECT_THROW(inferBackendFromModelPath("/tmp/aifi-best.pt"), std::invalid_argument);
    EXPECT_THROW(inferBackendFromModelPath("/tmp/aifi-best"), std::invalid_argument);
}

TEST(YoloDetectorPostprocessTest, DecodesCurrentRankThreeYolov8Output)
{
    std::vector<float> output = {
        320.0f, 320.0f, 120.0f, 200.0f, 0.95f
    };
    const Ort::Value tensor = makeTensor({1, 1, 5}, output);

    const std::vector<Detection> detections = decodeCurrentYolov8Tensor(
        tensor,
        cv::Size(640, 640),
        cv::Size(640, 640),
        0.25f,
        0.50f
    );

    ASSERT_EQ(detections.size(), 1u);
    EXPECT_EQ(detections[0].classId, 0);
    EXPECT_GT(detections[0].conf, 0.90f);
}

TEST(YoloDetectorPostprocessTest, DecodesFloat16Yolov8Output)
{
    std::vector<half_float::half> output = {
        half_float::half(320.0f),
        half_float::half(320.0f),
        half_float::half(120.0f),
        half_float::half(200.0f),
        half_float::half(0.95f)
    };
    const Ort::Value tensor = makeHalfTensor({1, 1, 5}, output);

    const std::vector<Detection> detections = decodeCurrentYolov8Tensor(
        tensor,
        cv::Size(640, 640),
        cv::Size(640, 640),
        0.25f,
        0.50f
    );

    ASSERT_EQ(detections.size(), 1u);
    EXPECT_EQ(detections[0].classId, 0);
    EXPECT_GT(detections[0].conf, 0.90f);
}

TEST(YoloDetectorPostprocessTest, RejectsLegacyRankFiveOutput)
{
    std::vector<float> output(1 * 3 * 2 * 2 * 6, 0.0f);
    const Ort::Value tensor = makeTensor({1, 3, 2, 2, 6}, output);

    EXPECT_THROW(
        decodeCurrentYolov8Tensor(
            tensor,
            cv::Size(640, 640),
            cv::Size(640, 640),
            0.25f,
            0.50f
        ),
        std::runtime_error
    );
}
}  // namespace ros_yolo
