#include "ros_yolo/detector.h"

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <fstream>
#include <string>

#ifndef TWO_STREAM_ENGINE_PATH
#define TWO_STREAM_ENGINE_PATH ""
#endif

#ifndef TWO_STREAM_ONNX_PATH
#define TWO_STREAM_ONNX_PATH ""
#endif

TEST(YoloDetectorIntegrationTest, RunsProvidedOnnxModel)
{
    const std::string onnxPath = TWO_STREAM_ONNX_PATH;
    std::ifstream onnxFile(onnxPath, std::ios::binary);
    if (onnxPath.empty() || !onnxFile.good())
    {
        GTEST_SKIP() << "ONNX model is not present: " << onnxPath;
    }

    ros_yolo::YoloDetector detector(onnxPath, true, cv::Size(640, 640));
    cv::Mat rgb(640, 640, CV_8UC3, cv::Scalar::all(0));
    cv::Mat ir(640, 640, CV_8UC3, cv::Scalar::all(0));

    try
    {
        const auto detections = detector.detect(rgb, ir, 0.30F, 0.40F);
        (void)detections;
    }
    catch (const std::exception& error)
    {
        FAIL() << "ONNX inference failed: " << error.what();
    }
}

TEST(YoloDetectorTensorRtTest, LoadsProvidedEngineWhenRuntimeIsAvailable)
{
    const std::string enginePath = TWO_STREAM_ENGINE_PATH;
    std::ifstream engineFile(enginePath, std::ios::binary);
    if (enginePath.empty() || !engineFile.good())
    {
        GTEST_SKIP() << "TensorRT engine is not present: " << enginePath;
    }

    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
    {
        GTEST_SKIP() << "No CUDA device is available";
    }

    ros_yolo::YoloDetector detector(enginePath, true, cv::Size(640, 640));
    cv::Mat rgb(640, 640, CV_8UC3, cv::Scalar::all(0));
    cv::Mat ir(640, 640, CV_8UC3, cv::Scalar::all(0));

    EXPECT_NO_THROW({
        const auto detections = detector.detect(rgb, ir, 0.30F, 0.40F);
        (void)detections;
    });
}
