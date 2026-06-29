#include "ros_yolo/yolov8_decode.h"

#include <gtest/gtest.h>

namespace ros_yolo
{
TEST(Yolov8DecodeTest, DecodesChannelFirstOutputAndAppliesNms)
{
    const std::vector<float> output = {
        320.0f, 322.0f,
        320.0f, 322.0f,
        120.0f, 118.0f,
        200.0f, 198.0f,
        0.95f, 0.70f
    };
    const std::vector<int64_t> shape = {1, 5, 2};

    const std::vector<Detection> detections = decodeYolov8Detections(
        output,
        shape,
        cv::Size(640, 640),
        cv::Size(640, 640),
        0.25f,
        0.50f
    );

    ASSERT_EQ(detections.size(), 1u);
    EXPECT_EQ(detections[0].classId, 0);
    EXPECT_GT(detections[0].conf, 0.90f);
    EXPECT_EQ(detections[0].box.x, 260);
    EXPECT_EQ(detections[0].box.y, 220);
    EXPECT_EQ(detections[0].box.width, 120);
    EXPECT_EQ(detections[0].box.height, 200);
}

TEST(Yolov8DecodeTest, AcceptsChannelLastOutput)
{
    const std::vector<float> output = {
        320.0f, 320.0f, 120.0f, 200.0f, 0.95f,
        100.0f, 120.0f, 40.0f, 60.0f, 0.80f
    };
    const std::vector<int64_t> shape = {1, 2, 5};

    const std::vector<Detection> detections = decodeYolov8Detections(
        output,
        shape,
        cv::Size(640, 640),
        cv::Size(640, 640),
        0.25f,
        0.50f
    );

    ASSERT_EQ(detections.size(), 2u);
    EXPECT_GT(detections[0].conf, 0.0f);
    EXPECT_GT(detections[1].conf, 0.0f);
}

TEST(Yolov8DecodeTest, FiltersOutLowConfidenceRows)
{
    const std::vector<float> output = {
        320.0f, 320.0f, 120.0f, 200.0f, 0.20f,
        100.0f, 120.0f, 40.0f, 60.0f, 0.24f
    };
    const std::vector<int64_t> shape = {1, 2, 5};

    const std::vector<Detection> detections = decodeYolov8Detections(
        output,
        shape,
        cv::Size(640, 640),
        cv::Size(640, 640),
        0.25f,
        0.50f
    );

    EXPECT_TRUE(detections.empty());
}

TEST(Yolov8DecodeTest, DecodesTransposedMultiClassLayoutWhenBothDimsLookPlausible)
{
    const std::vector<float> output = {
        320.0f, 330.0f, 120.0f, 200.0f, 0.10f, 0.05f, 0.20f,
        100.0f, 200.0f, 40.0f, 20.0f, 0.10f, 0.05f, 0.20f,
        500.0f, 300.0f, 70.0f, 10.0f, 0.10f, 0.05f, 0.20f,
        50.0f, 60.0f, 90.0f, 10.0f, 0.10f, 0.05f, 0.20f,
        0.10f, 0.05f, 0.20f, 0.20f, 0.10f, 0.05f, 0.20f,
        0.95f, 0.10f, 0.15f, 0.05f, 0.10f, 0.05f, 0.20f
    };
    const std::vector<int64_t> shape = {1, 6, 7};

    EXPECT_NO_THROW({
        const std::vector<Detection> detections = decodeYolov8Detections(
            output,
            shape,
            cv::Size(640, 640),
            cv::Size(640, 640),
            0.25f,
            0.50f
        );

        ASSERT_EQ(detections.size(), 1u);
        EXPECT_EQ(detections[0].classId, 1);
        EXPECT_GT(detections[0].conf, 0.90f);
    });
}

TEST(Yolov8DecodeTest, ThrowsOnMismatchedBufferSize)
{
    const std::vector<float> output = {
        320.0f, 320.0f, 120.0f, 200.0f
    };
    const std::vector<int64_t> shape = {1, 2, 5};

    EXPECT_THROW(
        decodeYolov8Detections(
            output,
            shape,
            cv::Size(640, 640),
            cv::Size(640, 640),
            0.25f,
            0.50f
        ),
        std::runtime_error
    );
}

TEST(Yolov8DecodeTest, ThrowsOnRankTwoShape)
{
    const std::vector<float> output = {
        320.0f, 320.0f, 120.0f, 200.0f, 0.95f
    };
    const std::vector<int64_t> shape = {1, 5};

    EXPECT_THROW(
        decodeYolov8Detections(
            output,
            shape,
            cv::Size(640, 640),
            cv::Size(640, 640),
            0.25f,
            0.50f
        ),
        std::runtime_error
    );
}
}  // namespace ros_yolo
