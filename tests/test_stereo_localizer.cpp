#include "ros_yolo/stereo_localizer.h"
#include <gtest/gtest.h>

#include <cmath>

TEST(StereoLocalizerTest, TriangulatesAndTransformsMatchedPointIntoBody)
{
    const cv::Matx33d k(100, 0, 100, 0, 100, 100, 0, 0, 1);
    ros_yolo::StereoLocalizer localizer(k, k, cv::Matx33d::eye(), cv::Vec3d(-1, 0, 0),
                                        cv::Matx33d::eye(), cv::Vec3d(1, 2, 3));
    std::vector<cv::KeyPoint> rgb{{100.f, 100.f, 8.f}, {100.f, 100.f, 8.f}};
    std::vector<cv::KeyPoint> ir{{80.f, 100.f, 8.f}, {80.f, 120.f, 8.f}};
    std::vector<cv::DMatch> matches{{0, 0, 0.f}, {1, 1, 1.f}};
    const std::vector<ros_yolo::Detection> detections{{cv::Rect(90, 90, 20, 20), 1.f, 0}};

    const auto result = localizer.localize(rgb, ir, matches, detections);

    ASSERT_EQ(result.filteredMatches.size(), 1u);
    ASSERT_EQ(result.pointsBody.size(), 1u);
    EXPECT_NEAR(result.pointsBody[0].x, 1.f, 1e-5f);
    EXPECT_NEAR(result.pointsBody[0].y, 2.f, 1e-5f);
    EXPECT_NEAR(result.pointsBody[0].z, 8.f, 1e-4f);
}

TEST(StereoLocalizerTest, RectificationAndFallbackPreserveCameraToBodyTransform)
{
    const cv::Matx33d k(400, 0, 320, 0, 410, 240, 0, 0, 1);
    cv::Mat rotation;
    cv::Rodrigues(cv::Vec3d(0.01, 0.04, -0.02), rotation);
    const cv::Matx33d rIrRgb(rotation);
    const cv::Vec3d tIrRgb(-0.5, 0.04, 0.01);
    // Camera z maps to body x, camera x maps to body -y.
    const cv::Matx33d rBodyRgb(0, 0, 1, -1, 0, 0, 0, -1, 0);
    const cv::Vec3d tBodyRgb(0.2, -0.1, 0.3);
    const cv::Vec3d rgbPoint(0.3, -0.2, 5.0);
    const cv::Vec3d irPoint = rIrRgb * rgbPoint + tIrRgb;
    const cv::Vec3d rgbPixel = k * rgbPoint;
    const cv::Vec3d irPixel = k * irPoint;
    const std::vector<cv::KeyPoint> rgb{{float(rgbPixel[0] / rgbPixel[2]), float(rgbPixel[1] / rgbPixel[2]), 8.f}};
    const std::vector<cv::KeyPoint> ir{{float(irPixel[0] / irPixel[2]), float(irPixel[1] / irPixel[2]), 8.f}};
    const std::vector<cv::DMatch> matches{{0, 0, 0.f}};
    const std::vector<ros_yolo::Detection> boxes{{cv::Rect(0, 0, 640, 480), 1.f, 0}};
    const cv::Vec3d expected = rBodyRgb * rgbPoint + tBodyRgb;
    ros_yolo::StereoLocalizer localizer(k, k, rIrRgb, tIrRgb, rBodyRgb, tBodyRgb);

    for (int state = 0; state < 3; ++state)
    {
        if (state == 1) localizer.disableRectification();
        else localizer.updateRectification(cv::Size(640, 480));
        const auto result = localizer.localize(rgb, ir, matches, boxes);
        ASSERT_EQ(result.filteredMatches.size(), 1u);
        ASSERT_EQ(result.pointsBody.size(), 1u);
        EXPECT_NEAR(result.pointsBody[0].x, expected[0], 1e-4);
        EXPECT_NEAR(result.pointsBody[0].y, expected[1], 1e-4);
        EXPECT_NEAR(result.pointsBody[0].z, expected[2], 1e-4);
    }
}

TEST(StereoLocalizerTest, AssociatesWithNearestBoxAndKeepsMissingEstimatesAsNan)
{
    const cv::Matx33d k(100, 0, 100, 0, 100, 100, 0, 0, 1);
    ros_yolo::StereoLocalizer localizer(k, k, cv::Matx33d::eye(), cv::Vec3d(-1, 0, 0),
                                       cv::Matx33d::eye(), cv::Vec3d(0, 0, 0));
    localizer.updateRectification(cv::Size(200, 200));
    const std::vector<cv::KeyPoint> rgb{{100.f, 100.f, 8.f}};
    const std::vector<cv::KeyPoint> ir{{80.f, 100.f, 8.f}};
    const std::vector<cv::DMatch> matches{{0, 0, 0.f}, {-1, 0, 0.f}, {0, 10, 0.f}};
    const std::vector<ros_yolo::Detection> boxes{
        {cv::Rect(50, 50, 100, 100), 1.f, 0},
        {cv::Rect(90, 90, 100, 100), 1.f, 0}};
    const auto result = localizer.localize(rgb, ir, matches, boxes);
    ASSERT_EQ(result.filteredMatches.size(), 1u);
    ASSERT_EQ(result.pointsBody.size(), 2u);
    EXPECT_NEAR(result.pointsBody[0].z, 5.f, 1e-4f);
    EXPECT_TRUE(std::isnan(result.pointsBody[1].x));
    const auto empty = localizer.localize({}, {}, {}, boxes);
    ASSERT_EQ(empty.pointsBody.size(), 2u);
    EXPECT_TRUE(std::isnan(empty.pointsBody[0].x));
    EXPECT_TRUE(std::isnan(empty.pointsBody[1].x));
}

TEST(StereoLocalizerTest, RejectsPointsBehindCameraAndBeyondThirtyMetres)
{
    const cv::Matx33d k(100, 0, 100, 0, 100, 100, 0, 0, 1);
    ros_yolo::StereoLocalizer localizer(k, k, cv::Matx33d::eye(), cv::Vec3d(-1, 0, 0),
                                       cv::Matx33d::eye(), cv::Vec3d(0, 0, 0));
    const std::vector<cv::KeyPoint> rgb{{100.f, 100.f, 8.f}, {100.f, 100.f, 8.f}};
    const std::vector<cv::KeyPoint> ir{{120.f, 100.f, 8.f}, {98.f, 100.f, 8.f}};
    const std::vector<ros_yolo::Detection> boxes{{cv::Rect(90, 90, 20, 20), 1.f, 0}};
    const auto result = localizer.localize(rgb, ir, {{0, 0, 0.f}, {1, 1, 0.f}}, boxes);
    ASSERT_EQ(result.pointsBody.size(), 1u);
    EXPECT_TRUE(std::isnan(result.pointsBody[0].x));
}

TEST(StereoLocalizerTest, AggregatesSmallAndLargeGroupsAtKnownDepth)
{
    const cv::Matx33d k(100, 0, 100, 0, 100, 100, 0, 0, 1);
    ros_yolo::StereoLocalizer localizer(k, k, cv::Matx33d::eye(), cv::Vec3d(-1, 0, 0),
                                       cv::Matx33d::eye(), cv::Vec3d(0, 0, 0));
    const std::vector<ros_yolo::Detection> boxes{{cv::Rect(80, 80, 40, 40), 1.f, 0}};
    for (int count : {2, 3, 6, 8})
    {
        std::vector<cv::KeyPoint> rgb, ir;
        std::vector<cv::DMatch> matches;
        for (int i = 0; i < count; ++i)
        {
            const float x = 95.f + i;
            rgb.emplace_back(x, 100.f, 8.f);
            ir.emplace_back(x - 20.f, 100.f, 8.f);
            matches.emplace_back(i, i, 0.f);
        }
        const auto result = localizer.localize(rgb, ir, matches, boxes);
        ASSERT_EQ(result.pointsBody.size(), 1u);
        EXPECT_TRUE(std::isfinite(result.pointsBody[0].x));
        EXPECT_NEAR(result.pointsBody[0].y, 0.f, 1e-5f);
        EXPECT_NEAR(result.pointsBody[0].z, 5.f, 1e-4f);
    }
}
