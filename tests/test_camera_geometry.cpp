#include "ros_yolo/camera_geometry.h"

#include <gtest/gtest.h>

namespace
{
void expectVecNear(const cv::Vec3d& actual, const cv::Vec3d& expected)
{
    constexpr double kTol = 1e-12;
    EXPECT_NEAR(actual[0], expected[0], kTol);
    EXPECT_NEAR(actual[1], expected[1], kTol);
    EXPECT_NEAR(actual[2], expected[2], kTol);
}

void expectMatNear(const cv::Matx33d& actual, const cv::Matx33d& expected)
{
    constexpr double kTol = 1e-12;
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            EXPECT_NEAR(actual(r, c), expected(r, c), kTol);
        }
    }
}
}  // namespace

TEST(CameraGeometryTest, InvertsBodyCameraTransform)
{
    const cv::Matx33d rotation(0.0, -1.0, 0.0,
                               1.0, 0.0, 0.0,
                               0.0, 0.0, 1.0);
    const cv::Vec3d translation(1.0, 2.0, 3.0);
    const ros_yolo::CameraTransform bodyCamera(rotation, translation);

    const ros_yolo::CameraTransform cameraBody = ros_yolo::inverseTransform(bodyCamera);

    expectMatNear(cameraBody.rotation, rotation.t());
    expectVecNear(cameraBody.translation, cv::Vec3d(-2.0, 1.0, -3.0));
}

TEST(CameraGeometryTest, DerivesLegacyRgbToIrTransformFromTemporaryBodyCameraPoses)
{
    const ros_yolo::CameraTransform bodyRgb(cv::Matx33d::eye(), cv::Vec3d(0.0, 0.0, 0.0));
    const ros_yolo::CameraTransform bodyIr(cv::Matx33d::eye(), cv::Vec3d(0.036, -0.03, 0.009));

    const ros_yolo::CameraTransform irBody = ros_yolo::inverseTransform(bodyIr);
    const ros_yolo::CameraTransform irRgb = ros_yolo::composeTransforms(irBody, bodyRgb);

    expectMatNear(irRgb.rotation, cv::Matx33d::eye());
    expectVecNear(irRgb.translation, cv::Vec3d(-0.036, 0.03, -0.009));
}

TEST(CameraGeometryTest, TransformsPointIntoBodyFrame)
{
    const ros_yolo::CameraTransform bodyRgb(cv::Matx33d::eye(), cv::Vec3d(0.1, -0.2, 0.3));
    const cv::Point3f pointRgb(1.0F, 2.0F, 3.0F);

    const cv::Point3f pointBody = ros_yolo::transformPoint(bodyRgb, pointRgb);

    EXPECT_NEAR(pointBody.x, 1.1F, 1e-6F);
    EXPECT_NEAR(pointBody.y, 1.8F, 1e-6F);
    EXPECT_NEAR(pointBody.z, 3.3F, 1e-6F);
}
