#include "ros_yolo/camera_geometry.h"
#include "ros_yolo/frustum_geometry.h"

#include <gtest/gtest.h>

#include <limits>
#include <random>

namespace
{
cv::Point2d projectPixel(const cv::Matx34d& projection, const cv::Vec3d& point)
{
    const cv::Vec3d projected = projection * cv::Vec4d(point[0], point[1], point[2], 1.0);
    return cv::Point2d(projected[0] / projected[2], projected[1] / projected[2]);
}

double reprojectionCost(const cv::Matx34d& p1, const cv::Matx34d& p2,
                        const cv::Vec3d& point,
                        const cv::Point2d& observed1, const cv::Point2d& observed2)
{
    const cv::Point2d error1 = projectPixel(p1, point) - observed1;
    const cv::Point2d error2 = projectPixel(p2, point) - observed2;
    return error1.dot(error1) + error2.dot(error2);
}

const cv::Matx34d kTestP1(600.0, 0.0, 320.0, 0.0,
                         0.0, 600.0, 240.0, 0.0,
                         0.0, 0.0, 1.0, 0.0);
const cv::Matx34d kTestP2(600.0, 0.0, 320.0, -36.0,
                         0.0, 600.0, 240.0, 0.0,
                         0.0, 0.0, 1.0, 0.0);

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

TEST(CameraGeometryTest, ConvertsRgbLidarAndIrRgbCalibrationToBodyCameraPoses)
{
    const cv::Matx33d rRgbLidar(0.0, -1.0, 0.0,
                                1.0, 0.0, 0.0,
                                0.0, 0.0, 1.0);
    const cv::Vec3d tRgbLidar(1.0, 2.0, 3.0);
    const cv::Matx33d rIrRgb(1.0, 0.0, 0.0,
                             0.0, 0.0, -1.0,
                             0.0, 1.0, 0.0);
    const cv::Vec3d tIrRgb(-0.4, 0.5, 0.2);

    const ros_yolo::BodyCameraTransforms bodyCameras =
        ros_yolo::bodyCameraTransformsFromRgbLidarAndIrRgb(
            ros_yolo::CameraTransform(rRgbLidar, tRgbLidar),
            ros_yolo::CameraTransform(rIrRgb, tIrRgb));

    expectMatNear(bodyCameras.bodyRgb.rotation,
                  cv::Matx33d(0.0, 1.0, 0.0,
                              -1.0, 0.0, 0.0,
                              0.0, 0.0, 1.0));
    expectVecNear(bodyCameras.bodyRgb.translation, cv::Vec3d(-2.0, 1.0, -3.0));
    expectMatNear(bodyCameras.bodyIr.rotation,
                  cv::Matx33d(0.0, 0.0, 1.0,
                              -1.0, 0.0, 0.0,
                              0.0, -1.0, 0.0));
    expectVecNear(bodyCameras.bodyIr.translation, cv::Vec3d(-2.2, 0.6, -2.5));

    const ros_yolo::CameraTransform rgbBody =
        ros_yolo::inverseTransform(bodyCameras.bodyRgb);
    const ros_yolo::CameraTransform irBody =
        ros_yolo::inverseTransform(bodyCameras.bodyIr);
    const ros_yolo::CameraTransform recoveredIrRgb =
        ros_yolo::composeTransforms(irBody, bodyCameras.bodyRgb);
    expectMatNear(recoveredIrRgb.rotation, rIrRgb);
    expectVecNear(recoveredIrRgb.translation, tIrRgb);
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

TEST(CameraGeometryTest, BuildsAndTestsRgbDetectionFrustumInBodyFrame)
{
    const cv::Matx33d intrinsic(100.0, 0.0, 100.0,
                                0.0, 100.0, 100.0,
                                0.0, 0.0, 1.0);
    ros_yolo::BodyFrustum frustum;
    ASSERT_TRUE(ros_yolo::buildBodyFrustum(
        cv::Rect(90, 90, 20, 20), intrinsic, cv::Matx33d::eye(), cv::Vec3d(1.0, 2.0, 3.0), frustum));

    EXPECT_TRUE(frustum.contains(cv::Vec3d(1.0, 2.0, 8.0)));
    EXPECT_FALSE(frustum.contains(cv::Vec3d(4.0, 2.0, 8.0)));
    EXPECT_FALSE(frustum.contains(cv::Vec3d(1.0, 2.0, 2.0)));
}

TEST(CameraGeometryTest, RotatesFrustumRaysWithCameraExtrinsic)
{
    const cv::Matx33d intrinsic(100.0, 0.0, 100.0,
                                0.0, 100.0, 100.0,
                                0.0, 0.0, 1.0);
    const cv::Matx33d rotateCameraZToBodyX(0.0, 0.0, 1.0,
                                           -1.0, 0.0, 0.0,
                                           0.0, -1.0, 0.0);
    ros_yolo::BodyFrustum frustum;
    ASSERT_TRUE(ros_yolo::buildBodyFrustum(
        cv::Rect(90, 90, 20, 20), intrinsic, rotateCameraZToBodyX, cv::Vec3d(0.0, 0.0, 0.0), frustum));

    EXPECT_TRUE(frustum.contains(cv::Vec3d(5.0, 0.0, 0.0)));
    EXPECT_FALSE(frustum.contains(cv::Vec3d(-5.0, 0.0, 0.0)));
}

TEST(CameraGeometryTest, RefinementNeverIncreasesReprojectionError)
{
    const cv::Matx33d k(600.0, 0.0, 320.0,
                       0.0, 600.0, 240.0,
                       0.0, 0.0, 1.0);
    const cv::Matx34d p1(k(0, 0), k(0, 1), k(0, 2), 0.0,
                         k(1, 0), k(1, 1), k(1, 2), 0.0,
                         k(2, 0), k(2, 1), k(2, 2), 0.0);
    const cv::Matx34d p2(k(0, 0), k(0, 1), k(0, 2), -36.0,
                         k(1, 0), k(1, 1), k(1, 2), 0.0,
                         k(2, 0), k(2, 1), k(2, 2), 0.0);
    const cv::Vec3d initialPoint(0.26, -0.49, 2.405);
    const cv::Point2d observed1(227.9113617, 73.07600403);
    const cv::Point2d observed2(216.0564117, 75.00697327);

    const cv::Vec3d refined = ros_yolo::refineTriangulatedPoint(
        initialPoint, p1, p2, observed1, observed2);

    const auto reprojectionCost = [&](const cv::Vec3d& point) {
        const cv::Vec4d homogeneous(point[0], point[1], point[2], 1.0);
        const cv::Vec3d projected1 = p1 * homogeneous;
        const cv::Vec3d projected2 = p2 * homogeneous;
        const double dx1 = projected1[0] / projected1[2] - observed1.x;
        const double dy1 = projected1[1] / projected1[2] - observed1.y;
        const double dx2 = projected2[0] / projected2[2] - observed2.x;
        const double dy2 = projected2[1] / projected2[2] - observed2.y;
        return dx1 * dx1 + dy1 * dy1 + dx2 * dx2 + dy2 * dy2;
    };

    EXPECT_LT(reprojectionCost(refined), reprojectionCost(initialPoint));
}

TEST(CameraGeometryTest, RefinementConvergesWhenFullStepOvershoots)
{
    const cv::Matx34d p1(600.0, 0.0, 320.0, 0.0,
                         0.0, 600.0, 240.0, 0.0,
                         0.0, 0.0, 1.0, 0.0);
    const cv::Matx34d p2(600.0, 0.0, 320.0, -36.0,
                         0.0, 600.0, 240.0, 0.0,
                         0.0, 0.0, 1.0, 0.0);
    // The first undamped Gauss-Newton step crosses the projection singularity.
    const cv::Vec3d refined = ros_yolo::refineTriangulatedPoint(
        cv::Vec3d(0.0, 0.0, 20.0), p1, p2,
        cv::Point2d(320.0, 240.0), cv::Point2d(302.0, 240.0), 20);

    EXPECT_NEAR(refined[0], 0.0, 1e-9);
    EXPECT_NEAR(refined[1], 0.0, 1e-9);
    EXPECT_NEAR(refined[2], 2.0, 1e-8);
}

TEST(CameraGeometryTest, RefinementPreservesExactSolutionAndZeroIterationInput)
{
    const cv::Vec3d exact(0.2, -0.3, 25.0);
    const cv::Point2d observed1 = projectPixel(kTestP1, exact);
    const cv::Point2d observed2 = projectPixel(kTestP2, exact);
    const cv::Vec3d refined = ros_yolo::refineTriangulatedPoint(
        exact, kTestP1, kTestP2, observed1, observed2);
    EXPECT_EQ(refined, exact);

    const cv::Vec3d initial(0.2, -0.3, 20.0);
    EXPECT_EQ(ros_yolo::refineTriangulatedPoint(
        initial, kTestP1, kTestP2, observed1, observed2, 0), initial);
}

TEST(CameraGeometryTest, RefinementRetainsDoublePrecisionAtLongRange)
{
    const cv::Vec3d exact(0.2, -0.3, 25.0);
    const cv::Vec3d initial = exact + cv::Vec3d(1e-7, -1e-7, 1e-7);
    const cv::Vec3d refined = ros_yolo::refineTriangulatedPoint(
        initial, kTestP1, kTestP2,
        projectPixel(kTestP1, exact), projectPixel(kTestP2, exact));

    EXPECT_LT(cv::norm(refined - exact), 1e-10);
}

TEST(CameraGeometryTest, RefinementKeepsInitialPointForInvalidProjection)
{
    for (const cv::Vec3d& initial : {cv::Vec3d(0.2, -0.3, 0.0),
                                    cv::Vec3d(0.2, -0.3, -1.0)})
    {
        const cv::Vec3d refined = ros_yolo::refineTriangulatedPoint(
            initial, kTestP1, kTestP2,
            cv::Point2d(320.0, 240.0), cv::Point2d(302.0, 240.0));
        EXPECT_EQ(refined, initial);
    }

    const cv::Vec3d initial(0.2, -0.3, 2.0);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(ros_yolo::refineTriangulatedPoint(
        initial, kTestP1, kTestP2,
        cv::Point2d(nan, 240.0), cv::Point2d(302.0, 240.0)), initial);
}

TEST(CameraGeometryTest, RefinementDoesNotWorsenNoisyStereoTriangulations)
{
    const cv::Matx33d k1(606.97900390625, 0.0, 320.3143615722656,
                         0.0, 607.1318359375, 247.97427368164062,
                         0.0, 0.0, 1.0);
    const cv::Matx33d k2(593.946114205786, 0.0, 322.242307522658,
                         0.0, 592.545920338569, 264.298408236226,
                         0.0, 0.0, 1.0);
    const cv::Matx34d p1 = k1 * cv::Matx34d(1.0, 0.0, 0.0, 0.0,
                                            0.0, 1.0, 0.0, 0.0,
                                            0.0, 0.0, 1.0, 0.0);
    std::mt19937 random(4242);
    std::normal_distribution<double> noise(0.0, 0.8);
    for (bool rectified : {false, true})
    {
        cv::Matx33d rotation;
        cv::Rodrigues(cv::Vec3d(0.03, -0.05, 0.1), rotation);
        const cv::Matx34d p2 = k2 * cv::Matx34d(
            rotation(0, 0), rotation(0, 1), rotation(0, 2), -0.036,
            rotation(1, 0), rotation(1, 1), rotation(1, 2), 0.03,
            rotation(2, 0), rotation(2, 1), rotation(2, 2), -0.009);
        cv::Matx34d testP1 = p1;
        cv::Matx34d testP2 = p2;
        if (rectified)
        {
            cv::Mat r1, r2, projection1, projection2, q;
            cv::stereoRectify(k1, cv::Mat(), k2, cv::Mat(), cv::Size(640, 480),
                              rotation, cv::Vec3d(-0.036, 0.03, -0.009),
                              r1, r2, projection1, projection2, q);
            testP1 = cv::Matx34d(projection1);
            testP2 = cv::Matx34d(projection2);
        }

        for (double z : {1.0, 3.0, 5.0, 15.0, 25.0})
        {
            for (int sample = 0; sample < 350; ++sample)
            {
                const cv::Vec3d truth((sample % 7 - 3) * z * 0.08,
                                      (sample % 5 - 2) * z * 0.08, z);
                const cv::Point2d observed1 = projectPixel(testP1, truth) +
                    cv::Point2d(noise(random), noise(random));
                const cv::Point2d observed2 = projectPixel(testP2, truth) +
                    cv::Point2d(noise(random), noise(random));
                cv::Mat homogeneous;
                cv::triangulatePoints(testP1, testP2,
                    std::vector<cv::Point2d>{observed1},
                    std::vector<cv::Point2d>{observed2}, homogeneous);
                const cv::Vec3d initial(homogeneous.at<double>(0) / homogeneous.at<double>(3),
                                        homogeneous.at<double>(1) / homogeneous.at<double>(3),
                                        homogeneous.at<double>(2) / homogeneous.at<double>(3));
                const cv::Vec3d refined = ros_yolo::refineTriangulatedPoint(
                    initial, testP1, testP2, observed1, observed2);

                SCOPED_TRACE(::testing::Message() << "rectified=" << rectified
                             << " z=" << z << " sample=" << sample);
                EXPECT_LE(reprojectionCost(testP1, testP2, refined, observed1, observed2),
                          reprojectionCost(testP1, testP2, initial, observed1, observed2));
            }
        }
    }
}

TEST(CameraGeometryTest, FrustumMatchesCameraProjectionWithRotatedTranslatedPose)
{
    const cv::Matx33d intrinsic(600, 0, 320, 0, 610, 240, 0, 0, 1);
    cv::Matx33d rotation;
    cv::Rodrigues(cv::Vec3d(0.2, -0.5, 0.1), rotation);
    const cv::Vec3d translation(2, -1, 0.7);
    const cv::Rect box(260, 180, 120, 100);
    ros_yolo::BodyFrustum frustum;
    ASSERT_TRUE(ros_yolo::buildBodyFrustum(box, intrinsic, rotation, translation, frustum));
    expectVecNear(frustum.origin, translation);
    for (double u : {250., 260., 280., 379., 380., 390.})
    {
        for (double v : {170., 180., 220., 279., 280., 290.})
        {
            for (double depth : {-5., 0.5, 50.})
            {
                const cv::Vec3d cameraPoint = intrinsic.inv() * cv::Vec3d(u, v, 1) * depth;
                const bool expected = depth > 0 && u >= 260 && u <= 380 && v >= 180 && v <= 280;
                EXPECT_EQ(frustum.contains(rotation * cameraPoint + translation), expected);
            }
        }
    }
    const cv::Point2d corners[] = {{260, 180}, {380, 180}, {380, 280}, {260, 280}};
    for (size_t i = 0; i < frustum.rays.size(); ++i)
    {
        EXPECT_NEAR(cv::norm(frustum.rays[i]), 1., 1e-12);
        const cv::Vec3d cameraRay = rotation.t() * frustum.rays[i];
        const cv::Vec3d pixel = intrinsic * cameraRay;
        EXPECT_NEAR(pixel[0] / pixel[2], corners[i].x, 1e-9);
        EXPECT_NEAR(pixel[1] / pixel[2], corners[i].y, 1e-9);
        EXPECT_TRUE(frustum.contains(translation + 10 * frustum.rays[i]));
    }
    EXPECT_FALSE(frustum.contains(translation));
    EXPECT_FALSE(frustum.contains(cv::Vec3d(std::numeric_limits<double>::infinity(), 0, 0)));
}

TEST(CameraGeometryTest, RejectsEmptyDetectionFrustum)
{
    ros_yolo::BodyFrustum frustum;
    EXPECT_FALSE(ros_yolo::buildBodyFrustum(cv::Rect(0, 0, 0, 10), cv::Matx33d::eye(),
                                           cv::Matx33d::eye(), cv::Vec3d(0, 0, 0), frustum));
}
