#include "ros_yolo/stereo_localizer.h"
#include "ros_yolo/camera_geometry.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace ros_yolo
{
cv::Point2f reprojection_error(const cv::Point3f& point3d,
                              const cv::Matx34d& P,
                              const cv::Point2f& observed);
namespace
{
struct TriangulatedSample
{
    cv::Point3f point;
    double reprojectionError;
};

std::vector<double> computeSoftmaxWeights(const std::vector<TriangulatedSample>& samples,
                                          const std::vector<int>& indices,
                                          double temperature = 1.0)
{
    std::vector<double> weights(indices.size(), 0.0);
    if (indices.empty())
    {
        return weights;
    }

    double maxLogit = -std::numeric_limits<double>::infinity();
    for (int idx : indices)
    {
        if (idx < 0 || idx >= static_cast<int>(samples.size()))
        {
            continue;
        }
        const double safeError = std::max(samples[static_cast<size_t>(idx)].reprojectionError, 1e-9);
        const double logit = -safeError / temperature;
        if (logit > maxLogit)
        {
            maxLogit = logit;
        }
    }

    double sumWeights = 0.0;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        const int idx = indices[i];
        if (idx < 0 || idx >= static_cast<int>(samples.size()))
        {
            continue;
        }
        const double safeError = std::max(samples[static_cast<size_t>(idx)].reprojectionError, 1e-9);
        weights[i] = std::exp((-safeError / temperature) - maxLogit);
        sumWeights += weights[i];
    }

    if (sumWeights > std::numeric_limits<double>::epsilon())
    {
        for (double& weight : weights)
        {
            weight /= sumWeights;
        }
    }

    return weights;
}

cv::Point3f weightedAverageFromIndices(const std::vector<TriangulatedSample>& samples,
                                       const std::vector<int>& indices,
                                       const std::vector<double>& weights,
                                       bool& ok)
{
    ok = false;
    if (indices.empty() || indices.size() != weights.size())
    {
        return cv::Point3f();
    }

    cv::Vec3d sum(0.0, 0.0, 0.0);
    double sumWeights = 0.0;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        const int idx = indices[i];
        if (idx < 0 || idx >= static_cast<int>(samples.size()))
        {
            continue;
        }
        const double w = weights[i];
        const cv::Point3f& point = samples[static_cast<size_t>(idx)].point;
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z) ||
            !std::isfinite(w) || w <= 0.0)
        {
            continue;
        }

        sum[0] += w * static_cast<double>(point.x);
        sum[1] += w * static_cast<double>(point.y);
        sum[2] += w * static_cast<double>(point.z);
        sumWeights += w;
    }

    if (sumWeights <= std::numeric_limits<double>::epsilon())
    {
        return cv::Point3f();
    }

    ok = true;
    return cv::Point3f(static_cast<float>(sum[0] / sumWeights),
                       static_cast<float>(sum[1] / sumWeights),
                       static_cast<float>(sum[2] / sumWeights));
}

std::vector<int> topKLowestErrorIndices(const std::vector<TriangulatedSample>& samples, int k)
{
    std::vector<int> indices(samples.size());
    for (size_t i = 0; i < samples.size(); ++i)
    {
        indices[i] = static_cast<int>(i);
    }

    std::sort(indices.begin(), indices.end(),
              [&samples](int lhs, int rhs)
              {
                  return samples[static_cast<size_t>(lhs)].reprojectionError <
                         samples[static_cast<size_t>(rhs)].reprojectionError;
              });

    if (k < static_cast<int>(indices.size()))
    {
        indices.resize(static_cast<size_t>(k));
    }
    return indices;
}

cv::Point3f huberIrlsCenter(const std::vector<TriangulatedSample>& samples,
                            int maxIterations,
                            double huberDelta,
                            bool& ok)
{
    ok = false;
    if (samples.empty())
    {
        return cv::Point3f();
    }

    std::vector<int> allIndices(samples.size());
    for (size_t i = 0; i < samples.size(); ++i)
    {
        allIndices[i] = static_cast<int>(i);
    }

    std::vector<double> baseWeights = computeSoftmaxWeights(samples, allIndices, 1.0);
    cv::Point3f center = weightedAverageFromIndices(samples, allIndices, baseWeights, ok);
    if (!ok)
    {
        return cv::Point3f();
    }

    for (int iter = 0; iter < maxIterations; ++iter)
    {
        std::vector<double> combinedWeights(baseWeights.size(), 0.0);
        for (size_t i = 0; i < samples.size(); ++i)
        {
            const cv::Point3f& point = samples[i].point;
            const double dx = static_cast<double>(point.x - center.x);
            const double dy = static_cast<double>(point.y - center.y);
            const double dz = static_cast<double>(point.z - center.z);
            const double residual = std::sqrt(dx * dx + dy * dy + dz * dz);

            double huberWeight = 1.0;
            if (residual > huberDelta)
            {
                huberWeight = huberDelta / (residual + 1e-9);
            }
            combinedWeights[i] = baseWeights[i] * huberWeight;
        }

        bool avgOk = false;
        cv::Point3f newCenter = weightedAverageFromIndices(samples, allIndices, combinedWeights, avgOk);
        if (!avgOk)
        {
            break;
        }

        const double shiftX = static_cast<double>(newCenter.x - center.x);
        const double shiftY = static_cast<double>(newCenter.y - center.y);
        const double shiftZ = static_cast<double>(newCenter.z - center.z);
        const double shiftNorm = std::sqrt(shiftX * shiftX + shiftY * shiftY + shiftZ * shiftZ);

        center = newCenter;
        if (shiftNorm < 1e-3)
        {
            break;
        }
    }

    ok = true;
    return center;
}

double reprojectionErrorNorm(const cv::Point3f& point3d,
                             const cv::Matx34d& P1,
                             const cv::Matx34d& P2,
                             const cv::Point2f& observedRgb,
                             const cv::Point2f& observedIr)
{
    const cv::Point2f reprojErrorRgb = ::ros_yolo::reprojection_error(point3d, P1, observedRgb);
    const cv::Point2f reprojErrorIr = ::ros_yolo::reprojection_error(point3d, P2, observedIr);
    if (!std::isfinite(reprojErrorRgb.x) || !std::isfinite(reprojErrorRgb.y) ||
        !std::isfinite(reprojErrorIr.x) || !std::isfinite(reprojErrorIr.y))
    {
        return std::numeric_limits<double>::infinity();
    }

    const double exRgb = static_cast<double>(reprojErrorRgb.x);
    const double eyRgb = static_cast<double>(reprojErrorRgb.y);
    const double exIr = static_cast<double>(reprojErrorIr.x);
    const double eyIr = static_cast<double>(reprojErrorIr.y);
    return std::sqrt(exRgb * exRgb + eyRgb * eyRgb + exIr * exIr + eyIr * eyIr);
}
}  // namespace

StereoLocalizer::StereoLocalizer(const cv::Matx33d& kRgb, const cv::Matx33d& kIr,
                                 const cv::Matx33d& rIrRgb, const cv::Vec3d& tIrRgb,
                                 const cv::Matx33d& rBodyRgb, const cv::Vec3d& tBodyRgb)
    : kRgb_(kRgb), kIr_(kIr), rIrRgb_(rIrRgb), rBodyRgb_(rBodyRgb),
      tIrRgb_(tIrRgb), tBodyRgb_(tBodyRgb), kRgbTri_(kRgb), kIrTri_(kIr),
      rTri_(rIrRgb), tTri_(tIrRgb)
{
    const cv::Matx33d tCross(0.0, -tIrRgb_[2], tIrRgb_[1],
                            tIrRgb_[2], 0.0, -tIrRgb_[0],
                            -tIrRgb_[1], tIrRgb_[0], 0.0);
    const cv::Matx33d essential = tCross * rIrRgb_;
    fundamental_ = kIr_.inv(cv::DECOMP_SVD).t() * essential * kRgb_.inv(cv::DECOMP_SVD);
}

void StereoLocalizer::disableRectification()
{
    rectificationReady_ = false;
    kRgbTri_ = kRgb_;
    kIrTri_ = kIr_;
    rTri_ = rIrRgb_;
    tTri_ = tIrRgb_;
    rRectRgb_ = cv::Matx33d::eye();
}

void StereoLocalizer::updateRectification(const cv::Size& imageSize)
{
    if (imageSize.width <= 0 || imageSize.height <= 0)
    {
        return;
    }

    if (rectificationReady_ && rectificationImageSize_ == imageSize)
    {
        return;
    }

    const cv::Mat cameraMatrixRgb = (cv::Mat_<double>(3, 3)
        << kRgb_(0, 0), kRgb_(0, 1), kRgb_(0, 2),
           kRgb_(1, 0), kRgb_(1, 1), kRgb_(1, 2),
           kRgb_(2, 0), kRgb_(2, 1), kRgb_(2, 2));
    const cv::Mat cameraMatrixIr = (cv::Mat_<double>(3, 3)
        << kIr_(0, 0), kIr_(0, 1), kIr_(0, 2),
           kIr_(1, 0), kIr_(1, 1), kIr_(1, 2),
           kIr_(2, 0), kIr_(2, 1), kIr_(2, 2));
    const cv::Mat distCoeffsRgb = cv::Mat::zeros(1, 5, CV_64F);
    const cv::Mat distCoeffsIr = cv::Mat::zeros(1, 5, CV_64F);
    const cv::Mat rotation = (cv::Mat_<double>(3, 3)
        << rIrRgb_(0, 0), rIrRgb_(0, 1), rIrRgb_(0, 2),
           rIrRgb_(1, 0), rIrRgb_(1, 1), rIrRgb_(1, 2),
           rIrRgb_(2, 0), rIrRgb_(2, 1), rIrRgb_(2, 2));
    const cv::Mat translation = (cv::Mat_<double>(3, 1) << tIrRgb_(0), tIrRgb_(1), tIrRgb_(2));

    if (cameraMatrixRgb.empty() || cameraMatrixIr.empty() || rotation.empty() || translation.empty())
    {
        throw std::runtime_error("Rectification input matrix is empty.");
    }

    cv::Mat rRectRgb; //RGB相机旋转到矫正坐标系的旋转矩阵
    cv::Mat rRectIr;  //IR相机旋转到矫正坐标系的旋转矩阵
    cv::Mat pRectRgb; //矫正后的投影矩阵，3X4，K‘[R|t]
    cv::Mat pRectIr;  //矫正后的投影矩阵，3X4，K‘[R|t]
    cv::Mat q;

    cv::stereoRectify(cameraMatrixRgb,
                      distCoeffsRgb,
                      cameraMatrixIr,
                      distCoeffsIr,
                      imageSize,
                      rotation,
                      translation,
                      rRectRgb,
                      rRectIr,
                      pRectRgb,
                      pRectIr,
                      q,
                      cv::CALIB_ZERO_DISPARITY,
                      -1,
                      imageSize);

    if (rRectRgb.empty() || rRectIr.empty() || pRectRgb.empty() || pRectIr.empty())
    {
        throw std::runtime_error("stereoRectify returned empty matrices.");
    }
    //std::cout << rRectRgb << std::endl;

    cv::Matx33d leftK = cv::Matx33d::eye();
    cv::Matx33d rightK = cv::Matx33d::eye();
    cv::Vec3d rectifiedTranslation(0.0, 0.0, 0.0);

    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            leftK(r, c) = pRectRgb.at<double>(r, c);
            rightK(r, c) = pRectIr.at<double>(r, c);
        }
    }

    for (int r = 0; r < 3; ++r)
    {
        rectifiedTranslation[r] = pRectIr.at<double>(r, 3);
    }

    if (std::abs(rightK(2, 2)) > 1e-9)
    {
        rightK(2, 2) = 1.0;
    }

    cv::Matx33d rightKInv = rightK.inv(cv::DECOMP_SVD);
    cv::Matx33d rectifiedRotation = rightKInv * rightK;  //矫正后两个相机之间的旋转矩阵，理论上应该是单位矩阵，但实际计算中可能存在数值误差
    cv::Vec3d rectifiedT = rightKInv * rectifiedTranslation; //矫正后两个相机之间的平移向量，理论上应该是 [baseline, 0, 0]，但实际计算中可能存在数值误差
    std::cout << rectifiedT << std::endl;

    rRectRgb_ = cv::Matx33d(rRectRgb);
    rRectIr_ = cv::Matx33d(rRectIr);
    pRectRgb_ = cv::Matx34d(pRectRgb);
    pRectIr_ = cv::Matx34d(pRectIr);

    kRgbTri_ = leftK;
    kIrTri_ = rightK;
    rTri_ = rectifiedRotation;
    tTri_ = rectifiedT;

    rectificationImageSize_ = imageSize;
    rectificationReady_ = true;
}

// 矫正坐标系，将特征点从原始图像坐标系转换到校正图像坐标系，以便后续使用校正后的相机参数进行三角测量
void StereoLocalizer::rectifyMatchedPoints(const std::vector<cv::Point2f>& pointsRgb,
                                    const std::vector<cv::Point2f>& pointsIr,
                                    std::vector<cv::Point2f>& rectifiedRgb,
                                    std::vector<cv::Point2f>& rectifiedIr) const
{
    rectifiedRgb.clear();
    rectifiedIr.clear();

    if (!rectificationReady_)
    {
        // 原始匹配点来自已去畸变图像，与回退后的三角化参数对应。
        rectifiedRgb = pointsRgb;
        rectifiedIr = pointsIr;
        return;
    }

    if (pointsRgb.empty() || pointsIr.empty())
    {
        return;
    }

    if (pointsRgb.size() != pointsIr.size())
    {
        return;
    }

    const cv::Mat distCoeffsRgb = cv::Mat::zeros(1, 5, CV_64F);
    const cv::Mat distCoeffsIr = cv::Mat::zeros(1, 5, CV_64F);

    cv::undistortPoints(pointsRgb,
                        rectifiedRgb,
                        cv::Mat(kRgb_),
                        distCoeffsRgb,
                        cv::Mat(rRectRgb_),
                        cv::Mat(pRectRgb_));
    cv::undistortPoints(pointsIr,
                        rectifiedIr,
                        cv::Mat(kIr_),
                        distCoeffsIr,
                        cv::Mat(rRectIr_),
                        cv::Mat(pRectIr_));
}

void StereoLocalizer::filterMatches(
                              const std::vector<cv::KeyPoint>& points0,
                              const std::vector<cv::KeyPoint>& points1,
                              std::vector<cv::DMatch>& matches,
                              double threshold) const
{
    std::vector<cv::DMatch> corrected_matches;
    corrected_matches.reserve(matches.size());

    for (const auto& match : matches) {
        if (match.queryIdx < 0 || match.trainIdx < 0 ||
            match.queryIdx >= static_cast<int>(points0.size()) ||
            match.trainIdx >= static_cast<int>(points1.size())) {
            continue;
        }

        const cv::KeyPoint& kp0 = points0[match.queryIdx];
        const cv::KeyPoint& kp1 = points1[match.trainIdx];

        cv::Matx31d pt0(kp0.pt.x, kp0.pt.y, 1.0);
        cv::Matx31d pt1(kp1.pt.x, kp1.pt.y, 1.0);

        cv::Matx31d epipolar_line_ir = fundamental_ * pt0;
        cv::Matx31d epipolar_line_rgb = fundamental_.t() * pt1;
        float a1 = epipolar_line_ir(0), b1 = epipolar_line_ir(1), c1 = epipolar_line_ir(2);
        float a2 = epipolar_line_rgb(0), b2 = epipolar_line_rgb(1), c2 = epipolar_line_rgb(2);
        float norm1 = std::sqrt(a1 * a1 + b1 * b1);
        float norm2 = std::sqrt(a2 * a2 + b2 * b2);
        if (norm1 < 1e-6 || norm2 < 1e-6) {
            corrected_matches.push_back(match);
            continue;
        }
        double distance1 = std::abs(a1 * pt1(0) + b1 * pt1(1) + c1) / norm1;
        double distance2 = std::abs(a2 * pt0(0) + b2 * pt0(1) + c2) / norm2;

        if (distance1 < threshold && distance2 < threshold) {
            corrected_matches.push_back(match);
        }
    }
    matches.swap(corrected_matches);
}

cv::Point2d StereoLocalizer::normalizePixel(const cv::Matx33d& K, const cv::Point2f& pt) const
{
    cv::Matx31d normalized_pt = K.inv(cv::DECOMP_SVD) * cv::Matx31d(pt.x, pt.y, 1.0);
    return cv::Point2d(normalized_pt(0) / normalized_pt(2), normalized_pt(1) / normalized_pt(2));
}

cv::Vec3d StereoLocalizer::triangulateSvd(const cv::Matx34d& P1,
                                    const cv::Matx34d& P2,
                                    const cv::Point2d& ptn1,
                                    const cv::Point2d& ptn2) const
{
    const double u1 = static_cast<double>(ptn1.x);
    const double v1 = static_cast<double>(ptn1.y);
    const double u2 = static_cast<double>(ptn2.x);
    const double v2 = static_cast<double>(ptn2.y);

    cv::Matx44d A;
    A(0, 0) = u1 * P1(2, 0) - P1(0, 0);
    A(0, 1) = u1 * P1(2, 1) - P1(0, 1);
    A(0, 2) = u1 * P1(2, 2) - P1(0, 2);
    A(0, 3) = u1 * P1(2, 3) - P1(0, 3);

    A(1, 0) = v1 * P1(2, 0) - P1(1, 0);
    A(1, 1) = v1 * P1(2, 1) - P1(1, 1);
    A(1, 2) = v1 * P1(2, 2) - P1(1, 2);
    A(1, 3) = v1 * P1(2, 3) - P1(1, 3);

    A(2, 0) = u2 * P2(2, 0) - P2(0, 0);
    A(2, 1) = u2 * P2(2, 1) - P2(0, 1);
    A(2, 2) = u2 * P2(2, 2) - P2(0, 2);
    A(2, 3) = u2 * P2(2, 3) - P2(0, 3);

    A(3, 0) = v2 * P2(2, 0) - P2(1, 0);
    A(3, 1) = v2 * P2(2, 1) - P2(1, 1);
    A(3, 2) = v2 * P2(2, 2) - P2(1, 2);
    A(3, 3) = v2 * P2(2, 3) - P2(1, 3);

    cv::Mat w, u, vt;
    cv::SVD::compute(cv::Mat(A), w, u, vt, cv::SVD::FULL_UV);

    cv::Vec4d xh(vt.at<double>(3, 0),
                 vt.at<double>(3, 1),
                 vt.at<double>(3, 2),
                 vt.at<double>(3, 3));

    if (std::abs(xh[3]) < 1e-9) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return cv::Vec3d(nan, nan, nan);
    }

    return cv::Vec3d(xh[0] / xh[3],
                     xh[1] / xh[3],
                     xh[2] / xh[3]);
}

cv::Point2f reprojection_error(const cv::Point3f& point3d, const cv::Matx34d& P, const cv::Point2f& observed)
{
    cv::Matx41d homogeneous_point(point3d.x, point3d.y, point3d.z, 1.0);
    cv::Matx31d projected = P * homogeneous_point;
    if (std::abs(projected(2)) < 1e-9)
    {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        return cv::Point2f(nan, nan);
    }
    float u = static_cast<float>(projected(0) / projected(2));
    float v = static_cast<float>(projected(1) / projected(2));
    return cv::Point2f(u - observed.x, v - observed.y);
}

cv::Point3f StereoLocalizer::triangulate(const cv::Point2f& pt1, const cv::Point2f& pt2, int iters) const
{
    const cv::Matx34d P1_ext(1.0, 0.0, 0.0, 0.0,
                             0.0, 1.0, 0.0, 0.0,
                             0.0, 0.0, 1.0, 0.0);
    const cv::Matx34d P2_ext(rTri_(0, 0), rTri_(0, 1), rTri_(0, 2), tTri_(0),
                             rTri_(1, 0), rTri_(1, 1), rTri_(1, 2), tTri_(1),
                             rTri_(2, 0), rTri_(2, 1), rTri_(2, 2), tTri_(2));
    const cv::Point2d ptn1 = normalizePixel(kRgbTri_, pt1);
    const cv::Point2d ptn2 = normalizePixel(kIrTri_, pt2);
    const cv::Vec3d point3dSvd = triangulateSvd(P1_ext, P2_ext, ptn1, ptn2);
    if (!std::isfinite(point3dSvd[0]) || !std::isfinite(point3dSvd[1]) ||
        !std::isfinite(point3dSvd[2]))
    {
        return cv::Point3f(static_cast<float>(point3dSvd[0]),
                           static_cast<float>(point3dSvd[1]),
                           static_cast<float>(point3dSvd[2]));
    }

    const cv::Matx34d P1 = kRgbTri_ * P1_ext;
    const cv::Matx34d P2 = kIrTri_ * P2_ext;
    const cv::Vec3d point3dOptimized = refineTriangulatedPoint(
        point3dSvd,
        P1,
        P2,
        cv::Point2d(pt1.x, pt1.y),
        cv::Point2d(pt2.x, pt2.y),
        iters);
    return cv::Point3f(static_cast<float>(point3dOptimized[0]),
                       static_cast<float>(point3dOptimized[1]),
                       static_cast<float>(point3dOptimized[2]));
}

StereoLocalizationResult StereoLocalizer::localize(
    const std::vector<cv::KeyPoint>& pointsRgb,
    const std::vector<cv::KeyPoint>& pointsIr,
    const std::vector<cv::DMatch>& matches,
    const std::vector<Detection>& detectionsOnRgb) const
{
    StereoLocalizationResult result;
    result.filteredMatches = matches;
    filterMatches(pointsRgb, pointsIr, result.filteredMatches, 5.0);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    result.pointsBody.assign(detectionsOnRgb.size(), cv::Point3f(nan, nan, nan));
    std::vector<cv::Point2f> matchedRgbPoints;
    std::vector<cv::Point2f> matchedIrPoints;
    matchedRgbPoints.reserve(result.filteredMatches.size());
    matchedIrPoints.reserve(result.filteredMatches.size());
    for (const auto& match : result.filteredMatches)
    {
        if (match.queryIdx < 0 || match.trainIdx < 0 ||
            match.queryIdx >= static_cast<int>(pointsRgb.size()) ||
            match.trainIdx >= static_cast<int>(pointsIr.size()))
        {
            continue;
        }

        matchedRgbPoints.push_back(pointsRgb[match.queryIdx].pt);
        matchedIrPoints.push_back(pointsIr[match.trainIdx].pt);
    }

    std::vector<cv::Point2f> rectifiedRgbPoints;
    std::vector<cv::Point2f> rectifiedIrPoints;

    //matchedRgbPoints和matchedIrPoints是经过极线约束过滤后的匹配点对，rectifyMatchedPoints函数将它们从原始图像坐标系转换到校正图像坐标系，以便后续使用校正后的相机参数进行三角测量
    rectifyMatchedPoints(matchedRgbPoints, matchedIrPoints, rectifiedRgbPoints, rectifiedIrPoints);
    // 不再使用 swap 污染原有的 matchedRgbPoints ，以保证检测框验证使用的是原图特征点坐标

    const size_t pairCount = std::min(matchedRgbPoints.size(), matchedIrPoints.size());

    std::vector<std::vector<TriangulatedSample>> pointsPerDetection(detectionsOnRgb.size());
    const cv::Matx34d p1Ext(1.0, 0.0, 0.0, 0.0,
                            0.0, 1.0, 0.0, 0.0,
                            0.0, 0.0, 1.0, 0.0);
    const cv::Matx34d p2Ext(rTri_(0, 0), rTri_(0, 1), rTri_(0, 2), tTri_(0),
                            rTri_(1, 0), rTri_(1, 1), rTri_(1, 2), tTri_(1),
                            rTri_(2, 0), rTri_(2, 1), rTri_(2, 2), tTri_(2));
    const cv::Matx34d p1 = kRgbTri_ * p1Ext;
    const cv::Matx34d p2 = kIrTri_ * p2Ext;
    const cv::Matx33d rectifiedToLeft = rRectRgb_.t();
    const CameraTransform bodyRgb(rBodyRgb_, tBodyRgb_);

    for (size_t i = 0; i < pairCount; ++i)
    {
        // 原图坐标，用于包围框判断
        const cv::Point2f& ptRgbOriginal = matchedRgbPoints[i];
        const cv::Point2f& ptIrOriginal = matchedIrPoints[i];

        // 矫正后坐标，用于三角测量和重投影误差计算
        const cv::Point2f& ptRgbRect = rectifiedRgbPoints[i];
        const cv::Point2f& ptIrRect = rectifiedIrPoints[i];

        cv::Point3f point3dRect = triangulate(ptRgbRect, ptIrRect);
        if (!std::isfinite(point3dRect.x) || !std::isfinite(point3dRect.y) || !std::isfinite(point3dRect.z))
        {
            continue;
        }

        const cv::Vec3d pointRectVec(point3dRect.x, point3dRect.y, point3dRect.z);
        const cv::Vec3d pointLeftVec = rectifiedToLeft * pointRectVec; //把矫正坐标系下的3D点转换回左相机坐标系下，以便与检测框进行空间位置验证
        cv::Point3f point3dRgb(static_cast<float>(pointLeftVec[0]),
                               static_cast<float>(pointLeftVec[1]),
                               static_cast<float>(pointLeftVec[2]));
        if (!std::isfinite(point3dRgb.x) || !std::isfinite(point3dRgb.y) || !std::isfinite(point3dRgb.z))
        {
            continue;
        }
        if (point3dRgb.z <= 0.0f || point3dRgb.z > 30.0f)
        {
            continue;
        }

        const cv::Point3f point3dBody = transformPoint(bodyRgb, point3dRgb);
        if (!std::isfinite(point3dBody.x) || !std::isfinite(point3dBody.y) || !std::isfinite(point3dBody.z))
        {
            continue;
        }
        //判读该三维点是否落在某个检测框内，可能存在多个框都包含该点，选择中心距离最近的那个框进行关联
        int bestDetIdx = -1;
        double bestCenterDist2 = std::numeric_limits<double>::max();
        for (size_t detIdx = 0; detIdx < detectionsOnRgb.size(); ++detIdx)
        {
            const cv::Rect& box = detectionsOnRgb[detIdx].box;
            if (!box.contains(ptRgbOriginal))
            {
                continue;
            }

            const double cx = static_cast<double>(box.x) + static_cast<double>(box.width) * 0.5;
            const double cy = static_cast<double>(box.y) + static_cast<double>(box.height) * 0.5;
            const double dx = static_cast<double>(ptRgbOriginal.x) - cx;
            const double dy = static_cast<double>(ptRgbOriginal.y) - cy;
            const double dist2 = dx * dx + dy * dy;
            if (dist2 < bestCenterDist2)
            {
                bestCenterDist2 = dist2;
                bestDetIdx = static_cast<int>(detIdx);
            }
        }

        if (bestDetIdx >= 0)
        {
            const double pointReprojError = reprojectionErrorNorm(point3dRect, p1, p2, ptRgbRect, ptIrRect);
            if (std::isfinite(pointReprojError))
            {
                // body/IMU坐标系下的3D位置和矫正坐标系下的重投影误差
                pointsPerDetection[static_cast<size_t>(bestDetIdx)].push_back({point3dBody, pointReprojError});
            }
        }
    }

    //std::cout << "Triangulated " << triangulatedPoints.size() << " 3D points from " << pairCount << " matched pairs." << std::endl;

    for (size_t detIdx = 0; detIdx < detectionsOnRgb.size(); ++detIdx)
    {
        const auto& detectionSamples = pointsPerDetection[detIdx];
        const size_t sampleCount = detectionSamples.size();
        //std::cout << "Detection " << detIdx << " has " << sampleCount << " triangulated samples." << std::endl;
        if (sampleCount == 0)
        {
            continue;
        }

        cv::Point3f weightedPoint;
        bool hasEstimate = false;

        if (sampleCount >= 6)
        {
            weightedPoint = huberIrlsCenter(detectionSamples, 10, 1.0, hasEstimate);
        }
        else if (sampleCount >= 2)
        {
            const int topK = std::min(3, static_cast<int>(sampleCount));
            std::vector<int> selectedIndices = topKLowestErrorIndices(detectionSamples, topK);
            std::vector<double> selectedWeights = computeSoftmaxWeights(detectionSamples, selectedIndices, 1.0);
            weightedPoint = weightedAverageFromIndices(detectionSamples, selectedIndices, selectedWeights, hasEstimate);
        }
        else if (sampleCount == 1)
        {
            weightedPoint = detectionSamples.front().point;
            hasEstimate = std::isfinite(weightedPoint.x) && std::isfinite(weightedPoint.y) && std::isfinite(weightedPoint.z);
        }

        if (!hasEstimate)
        {
            continue;
        }

        result.pointsBody[detIdx] = weightedPoint;

    }
    return result;
}
}  // namespace ros_yolo
