#pragma once

#include <array>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

namespace ros_yolo
{
struct BodyFrustum
{
    cv::Vec3d origin;
    cv::Vec3d forward;
    std::array<cv::Vec3d, 4> rays;  // 单位方向；公共起点为 origin。
    std::array<cv::Vec3d, 4> edgeNormals;

    bool contains(const cv::Vec3d& point) const
    {
        const cv::Vec3d relative = point - origin;
        const double relativeNorm = std::sqrt(relative.dot(relative));
        if (!std::isfinite(relativeNorm) || relativeNorm <= 1e-9 || relative.dot(forward) <= 0.0)
        {
            return false;
        }

        const double tolerance = 1e-9 * std::max(1.0, relativeNorm);
        for (const cv::Vec3d& normal : edgeNormals)
        {
            if (normal.dot(relative) < -tolerance)
            {
                return false;
            }
        }
        return true;
    }
};

inline bool buildBodyFrustum(const cv::Rect& box,
                      const cv::Matx33d& intrinsic,
                      const cv::Matx33d& bodyFromRgbRotation,
                      const cv::Vec3d& bodyFromRgbTranslation,
                      BodyFrustum& frustum)
{
    if (box.width <= 0 || box.height <= 0)
    {
        return false;
    }
    const std::array<cv::Point2d, 4> corners = {
        cv::Point2d(box.x, box.y),
        cv::Point2d(box.x + box.width, box.y),
        cv::Point2d(box.x + box.width, box.y + box.height),
        cv::Point2d(box.x, box.y + box.height)};

    const cv::Matx33d intrinsicInverse = intrinsic.inv(cv::DECOMP_SVD);
    std::array<cv::Vec3d, 4> rays;
    for (size_t i = 0; i < corners.size(); ++i)
    {
        const cv::Vec3d pixel(corners[i].x, corners[i].y, 1.0);
        const cv::Vec3d rayRgb = intrinsicInverse * pixel;
        const double rayNorm = std::sqrt(rayRgb.dot(rayRgb));
        if (!std::isfinite(rayNorm) || rayNorm <= 1e-12)
        {
            return false;
        }

        rays[i] = bodyFromRgbRotation * (rayRgb * (1.0 / rayNorm));
        const double bodyRayNorm = std::sqrt(rays[i].dot(rays[i]));
        if (!std::isfinite(bodyRayNorm) || bodyRayNorm <= 1e-12)
        {
            return false;
        }
        rays[i] *= 1.0 / bodyRayNorm;
    }

    cv::Vec3d centerRay(0.0, 0.0, 0.0);
    for (const cv::Vec3d& ray : rays)
    {
        centerRay += ray;
    }
    const double centerNorm = std::sqrt(centerRay.dot(centerRay));
    if (!std::isfinite(centerNorm) || centerNorm <= 1e-12)
    {
        return false;
    }
    centerRay *= 1.0 / centerNorm;

    frustum.rays = rays;
    frustum.origin = bodyFromRgbTranslation;
    frustum.forward = bodyFromRgbRotation * cv::Vec3d(0.0, 0.0, 1.0);
    const double forwardNorm = std::sqrt(frustum.forward.dot(frustum.forward));
    if (!std::isfinite(forwardNorm) || forwardNorm <= 1e-12)
    {
        return false;
    }
    frustum.forward *= 1.0 / forwardNorm;

    for (size_t i = 0; i < rays.size(); ++i)
    {
        const cv::Vec3d nextRay = rays[(i + 1) % rays.size()];
        cv::Vec3d normal(rays[i][1] * nextRay[2] - rays[i][2] * nextRay[1],
                         rays[i][2] * nextRay[0] - rays[i][0] * nextRay[2],
                         rays[i][0] * nextRay[1] - rays[i][1] * nextRay[0]);
        const double normalNorm = std::sqrt(normal.dot(normal));
        if (!std::isfinite(normalNorm) || normalNorm <= 1e-12)
        {
            return false;
        }
        if (normal.dot(centerRay) < 0.0)
        {
            normal *= -1.0;
        }
        frustum.edgeNormals[i] = normal * (1.0 / normalNorm);
    }
    return true;
}

}  // namespace ros_yolo
