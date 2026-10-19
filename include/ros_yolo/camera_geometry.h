#pragma once

#include <cmath>

#include <opencv2/opencv.hpp>

namespace ros_yolo
{
struct CameraTransform
{
    CameraTransform()
        : rotation(cv::Matx33d::eye()), translation(0.0, 0.0, 0.0)
    {
    }

    CameraTransform(const cv::Matx33d& rotationIn, const cv::Vec3d& translationIn)
        : rotation(rotationIn), translation(translationIn)
    {
    }

    cv::Matx33d rotation;
    cv::Vec3d translation;
};

inline CameraTransform inverseTransform(const CameraTransform& transform)
{
    const cv::Matx33d inverseRotation = transform.rotation.t();
    return CameraTransform(inverseRotation, -(inverseRotation * transform.translation));
}

inline CameraTransform composeTransforms(const CameraTransform& lhs, const CameraTransform& rhs)
{
    return CameraTransform(lhs.rotation * rhs.rotation,
                           lhs.rotation * rhs.translation + lhs.translation);
}

inline cv::Point3f transformPoint(const CameraTransform& transform, const cv::Point3f& point)
{
    const cv::Vec3d pointVec(point.x, point.y, point.z);
    const cv::Vec3d transformed = transform.rotation * pointVec + transform.translation;
    return cv::Point3f(static_cast<float>(transformed[0]),
                       static_cast<float>(transformed[1]),
                       static_cast<float>(transformed[2]));
}

// Refines a DLT triangulation in the coordinate system represented by P1/P2.
// The residual is pixel reprojection error and each update is accepted only if
// it lowers the combined squared error.
inline cv::Vec3d refineTriangulatedPoint(const cv::Vec3d& initialPoint,
                                         const cv::Matx34d& p1,
                                         const cv::Matx34d& p2,
                                         const cv::Point2d& observed1,
                                         const cv::Point2d& observed2,
                                         int maxIterations = 10)
{
    auto project = [](const cv::Matx34d& projection,
                      const cv::Vec3d& point,
                      cv::Vec3d& homogeneous,
                      cv::Point2d& pixel) -> bool {
        homogeneous = projection * cv::Vec4d(point[0], point[1], point[2], 1.0);
        if (!std::isfinite(homogeneous[0]) || !std::isfinite(homogeneous[1]) ||
            !std::isfinite(homogeneous[2]) || homogeneous[2] <= 1e-12)
        {
            return false;
        }
        pixel = cv::Point2d(homogeneous[0] / homogeneous[2],
                            homogeneous[1] / homogeneous[2]);
        return std::isfinite(pixel.x) && std::isfinite(pixel.y);
    };

    auto residualAndCost = [&](const cv::Vec3d& point,
                               cv::Vec4d& residual,
                               double& cost) -> bool {
        cv::Vec3d projected1;
        cv::Vec3d projected2;
        cv::Point2d pixel1;
        cv::Point2d pixel2;
        if (!project(p1, point, projected1, pixel1) ||
            !project(p2, point, projected2, pixel2))
        {
            return false;
        }
        residual = cv::Vec4d(pixel1.x - observed1.x,
                             pixel1.y - observed1.y,
                             pixel2.x - observed2.x,
                             pixel2.y - observed2.y);
        cost = residual.dot(residual);
        return std::isfinite(cost);
    };

    cv::Vec3d point = initialPoint;
    cv::Vec4d residual;
    double cost = 0.0;
    if (!std::isfinite(point[0]) || !std::isfinite(point[1]) ||
        !std::isfinite(point[2]) || !residualAndCost(point, residual, cost))
    {
        return initialPoint;
    }

    for (int iteration = 0; iteration < maxIterations; ++iteration)
    {
        cv::Matx43d jacobian;
        const cv::Matx34d projections[2] = {p1, p2};
        for (int view = 0; view < 2; ++view)
        {
            const cv::Vec3d homogeneous = projections[view] *
                cv::Vec4d(point[0], point[1], point[2], 1.0);
            if (!std::isfinite(homogeneous[2]) || std::abs(homogeneous[2]) < 1e-12)
            {
                return point;
            }

            const double z = homogeneous[2];
            const double zSquared = z * z;
            for (int coordinate = 0; coordinate < 3; ++coordinate)
            {
                jacobian(2 * view, coordinate) =
                    (projections[view](0, coordinate) * z -
                     homogeneous[0] * projections[view](2, coordinate)) / zSquared;
                jacobian(2 * view + 1, coordinate) =
                    (projections[view](1, coordinate) * z -
                     homogeneous[1] * projections[view](2, coordinate)) / zSquared;
            }
        }

        cv::Mat delta;
        if (!cv::solve(cv::Mat(jacobian), cv::Mat(residual), delta, cv::DECOMP_SVD) ||
            delta.rows != 3 || delta.cols != 1)
        {
            break;
        }

        const cv::Vec3d fullStep(delta.at<double>(0, 0),
                                 delta.at<double>(1, 0),
                                 delta.at<double>(2, 0));
        if (!std::isfinite(fullStep[0]) || !std::isfinite(fullStep[1]) ||
            !std::isfinite(fullStep[2]))
        {
            break;
        }

        const double stepNorm = cv::norm(fullStep);
        if (stepNorm < 1e-10 * (1.0 + cv::norm(point)))
        {
            break;
        }

        bool accepted = false;
        cv::Vec3d acceptedPoint = point;
        cv::Vec4d acceptedResidual = residual;
        double acceptedCost = cost;
        double stepScale = 1.0;
        for (int backtrack = 0; backtrack < 12; ++backtrack)
        {
            const cv::Vec3d candidate = point - stepScale * fullStep;
            cv::Vec4d candidateResidual;
            double candidateCost = 0.0;
            if (residualAndCost(candidate, candidateResidual, candidateCost) &&
                candidateCost < cost)
            {
                accepted = true;
                acceptedPoint = candidate;
                acceptedResidual = candidateResidual;
                acceptedCost = candidateCost;
                break;
            }
            stepScale *= 0.5;
        }

        if (!accepted)
        {
            break;
        }

        const double costImprovement = cost - acceptedCost;
        point = acceptedPoint;
        residual = acceptedResidual;
        cost = acceptedCost;
        if (costImprovement < 1e-12 * (1.0 + cost))
        {
            break;
        }
    }

    return point;
}
}  // namespace ros_yolo
