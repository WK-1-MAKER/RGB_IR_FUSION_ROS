#pragma once

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
}  // namespace ros_yolo
