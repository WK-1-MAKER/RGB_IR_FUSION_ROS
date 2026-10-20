#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

#include "ros_yolo/yolo_utils.h"

namespace ros_yolo
{
struct StereoLocalizationResult
{
    std::vector<cv::DMatch> filteredMatches;
    // Same order as detectionsOnRgb; NaN means no position estimate.
    std::vector<cv::Point3f> pointsBody;
};

class StereoLocalizer
{
public:
    // Input pixels come from undistorted images. rIrRgb/tIrRgb map RGB to IR;
    // rBodyRgb/tBodyRgb map RGB camera coordinates to body coordinates.
    StereoLocalizer(const cv::Matx33d& kRgb,
                    const cv::Matx33d& kIr,
                    const cv::Matx33d& rIrRgb,
                    const cv::Vec3d& tIrRgb,
                    const cv::Matx33d& rBodyRgb,
                    const cv::Vec3d& tBodyRgb);

    const cv::Matx33d& fundamentalMatrix() const { return fundamental_; }
    void updateRectification(const cv::Size& imageSize);
    void disableRectification();  // Restore original pixels and camera parameters.
    void filterMatches(const std::vector<cv::KeyPoint>& pointsRgb,
                       const std::vector<cv::KeyPoint>& pointsIr,
                       std::vector<cv::DMatch>& matches,
                       double threshold = 5.0) const;
    StereoLocalizationResult localize(const std::vector<cv::KeyPoint>& pointsRgb,
                                      const std::vector<cv::KeyPoint>& pointsIr,
                                      const std::vector<cv::DMatch>& matches,
                                      const std::vector<Detection>& detectionsOnRgb) const;

private:
    void rectifyMatchedPoints(const std::vector<cv::Point2f>& pointsRgb,
                              const std::vector<cv::Point2f>& pointsIr,
                              std::vector<cv::Point2f>& rectifiedRgb,
                              std::vector<cv::Point2f>& rectifiedIr) const;
    cv::Point2d normalizePixel(const cv::Matx33d& intrinsic, const cv::Point2f& point) const;
    cv::Vec3d triangulateSvd(const cv::Matx34d& p1,
                            const cv::Matx34d& p2,
                            const cv::Point2d& point1,
                            const cv::Point2d& point2) const;
    cv::Point3f triangulate(const cv::Point2f& pointRgb, const cv::Point2f& pointIr, int iters = 10) const;

    cv::Matx33d kRgb_, kIr_, rIrRgb_, rBodyRgb_;
    cv::Vec3d tIrRgb_, tBodyRgb_;
    cv::Matx33d fundamental_;
    cv::Matx33d kRgbTri_, kIrTri_, rTri_;
    cv::Vec3d tTri_;
    cv::Matx33d rRectRgb_ = cv::Matx33d::eye();
    cv::Matx33d rRectIr_ = cv::Matx33d::eye();
    cv::Matx34d pRectRgb_ = cv::Matx34d::zeros();
    cv::Matx34d pRectIr_ = cv::Matx34d::zeros();
    bool rectificationReady_{false};
    cv::Size rectificationImageSize_;
};
}  // namespace ros_yolo
