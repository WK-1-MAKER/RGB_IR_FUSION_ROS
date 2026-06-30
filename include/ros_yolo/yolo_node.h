#pragma once

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include <opencv2/opencv.hpp>

#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <chrono>

#include "ros_yolo/detector.h"
#include "ros_yolo/yolo_utils.h"
#include "lightglue/super_point.h"
#include "lightglue/light_glue.h"
#include "lightglue/utils.h"

namespace ros_yolo
{
class YoloNode
{
public:
    YoloNode(ros::NodeHandle& nh, ros::NodeHandle& pnh);

private:
    static cv::Mat toBgrImage(const sensor_msgs::ImageConstPtr& msg);
    void loadCameraCalibration();
    void compute_H();
    void compute_F();
    void updateRectification(const cv::Size& imageSize);
    void rectifyMatchedPoints(const std::vector<cv::Point2f>& pointsRgb,
                              const std::vector<cv::Point2f>& pointsIr,
                              std::vector<cv::Point2f>& rectifiedRgb,
                              std::vector<cv::Point2f>& rectifiedIr) const;
    void syncImageCallback(const sensor_msgs::ImageConstPtr& rgbMsg,
                           const sensor_msgs::ImageConstPtr& irMsg);
    std::vector<Detection> remapDetectionsToOriginalRgb(const std::vector<Detection>& detections,
                                                        const cv::Size& rgbSize) const;
    void filterFeaturesByDetections(const Eigen::Matrix<double, 258, Eigen::Dynamic>& features,
                                    const Eigen::Matrix<double, 1, Eigen::Dynamic>& scores,
                                    const std::vector<Detection>& detections,
                                    double scaleX,
                                    double scaleY,
                                    Eigen::Matrix<double, 258, Eigen::Dynamic>& filteredFeatures,
                                    Eigen::Matrix<double, 1, Eigen::Dynamic>& filteredScores) const;
    void annotatedImageCallback(const sensor_msgs::ImageConstPtr& msg);
    void publishCvImage(const cv::Mat& image, const ros::Time& stamp);
    void correct_points(const cv::Matx33d& F,
                        const std::vector<cv::KeyPoint>& points0,
                        const std::vector<cv::KeyPoint>& points1,
                        std::vector<cv::DMatch>& matches,
                        double threshold = 3.0);
    cv::Point2f normalize_pixel(const cv::Matx33d& K, const cv::Point2f& pt) const;
    cv::Point3f triangulate_svd(const cv::Matx34d& P1,
                                const cv::Matx34d& P2,
                                const cv::Point2f& ptn1,
                                const cv::Point2f& ptn2) const;
    cv::Point3f triangulate(const cv::Point2f& pt1, const cv::Point2f& pt2, int iters = 10) const;

private:
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image>;

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    image_transport::ImageTransport it_;
    message_filters::Subscriber<sensor_msgs::Image> rgbSub_;
    message_filters::Subscriber<sensor_msgs::Image> irSub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    image_transport::Subscriber annotatedSub_;
    image_transport::Publisher imagePub_;

    std::unique_ptr<YoloDetector> detector_;
    std::vector<std::string> classNames_;

    std::string modelPath_;
    std::string classNamesPath_;
    std::string rgbTopic_;
    std::string irTopic_;
    std::string outputTopic_;
    std::string config_path_;
    std::string model_dir_;
    std::unique_ptr<Configs> configs_;

    cv::Matx33d kRgb_;
    cv::Matx33d kIr_;
    cv::Mat distRgb_;
    cv::Mat distIr_;
    cv::Matx33d rRgbWorld_;
    cv::Matx33d rIrWorld_;
    cv::Vec3d tRgbWorld_;
    cv::Vec3d tIrWorld_;
    cv::Matx33d r_;
    cv::Vec3d t_;
    cv::Matx33d H_;
    cv::Matx33d F_;

    cv::Matx33d kRgbTri_;
    cv::Matx33d kIrTri_;
    cv::Matx33d rTri_;
    cv::Vec3d tTri_;

    cv::Matx33d rRectRgb_ = cv::Matx33d::eye();
    cv::Matx33d rRectIr_ = cv::Matx33d::eye();
    cv::Matx34d pRectRgb_ = cv::Matx34d::zeros();
    cv::Matx34d pRectIr_ = cv::Matx34d::zeros();
    bool rectificationReady_{false};
    cv::Size rectificationImageSize_;

    bool useGPU_{};
    bool showWindow_{};

    double confThreshold_{};
    double iouThreshold_{};

    std::shared_ptr<SuperPoint> superpoint_;
    std::shared_ptr<SuperPointLightGlue> superpointLightglue_;
    Eigen::Matrix<double, 258, Eigen::Dynamic> feature_points0, feature_points1;
    Eigen::Matrix<double, 1, Eigen::Dynamic> feature_scores0, feature_scores1;
    std::vector<cv::DMatch> lightglue_matches_;

};
}  // namespace ros_yolo
