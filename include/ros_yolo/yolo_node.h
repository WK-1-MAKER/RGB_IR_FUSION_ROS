#pragma once

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>

#include <opencv2/opencv.hpp>

#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "ros_yolo/detector.h"
#include "ros_yolo/byte_tracker_3d.h"
#include "ros_yolo/frustum_filter.h"
#include "ros_yolo/lidar_localizer.h"
#include "ros_yolo/yolo_utils.h"
#include "ros_yolo/stereo_localizer.h"
#include "lightglue/super_point.h"
#include "lightglue/light_glue.h"
#include "lightglue/utils.h"

namespace ros_yolo
{
class YoloNode
{
public:
    YoloNode(ros::NodeHandle& nh, ros::NodeHandle& pnh);
    ~YoloNode();

private:
    struct BodyPose
    {
        cv::Matx33d rotation = cv::Matx33d::eye();
        cv::Vec3d translation{0.0, 0.0, 0.0};

        cv::Vec3d transform(const cv::Vec3d& point) const
        {
            return rotation * point + translation;
        }
    };

    static BodyPose poseFromOdometry(const nav_msgs::Odometry& message);
    static cv::Mat toBgrImage(const sensor_msgs::ImageConstPtr& msg);
    void loadCameraCalibration();
    void compute_H();
    void rgbCallback(const sensor_msgs::ImageConstPtr& message);
    void irCallback(const sensor_msgs::ImageConstPtr& message);
    void lidarCallback(const sensor_msgs::PointCloud2ConstPtr& message);
    void odometryCallback(const nav_msgs::OdometryConstPtr& message);
    void processingLoop();
    void processImagePair(const sensor_msgs::ImageConstPtr& rgbMsg,
                          const sensor_msgs::ImageConstPtr& irMsg,
                          const sensor_msgs::PointCloud2ConstPtr& lidarMsg,
                          const ros::Time& anchorStamp,
                          const BodyPose& worldFromBody);
    std::vector<Detection> remapDetectionsToOriginalRgb(const std::vector<Detection>& detections,
                                                        const cv::Size& rgbSize) const;
    void filterFeaturesByDetections(const Eigen::Matrix<double, 258, Eigen::Dynamic>& features,
                                    const Eigen::Matrix<double, 1, Eigen::Dynamic>& scores,
                                    const std::vector<Detection>& detections,
                                    double scaleX,
                                    double scaleY,
                                    Eigen::Matrix<double, 258, Eigen::Dynamic>& filteredFeatures,
                                    Eigen::Matrix<double, 1, Eigen::Dynamic>& filteredScores) const;
    void publishCvImage(const cv::Mat& image, const ros::Time& stamp);
    void publishClusterBoundingBoxes(const std::vector<AxisAlignedBounds>& bounds,
                                     const std::vector<bool>& valid,
                                     const std::vector<TrackAnnotation>& tracks,
                                     const ros::Time& stamp);
    BatchedPointCloudResult filterAndPublishPointClouds(
        const std::vector<Detection>& detections,
        const sensor_msgs::PointCloud2ConstPtr& cloudMsg,
        const BodyPose& worldFromBody,
        std::vector<BodyFrustum>& frustums);

private:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    image_transport::ImageTransport it_;
    ros::Subscriber rgbSub_;
    ros::Subscriber irSub_;
    image_transport::Publisher imagePub_;
    ros::Subscriber lidarSub_;
    ros::Publisher filteredPointCloudPub_;
    ros::Publisher frustumRaysPub_;  // 临时 RViz 可视化。
    ros::Publisher clusterBoundingBoxesPub_;
    ros::Subscriber odometrySub_;
    std::mutex inputMutex_;
    std::condition_variable inputCondition_;
    std::deque<sensor_msgs::ImageConstPtr> rgbCache_;
    std::deque<sensor_msgs::ImageConstPtr> irCache_;
    std::deque<sensor_msgs::PointCloud2ConstPtr> lidarCache_;
    struct PendingOdometry
    {
        nav_msgs::OdometryConstPtr message;
        std::chrono::steady_clock::time_point receivedAt;
    };
    std::deque<PendingOdometry> odometryQueue_;
    std::size_t syncQueueSize_{10};
    bool stopping_{false};
    std::thread processingThread_;

    std::unique_ptr<YoloDetector> detector_;
    std::vector<std::string> classNames_;

    std::string modelPath_;
    std::string classNamesPath_;
    std::string rgbTopic_;
    std::string irTopic_;
    std::string outputTopic_;
    std::string lidarTopic_;
    std::string filteredPointCloudTopic_;
    std::string clusterBoundingBoxesTopic_;
    std::string config_path_;
    std::string model_dir_;
    std::unique_ptr<Configs> configs_;
    std::unique_ptr<StereoLocalizer> stereoLocalizer_;
    std::unique_ptr<BYTETracker3D> tracker_;
    std::unique_ptr<LidarLocalizer> lidarLocalizer_;
    double clusterTolerance_{};
    int clusterMinSize_{};
    int clusterMaxSize_{};
    double trackBufferSeconds_{};
    double worldMeasurementTimeoutSeconds_{};
    std::string odometryTopic_;
    ros::Time lastAnchorTimestamp_;

    cv::Matx33d kRgb_;
    cv::Matx33d kIr_;
    cv::Mat distRgb_;
    cv::Mat distIr_;
    cv::Matx33d rBodyRgb_;
    cv::Matx33d rBodyIr_;
    cv::Vec3d tBodyRgb_;
    cv::Vec3d tBodyIr_;
    cv::Matx33d rRgbBody_;
    cv::Matx33d rIrBody_;
    cv::Vec3d tRgbBody_;
    cv::Vec3d tIrBody_;
    cv::Matx33d rIrRgb_;
    cv::Vec3d tIrRgb_;
    cv::Matx33d H_;

    bool useGPU_{};
    bool showWindow_{};
    bool triangulate_{};
    bool cameraOnly_{};

    double confThreshold_{};
    double iouThreshold_{};

    std::shared_ptr<SuperPoint> superpoint_;
    std::shared_ptr<SuperPointLightGlue> superpointLightglue_;
    Eigen::Matrix<double, 258, Eigen::Dynamic> feature_points0, feature_points1;
    Eigen::Matrix<double, 1, Eigen::Dynamic> feature_scores0, feature_scores1;
    std::vector<cv::DMatch> lightglue_matches_;

};
}  // namespace ros_yolo
