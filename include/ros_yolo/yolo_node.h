#pragma once

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>

#include <opencv2/opencv.hpp>

#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "ros_yolo/detector.h"
#include "ros_yolo/latest_frame_buffer.h"
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
    static cv::Mat toBgrImage(const sensor_msgs::ImageConstPtr& msg);
    void loadCameraCalibration();
    void compute_H();
    void syncImageCallback(const sensor_msgs::ImageConstPtr& rgbMsg,
                           const sensor_msgs::ImageConstPtr& irMsg,
                           const sensor_msgs::PointCloud2ConstPtr& lidarMsg);
    void processingLoop();
    void processImagePair(const sensor_msgs::ImageConstPtr& rgbMsg,
                          const sensor_msgs::ImageConstPtr& irMsg,
                          const sensor_msgs::PointCloud2ConstPtr& lidarMsg);
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
    void publishFilteredPointClouds(const std::vector<Detection>& detections,
                                    const sensor_msgs::PointCloud2ConstPtr& cloudMsg);

private:
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::PointCloud2>;

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    image_transport::ImageTransport it_;
    message_filters::Subscriber<sensor_msgs::Image> rgbSub_;
    message_filters::Subscriber<sensor_msgs::Image> irSub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    image_transport::Publisher imagePub_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> lidarSub_;
    ros::Publisher filteredPointCloudPub_;
    ros::Publisher frustumRaysPub_;  // 临时 RViz 可视化。
    LatestFrameBuffer frameBuffer_;
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
    std::string config_path_;
    std::string model_dir_;
    std::unique_ptr<Configs> configs_;
    std::unique_ptr<StereoLocalizer> stereoLocalizer_;

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

    double confThreshold_{};
    double iouThreshold_{};
    double syncMaxInterval_{};

    std::shared_ptr<SuperPoint> superpoint_;
    std::shared_ptr<SuperPointLightGlue> superpointLightglue_;
    Eigen::Matrix<double, 258, Eigen::Dynamic> feature_points0, feature_points1;
    Eigen::Matrix<double, 1, Eigen::Dynamic> feature_scores0, feature_scores1;
    std::vector<cv::DMatch> lightglue_matches_;

};
}  // namespace ros_yolo
