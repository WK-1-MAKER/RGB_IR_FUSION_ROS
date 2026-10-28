#include "ros_yolo/yolo_node.h"
#include "ros_yolo/frustum_filter.h"
#include "ros_yolo/camera_geometry.h"

#include <boost/bind.hpp>
#include <visualization_msgs/MarkerArray.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace ros_yolo
{
namespace
{
cv::Matx33d matx33FromVector(const std::vector<double>& values)
{
    if (values.size() != 9)
    {
        throw std::runtime_error("Camera matrix must contain exactly 9 values.");
    }

    return cv::Matx33d(values[0], values[1], values[2],
                      values[3], values[4], values[5],
                      values[6], values[7], values[8]);
}

cv::Vec3d vec3FromVector(const std::vector<double>& values)
{
    if (values.size() != 3)
    {
        throw std::runtime_error("Camera translation must contain exactly 3 values.");
    }

    return cv::Vec3d(values[0], values[1], values[2]);
}

cv::Mat distortionFromVector(const std::vector<double>& values)
{
    if (values.size() != 5)
    {
        throw std::runtime_error("Camera distortion must contain exactly 5 values.");
    }

    return (cv::Mat_<double>(1, 5) << values[0], values[1], values[2], values[3], values[4]);
}

}  // namespace

YoloNode::YoloNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
    : nh_(nh), pnh_(pnh), it_(nh_)
{
    pnh_.param<std::string>("model_path", modelPath_, "");
    pnh_.param<std::string>("class_names_path", classNamesPath_, "");
    pnh_.param<std::string>("rgb_topic", rgbTopic_, "/camera/color/image_raw");
    pnh_.param<std::string>("ir_topic", irTopic_, "/pub_ir");
    pnh_.param<std::string>("output_topic", outputTopic_, "/two_stream/image_annotated");
    pnh_.param<std::string>("lidar_topic", lidarTopic_, "/fusion/points");
    pnh_.param<std::string>("filtered_point_cloud_topic",
                            filteredPointCloudTopic_,
                            "/fusion/points_filtered");
    pnh_.param<double>("sync_max_interval", syncMaxInterval_, 0.1);
    if (!std::isfinite(syncMaxInterval_) || syncMaxInterval_ < 0.0)
    {
        throw std::runtime_error("sync_max_interval must be finite and nonnegative.");
    }

    pnh_.param<bool>("use_gpu", useGPU_, true);
    pnh_.param<bool>("show_window", showWindow_, false);
    pnh_.param<bool>("triangulate", triangulate_, true);
    pnh_.param<bool>("camera_only", cameraOnly_, false);

    pnh_.param<double>("conf_threshold", confThreshold_, 0.30);
    pnh_.param<double>("iou_threshold", iouThreshold_, 0.40);

    pnh_.param<std::string>("config_path", config_path_, "");
    pnh_.param<std::string>("model_dir", model_dir_, "");

    configs_ = std::make_unique<Configs>(config_path_, model_dir_);
    loadCameraCalibration();
    compute_H();
    stereoLocalizer_ = std::make_unique<StereoLocalizer>(kRgb_, kIr_, rIrRgb_, tIrRgb_, rBodyRgb_, tBodyRgb_);

    int inputWidth = 640;
    int inputHeight = 640;
    
    pnh_.param<int>("input_width", inputWidth, 640);
    pnh_.param<int>("input_height", inputHeight, 640);

    classNames_ = utils::loadNames(classNamesPath_);
    if (classNames_.empty())
    {
        throw std::runtime_error("Class names file is empty or invalid: " + classNamesPath_);
    }

    detector_ = std::make_unique<YoloDetector>(modelPath_, useGPU_, cv::Size(inputWidth, inputHeight));
    imagePub_ = it_.advertise(outputTopic_, 1);
    if (!cameraOnly_)
    {
        filteredPointCloudPub_ = nh_.advertise<sensor_msgs::PointCloud2>(filteredPointCloudTopic_, 100);
        frustumRaysPub_ = nh_.advertise<visualization_msgs::MarkerArray>("/fusion/frustum_rays", 1);
    }

    if (triangulate_)
    {
        superpoint_ = std::make_shared<SuperPoint>(configs_->superpoint_config);
        superpointLightglue_ = std::make_shared<SuperPointLightGlue>(configs_->superpoint_lightglue_config);

        if (!superpoint_->build()) {
            throw std::runtime_error("Failed to build SuperPoint TensorRT engine.");
        }

        if (!superpointLightglue_->build()) {
            throw std::runtime_error("Failed to build SuperPoint-LightGlue TensorRT engine.");
        }
    }

    rgbSub_.subscribe(nh_, rgbTopic_, 10);
    irSub_.subscribe(nh_, irTopic_, 10);
    if (cameraOnly_)
    {
        cameraSync_ = std::make_shared<message_filters::Synchronizer<CameraSyncPolicy>>(
            CameraSyncPolicy(10), rgbSub_, irSub_);
        cameraSync_->setMaxIntervalDuration(ros::Duration(syncMaxInterval_));
        cameraSync_->registerCallback(boost::bind(&YoloNode::syncImageCallback, this, _1, _2));
    }
    else
    {
        lidarSub_.subscribe(nh_, lidarTopic_, 10);
        cameraLidarSync_ = std::make_shared<message_filters::Synchronizer<CameraLidarSyncPolicy>>(
            CameraLidarSyncPolicy(10), rgbSub_, irSub_, lidarSub_);
        cameraLidarSync_->setMaxIntervalDuration(ros::Duration(syncMaxInterval_));
        cameraLidarSync_->registerCallback(boost::bind(&YoloNode::syncImageLidarCallback, this, _1, _2, _3));
    }
    ROS_INFO_STREAM("ros_yolo subscribed RGB: " << rgbTopic_);
    ROS_INFO_STREAM("ros_yolo subscribed IR : " << irTopic_);
    ROS_INFO_STREAM("ros_yolo camera_only: " << std::boolalpha << cameraOnly_);
    if (!cameraOnly_)
    {
        ROS_INFO_STREAM("ros_yolo subscribed cloud: " << lidarTopic_);
        ROS_INFO_STREAM("ros_yolo filtered cloud: " << filteredPointCloudTopic_);
    }
    ROS_INFO_STREAM("ros_yolo output topic  : " << outputTopic_);

    // Start last so a constructor failure cannot leave a joinable thread behind.
    // ROS callbacks are not dispatched until main() enters ros::spin().
    processingThread_ = std::thread(&YoloNode::processingLoop, this);
}

YoloNode::~YoloNode()
{
    rgbSub_.unsubscribe();
    irSub_.unsubscribe();
    lidarSub_.unsubscribe();
    cameraSync_.reset();
    cameraLidarSync_.reset();
    frameBuffer_.stop();
    if (processingThread_.joinable())
    {
        processingThread_.join();
    }
}

void YoloNode::processingLoop()
{
    FramePair frame;
    while (frameBuffer_.waitAndTake(frame))
    {
        try
        {
            processImagePair(frame.rgb, frame.ir, frame.lidar);
        }
        catch (const cv_bridge::Exception& e)
        {
            ROS_ERROR_STREAM("Image conversion failed: " << e.what());
        }
        catch (const cv::Exception& e)
        {
            ROS_ERROR_STREAM("OpenCV processing failed: " << e.what());
        }
        catch (const std::exception& e)
        {
            ROS_ERROR_STREAM("Frame processing failed: " << e.what());
        }
        catch (...)
        {
            ROS_ERROR("Frame processing failed with an unknown exception.");
        }
    }
}

void YoloNode::loadCameraCalibration()
{
    if (!configs_)
    {
        throw std::runtime_error("Cannot load camera calibration before config is initialized.");
    }

    const CameraCalibrationConfig& cameraConfig = configs_->camera_config;
    kRgb_ = matx33FromVector(cameraConfig.rgb.intrinsic);
    kIr_ = matx33FromVector(cameraConfig.ir.intrinsic);
    distRgb_ = distortionFromVector(cameraConfig.rgb.distortion);
    distIr_ = distortionFromVector(cameraConfig.ir.distortion);

    const CameraTransform rgbLidar(
        matx33FromVector(cameraConfig.rgb.rotation),
        vec3FromVector(cameraConfig.rgb.translation));
    const CameraTransform irRgbCalibration(
        matx33FromVector(cameraConfig.ir.rotation),
        vec3FromVector(cameraConfig.ir.translation));
    const BodyCameraTransforms bodyCameras =
        bodyCameraTransformsFromRgbLidarAndIrRgb(rgbLidar, irRgbCalibration);

    rBodyRgb_ = bodyCameras.bodyRgb.rotation;
    tBodyRgb_ = bodyCameras.bodyRgb.translation;
    rBodyIr_ = bodyCameras.bodyIr.rotation;
    tBodyIr_ = bodyCameras.bodyIr.translation;

    const CameraTransform rgbBody = inverseTransform(bodyCameras.bodyRgb);
    const CameraTransform irBody = inverseTransform(bodyCameras.bodyIr);
    const CameraTransform irRgb = composeTransforms(irBody, bodyCameras.bodyRgb);

    rRgbBody_ = rgbBody.rotation;
    tRgbBody_ = rgbBody.translation;
    rIrBody_ = irBody.rotation;
    tIrBody_ = irBody.translation;
    rIrRgb_ = irRgb.rotation;
    tIrRgb_ = irRgb.translation;

}

cv::Mat YoloNode::toBgrImage(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImageConstPtr cvPtr;
    if (msg->encoding == sensor_msgs::image_encodings::BGR8 || msg->encoding == sensor_msgs::image_encodings::RGB8)
    {
        cvPtr = cv_bridge::toCvShare(msg, "bgr8");
        return cvPtr->image.clone();
    }

    if (msg->encoding == sensor_msgs::image_encodings::MONO8)
    {
        cvPtr = cv_bridge::toCvShare(msg, "mono8");
        cv::Mat bgr;
        cv::cvtColor(cvPtr->image, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
    }

    cvPtr = cv_bridge::toCvShare(msg, "bgr8");
    return cvPtr->image.clone();
}

void YoloNode::compute_H()
{
    const double refDepth = 10.0;
    const cv::Vec3d n(0.0, 0.0, 1.0);
    const cv::Matx33d tOuter(
        tIrRgb_[0] * n[0], tIrRgb_[0] * n[1], tIrRgb_[0] * n[2],
        tIrRgb_[1] * n[0], tIrRgb_[1] * n[1], tIrRgb_[1] * n[2],
        tIrRgb_[2] * n[0], tIrRgb_[2] * n[1], tIrRgb_[2] * n[2]);

    H_ = kIr_ * (rIrRgb_ + tOuter * (1.0 / refDepth)) * kRgb_.inv(cv::DECOMP_SVD);
    // H_ maps RGB image coordinates into IR image coordinates for alignment.
}

void YoloNode::syncImageCallback(const sensor_msgs::ImageConstPtr& rgbMsg,
                                 const sensor_msgs::ImageConstPtr& irMsg)
{
    frameBuffer_.push(FramePair(rgbMsg, irMsg, sensor_msgs::PointCloud2ConstPtr()));
}

void YoloNode::syncImageLidarCallback(const sensor_msgs::ImageConstPtr& rgbMsg,
                                 const sensor_msgs::ImageConstPtr& irMsg,
                                 const sensor_msgs::PointCloud2ConstPtr& lidarMsg)
{
    // 整组替换待处理帧，检测结束后仍使用本组点云。
    frameBuffer_.push(FramePair(rgbMsg, irMsg, lidarMsg));
}

void YoloNode::publishFilteredPointClouds(
    const std::vector<Detection>& detections,
    const sensor_msgs::PointCloud2ConstPtr& cloudMsg)
{
    if (!cloudMsg)
    {
        return;
    }

    // 临时可视化：每帧清除旧射线，显示长度不影响无限远视锥体过滤。
    visualization_msgs::MarkerArray markers;
    visualization_msgs::Marker clear;
    clear.header = cloudMsg->header;
    clear.action = visualization_msgs::Marker::DELETEALL;
    markers.markers.push_back(clear);

    try
    {
        for (const Detection& detection : detections)
        {
            BodyFrustum frustum;
            if (buildBodyFrustum(detection.box, kRgb_, rBodyRgb_, tBodyRgb_, frustum))
            {
                visualization_msgs::Marker marker;
                marker.header = cloudMsg->header;
                marker.ns = "frustum_rays";
                marker.id = static_cast<int>(markers.markers.size() - 1);
                marker.type = visualization_msgs::Marker::LINE_LIST;
                marker.action = visualization_msgs::Marker::ADD;
                marker.pose.orientation.w = 1.0;
                marker.scale.x = 0.03;
                marker.color.g = 1.0;
                marker.color.a = 1.0;
                marker.lifetime = ros::Duration(0);  // 暂停 rosbag 时保留，下一帧通过 DELETEALL 更新。
                for (const auto& ray : frustum.rays)
                {
                    geometry_msgs::Point start, end;
                    start.x = frustum.origin[0];
                    start.y = frustum.origin[1];
                    start.z = frustum.origin[2];
                    const cv::Vec3d tip = frustum.origin + 30.0 * ray;
                    end.x = tip[0];
                    end.y = tip[1];
                    end.z = tip[2];
                    marker.points.push_back(start);
                    marker.points.push_back(end);
                }
                markers.markers.push_back(marker);
                filteredPointCloudPub_.publish(filterPointCloud(*cloudMsg, frustum));
            }
        }
    }
    catch (const std::exception& e)
    {
        ROS_WARN_THROTTLE(2.0, "Cannot filter PointCloud2: %s", e.what());
    }
    frustumRaysPub_.publish(markers);
}

void YoloNode::processImagePair(const sensor_msgs::ImageConstPtr& rgbMsg,
                                const sensor_msgs::ImageConstPtr& irMsg,
                                const sensor_msgs::PointCloud2ConstPtr& lidarMsg)
{
    cv::Mat rgbImage = toBgrImage(rgbMsg);
    cv::Mat irImage = toBgrImage(irMsg);

    if (rgbImage.empty())
    {
        ROS_WARN("Received empty RGB image frame.");
        return;
    }

    if (irImage.empty())
    {
        ROS_WARN("Received empty IR image frame.");
        return;
    }

    cv::Mat rgbUndistorted;
    cv::Mat irUndistorted;
    cv::undistort(rgbImage, rgbUndistorted, cv::Mat(kRgb_), distRgb_, cv::Mat(kRgb_));
    cv::undistort(irImage, irUndistorted, cv::Mat(kIr_), distIr_, cv::Mat(kIr_));
    rgbImage = rgbUndistorted;
    irImage = irUndistorted;

    const double stampDeltaSec = std::abs((rgbMsg->header.stamp - irMsg->header.stamp).toSec());
    //std::cout << "RGB-IR timestamp delta: " << stampDeltaSec * 1000.0 << " ms" << std::endl;
    if (stampDeltaSec > 0.03)
    {
        ROS_WARN_STREAM_THROTTLE(1.0,
                                 "RGB/IR stamp delta " << stampDeltaSec * 1000.0
                                                       << " ms exceeds 30 ms, continue inference.");
    }

    cv::Mat rgbAligned;  //配准后的RGB图像
    cv::warpPerspective(rgbImage,
                        rgbAligned,
                        cv::Mat(H_),
                        irImage.size(),
                        cv::INTER_LINEAR,
                        cv::BORDER_CONSTANT,
                        cv::Scalar(0, 0, 0));

    if (rgbAligned.empty())
    {
        ROS_WARN_THROTTLE(2.0, "RGB-IR alignment failed, skip frame.");
        return;
    }

    try
    {
        stereoLocalizer_->updateRectification(rgbImage.size());
    }
    catch (const cv::Exception& e)
    {
        stereoLocalizer_->disableRectification();
        ROS_WARN_STREAM_THROTTLE(1.0, "Rectification disabled for this frame (OpenCV): " << e.what());
    }
    catch (const std::exception& e)
    {
        stereoLocalizer_->disableRectification();
        ROS_WARN_STREAM_THROTTLE(1.0, "Rectification disabled for this frame: " << e.what());
    }

    std::vector<Detection> detections = detector_->detect(  //目标检测推理
        rgbAligned,
        irImage,
        static_cast<float>(confThreshold_),
        static_cast<float>(iouThreshold_));
    
    // 将检测结果从配准图像坐标系映射回原始RGB图像坐标系，以便后续与特征点匹配和三角测量使用
    // Detection包含类别ID，置信度，边界框坐标信息
    std::vector<Detection> detectionsOnRgb = remapDetectionsToOriginalRgb(detections, rgbImage.size());

    publishFilteredPointClouds(detectionsOnRgb, lidarMsg);

    std::vector<cv::Point3f> perDetectionWeightedPoints(
        detectionsOnRgb.size(),
        cv::Point3f(std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::quiet_NaN()));

    if (triangulate_)
    {
    
    //-------------------------------lightglue--------------------------------
    const int width = configs_->superpoint_lightglue_config.image_width;
    const int height = configs_->superpoint_lightglue_config.image_height;
    // 使用的是RGB原图和IR原图进行的特征点匹配
    cv::Mat img0_original = rgbImage;
    cv::Mat img1_original = irImage;

    cv::Mat img0_gray, img1_gray;
    if (img0_original.channels() == 3)
    {
        cv::cvtColor(img0_original, img0_gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        img0_gray = img0_original;
    }

    if (img1_original.channels() == 3)
    {
        cv::cvtColor(img1_original, img1_gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        img1_gray = img1_original;
    }

    cv::Mat img0_infer, img1_infer;
    cv::resize(img0_gray, img0_infer, cv::Size(width, height));
    cv::resize(img1_gray, img1_infer, cv::Size(width, height));

    superpoint_->infer(img0_infer, feature_points0, feature_scores0);
    superpoint_->infer(img1_infer, feature_points1, feature_scores1);

    // 计算特征点坐标从推理图像坐标系到原始图像坐标系的缩放比例
    const double scale_x0 = static_cast<double>(img0_original.cols) / static_cast<double>(img0_infer.cols);
    const double scale_y0 = static_cast<double>(img0_original.rows) / static_cast<double>(img0_infer.rows);
    const double scale_x1 = static_cast<double>(img1_original.cols) / static_cast<double>(img1_infer.cols);
    const double scale_y1 = static_cast<double>(img1_original.rows) / static_cast<double>(img1_infer.rows);

    Eigen::Matrix<double, 258, Eigen::Dynamic> filtered_feature_points0;
    Eigen::Matrix<double, 258, Eigen::Dynamic> filtered_feature_points1;
    Eigen::Matrix<double, 1, Eigen::Dynamic> filtered_feature_scores0;
    Eigen::Matrix<double, 1, Eigen::Dynamic> filtered_feature_scores1;

    // 筛选特征点，仅保留落在检测框内的特征点
    // 这里的scale_x0有冗余操作的可能，后续进行优化
    filterFeaturesByDetections(feature_points0,
                               feature_scores0,
                               detectionsOnRgb,
                               scale_x0,
                               scale_y0,
                               filtered_feature_points0,
                               filtered_feature_scores0);
    filterFeaturesByDetections(feature_points1,
                               feature_scores1,
                               detections,
                               scale_x1,
                               scale_y1,
                               filtered_feature_points1,
                               filtered_feature_scores1);

    lightglue_matches_.clear();
    if (filtered_feature_points0.cols() > 0 && filtered_feature_points1.cols() > 0)
    {
        superpointLightglue_->matching_points(filtered_feature_points0, filtered_feature_points1, lightglue_matches_);
    }

    cv::Mat match_image;
    std::vector<cv::KeyPoint> keypoints0, keypoints1;
    // 将位于框内的特征点坐标从推理图像坐标系转换回原始图像坐标系，并创建 OpenCV KeyPoint 对象
    for (size_t i = 0; i < filtered_feature_points0.cols(); ++i) {
        double score = filtered_feature_scores0(0, i);
        double x = filtered_feature_points0(0, i) * scale_x0;
        double y = filtered_feature_points0(1, i) * scale_y0;
        keypoints0.emplace_back(x, y, 8, -1, score);
    }
    for (size_t i = 0; i < filtered_feature_points1.cols(); ++i) {
        double score = filtered_feature_scores1(0, i);
        double x = filtered_feature_points1(0, i) * scale_x1;
        double y = filtered_feature_points1(1, i) * scale_y1;
        keypoints1.emplace_back(x, y, 8, -1, score); //特征点大小设置为8，方向和响应暂时不使用，保留原始得分作为响应值
    }

    // Filter matches and localize detections in the stereo module.
    const StereoLocalizationResult localization = stereoLocalizer_->localize(
        keypoints0, keypoints1, lightglue_matches_, detectionsOnRgb);
    lightglue_matches_ = localization.filteredMatches;
    perDetectionWeightedPoints = localization.pointsBody;

    if (keypoints0.size() > 0 && keypoints1.size() > 0 && !lightglue_matches_.empty()) {
        cv::drawMatches(img0_original, keypoints0, img1_original, keypoints1, lightglue_matches_, match_image,
                    cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        if (showWindow_)
        {
            cv::imshow("match_image_original_resolution", match_image);
            cv::waitKey(1);
        }
    }
    //----------------------------------------------------------------

    std::vector<std::string> perTargetLogs;
    perTargetLogs.reserve(perDetectionWeightedPoints.size());
    for (size_t i = 0; i < perDetectionWeightedPoints.size(); ++i)
    {
        const cv::Point3f& point = perDetectionWeightedPoints[i];
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
        {
            continue;
        }

        const int classId = detectionsOnRgb[i].classId;
        const std::string className = classId >= 0 && classId < static_cast<int>(classNames_.size())
            ? classNames_[classId]
            : "unknown";
        std::ostringstream oss;
        oss << "class=" << className << ", weighted3d=["
            << point.x << ", " << point.y << ", " << point.z << "]";
        perTargetLogs.push_back(oss.str());
    }
    if (!perTargetLogs.empty())
    {
        std::ostringstream log;
        log << "Frame targets(" << perTargetLogs.size() << "): ";
        for (size_t i = 0; i < perTargetLogs.size(); ++i)
        {
            if (i > 0)
            {
                log << " | ";
            }
            log << perTargetLogs[i];
        }
        ROS_INFO_STREAM(log.str());
    }
    //-------------------------------------------------------------
    }

    utils::visualizeDetection(rgbImage, detectionsOnRgb, classNames_, perDetectionWeightedPoints);
    if (showWindow_)
    {
        cv::imshow("two_stream_annotated", rgbImage);
        cv::waitKey(1);
    }
    publishCvImage(rgbImage, rgbMsg->header.stamp);

}

std::vector<Detection> YoloNode::remapDetectionsToOriginalRgb(const std::vector<Detection>& detections,
                                                              const cv::Size& rgbSize) const
{
    std::vector<Detection> remapped;
    remapped.reserve(detections.size());

    // Detections are produced in the IR/aligned image frame, so map their box
    // corners back into the original RGB image frame for feature association.
    const cv::Mat hIrToRgb = cv::Mat(H_.inv(cv::DECOMP_SVD));
    for (const auto& det : detections)
    {
        const cv::Rect& box = det.box;
        if (box.width <= 1 || box.height <= 1)
        {
            continue;
        }

        std::vector<cv::Point2f> corners(4);
        corners[0] = cv::Point2f(static_cast<float>(box.x), static_cast<float>(box.y));
        corners[1] = cv::Point2f(static_cast<float>(box.x + box.width), static_cast<float>(box.y));
        corners[2] = cv::Point2f(static_cast<float>(box.x + box.width), static_cast<float>(box.y + box.height));
        corners[3] = cv::Point2f(static_cast<float>(box.x), static_cast<float>(box.y + box.height));

        std::vector<cv::Point2f> mappedCorners;
        cv::perspectiveTransform(corners, mappedCorners, hIrToRgb);

        cv::Rect mappedBox = cv::boundingRect(mappedCorners);
        const cv::Rect imageRect(0, 0, rgbSize.width, rgbSize.height);
        mappedBox &= imageRect;
        if (mappedBox.width <= 1 || mappedBox.height <= 1)
        {
            continue;
        }

        Detection mappedDet;
        mappedDet.box = mappedBox;
        mappedDet.conf = det.conf;
        mappedDet.classId = det.classId;
        remapped.emplace_back(mappedDet);
    }

    return remapped;
}

void YoloNode::filterFeaturesByDetections(const Eigen::Matrix<double, 258, Eigen::Dynamic>& features,
                                          const Eigen::Matrix<double, 1, Eigen::Dynamic>& scores,
                                          const std::vector<Detection>& detections,
                                          double scaleX,
                                          double scaleY,
                                          Eigen::Matrix<double, 258, Eigen::Dynamic>& filteredFeatures,
                                          Eigen::Matrix<double, 1, Eigen::Dynamic>& filteredScores) const
{
    if (features.cols() <= 0 || scores.cols() <= 0 || detections.empty())
    {
        filteredFeatures.resize(258, 0);
        filteredScores.resize(1, 0);
        return;
    }

    std::vector<int> keepIndices;
    keepIndices.reserve(static_cast<size_t>(features.cols()));

    for (int col = 0; col < features.cols(); ++col)
    {
        // std::lround 用于将浮点数四舍五入到最近的整数，features(0, col) 和 features(1, col) 是特征点的坐标，乘以 scaleX 和 scaleY 将其从推理图像坐标系转换回原始图像坐标系
        const int xOriginal = static_cast<int>(std::lround(features(0, col) * scaleX));
        const int yOriginal = static_cast<int>(std::lround(features(1, col) * scaleY));

        bool insideAnyBox = false;
        for (const auto& det : detections)
        {
            if (det.box.contains(cv::Point(xOriginal, yOriginal)))
            {
                insideAnyBox = true;
                break;
            }
        }

        if (insideAnyBox)
        {
            keepIndices.push_back(col);
        }
    }

    filteredFeatures.resize(258, keepIndices.size());
    filteredScores.resize(1, keepIndices.size());
    for (int i = 0; i < static_cast<int>(keepIndices.size()); ++i)
    {
        filteredFeatures.col(i) = features.col(keepIndices[i]);
        filteredScores(0, i) = scores(0, keepIndices[i]);
    }
}

void YoloNode::publishCvImage(const cv::Mat& image, const ros::Time& stamp)
{
    std_msgs::Header header;
    header.stamp = stamp;
    sensor_msgs::ImagePtr outMsg = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();
    imagePub_.publish(outMsg);
}
}  // namespace ros_yolo
