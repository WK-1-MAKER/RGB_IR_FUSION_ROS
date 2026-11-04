#include "ros_yolo/yolo_node.h"
#include "ros_yolo/frustum_filter.h"
#include "ros_yolo/camera_geometry.h"

#include <visualization_msgs/MarkerArray.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>

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

YoloNode::BodyPose YoloNode::poseFromOdometry(const nav_msgs::Odometry& message)
{
    const auto& p = message.pose.pose.position;
    const auto& o = message.pose.pose.orientation;
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
        !std::isfinite(o.x) || !std::isfinite(o.y) || !std::isfinite(o.z) || !std::isfinite(o.w))
        throw std::invalid_argument("odometry pose contains a non-finite value");
    Eigen::Quaterniond q(o.w, o.x, o.y, o.z);
    if (!std::isfinite(q.norm()) || q.norm() <= 1e-12)
        throw std::invalid_argument("odometry orientation has zero norm");
    q.normalize();
    const Eigen::Matrix3d matrix = q.toRotationMatrix();
    BodyPose pose;
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col) pose.rotation(row, col) = matrix(row, col);
    pose.translation = cv::Vec3d(p.x, p.y, p.z);
    return pose;
}

YoloNode::YoloNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
    : nh_(nh), pnh_(pnh), it_(nh_)
{
    pnh_.param<std::string>("model_path", modelPath_, "");
    pnh_.param<std::string>("class_names_path", classNamesPath_, "");
    pnh_.param<std::string>("rgb_topic", rgbTopic_, "/camera/color/image_raw");
    pnh_.param<std::string>("ir_topic", irTopic_, "/pub_ir");
    pnh_.param<std::string>("output_topic", outputTopic_, "/two_stream/image_annotated");
    pnh_.param<std::string>("lidar_topic", lidarTopic_, "/fusion/points");
    pnh_.param<std::string>("odometry_topic", odometryTopic_, "/Odometry");
    pnh_.param<std::string>("filtered_point_cloud_topic",
                            filteredPointCloudTopic_,
                            "/fusion/points_filtered");
    pnh_.param<std::string>("cluster_bounding_boxes_topic",
                            clusterBoundingBoxesTopic_,
                            "/fusion/cluster_boxes");
    int syncQueueSize = 10;
    pnh_.param<int>("sync_queue_size", syncQueueSize, 10);
    if (syncQueueSize < 1)
        throw std::runtime_error("sync_queue_size must be positive.");
    syncQueueSize_ = static_cast<std::size_t>(syncQueueSize);

    pnh_.param<bool>("use_gpu", useGPU_, true);
    pnh_.param<bool>("show_window", showWindow_, false);
    pnh_.param<bool>("triangulate", triangulate_, true);
    pnh_.param<bool>("camera_only", cameraOnly_, false);

    pnh_.param<double>("conf_threshold", confThreshold_, 0.30);
    pnh_.param<double>("iou_threshold", iouThreshold_, 0.40);
    pnh_.param<double>("cluster_tolerance", clusterTolerance_, 0.5);
    pnh_.param<int>("cluster_min_size", clusterMinSize_, 3);
    pnh_.param<int>("cluster_max_size", clusterMaxSize_, 100000);
    pnh_.param<double>("track_buffer_seconds", trackBufferSeconds_, 1.0);
    pnh_.param<double>("world_measurement_timeout_seconds", worldMeasurementTimeoutSeconds_, 1.0);
    if (!std::isfinite(clusterTolerance_) || clusterTolerance_ <= 0.0 ||
        clusterMinSize_ < 1 || clusterMaxSize_ < clusterMinSize_ ||
        !std::isfinite(trackBufferSeconds_) || trackBufferSeconds_ <= 0.0 ||
        !std::isfinite(worldMeasurementTimeoutSeconds_) || worldMeasurementTimeoutSeconds_ <= 0.0)
    {
        throw std::runtime_error("invalid localization/tracking parameters");
    }
    EuclideanClusterConfig clusterConfig;
    clusterConfig.cluster_tolerance = clusterTolerance_;
    clusterConfig.min_cluster_size = static_cast<std::size_t>(clusterMinSize_);
    clusterConfig.max_cluster_size = static_cast<std::size_t>(clusterMaxSize_);
    lidarLocalizer_ = std::make_unique<LidarLocalizer>(clusterConfig);
    ByteTrack3DConfig trackerConfig;
    trackerConfig.track_buffer_seconds = trackBufferSeconds_;
    trackerConfig.world_measurement_timeout_seconds = worldMeasurementTimeoutSeconds_;
    trackerConfig.return_unconfirmed_tracks = true;
    tracker_ = std::make_unique<BYTETracker3D>(trackerConfig);

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
        clusterBoundingBoxesPub_ = nh_.advertise<visualization_msgs::MarkerArray>(clusterBoundingBoxesTopic_, 1);
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

    rgbSub_ = nh_.subscribe(rgbTopic_, static_cast<uint32_t>(syncQueueSize_), &YoloNode::rgbCallback, this);
    irSub_ = nh_.subscribe(irTopic_, static_cast<uint32_t>(syncQueueSize_), &YoloNode::irCallback, this);
    odometrySub_ = nh_.subscribe(odometryTopic_, static_cast<uint32_t>(syncQueueSize_),
                                 &YoloNode::odometryCallback, this);
    if (!cameraOnly_)
        lidarSub_ = nh_.subscribe(lidarTopic_, static_cast<uint32_t>(syncQueueSize_),
                                  &YoloNode::lidarCallback, this);
    ROS_INFO_STREAM("ros_yolo subscribed RGB: " << rgbTopic_);
    ROS_INFO_STREAM("ros_yolo subscribed IR : " << irTopic_);
    ROS_INFO_STREAM("ros_yolo camera_only: " << std::boolalpha << cameraOnly_);
    if (!cameraOnly_)
    {
        ROS_INFO_STREAM("ros_yolo subscribed cloud: " << lidarTopic_);
        ROS_INFO_STREAM("ros_yolo filtered cloud: " << filteredPointCloudTopic_);
        ROS_INFO_STREAM("ros_yolo cluster boxes: " << clusterBoundingBoxesTopic_ << " (camera_init)");
    }
    ROS_INFO_STREAM("ros_yolo output topic  : " << outputTopic_);

    processingThread_ = std::thread(&YoloNode::processingLoop, this);
}

YoloNode::~YoloNode()
{
    rgbSub_.shutdown();
    irSub_.shutdown();
    lidarSub_.shutdown();
    odometrySub_.shutdown();
    {
        std::lock_guard<std::mutex> lock(inputMutex_);
        stopping_ = true;
    }
    inputCondition_.notify_all();
    if (processingThread_.joinable())
    {
        processingThread_.join();
    }
}

void YoloNode::processingLoop()
{
    while (true)
    {
        nav_msgs::OdometryConstPtr anchor;
        sensor_msgs::ImageConstPtr rgb;
        sensor_msgs::ImageConstPtr ir;
        sensor_msgs::PointCloud2ConstPtr lidar;
        {
            std::unique_lock<std::mutex> lock(inputMutex_);
            inputCondition_.wait(lock, [this]() { return stopping_ || !odometryQueue_.empty(); });
            if (stopping_) return;
            const PendingOdometry pending = odometryQueue_.front();
            odometryQueue_.pop_front();
            const auto matchAfter = pending.receivedAt + std::chrono::milliseconds(90);
            // Release the cache lock while waiting so sensor callbacks can keep filling it.
            inputCondition_.wait_until(lock, matchAfter, [this]() { return stopping_; });
            if (stopping_) return;
            anchor = pending.message;

            const auto nearestWithin = [](const auto& cache,
                                          std::uint64_t referenceNanoseconds,
                                          std::uint64_t maxDeltaNanoseconds) {
                using MessagePtr = typename std::decay<decltype(cache.front())>::type;
                MessagePtr best;
                std::uint64_t bestDelta = std::numeric_limits<std::uint64_t>::max();
                std::uint64_t bestNanoseconds = std::numeric_limits<std::uint64_t>::max();
                for (const auto& candidate : cache)
                {
                    if (candidate->header.stamp.isZero()) continue;
                    const std::uint64_t candidateNanoseconds = candidate->header.stamp.toNSec();
                    const std::uint64_t delta = candidateNanoseconds >= referenceNanoseconds
                        ? candidateNanoseconds - referenceNanoseconds
                        : referenceNanoseconds - candidateNanoseconds;
                    if (delta < bestDelta || (delta == bestDelta && candidateNanoseconds < bestNanoseconds))
                    {
                        best = candidate;
                        bestDelta = delta;
                        bestNanoseconds = candidateNanoseconds;
                    }
                }
                return bestDelta < maxDeltaNanoseconds ? best : MessagePtr();
            };
            if (!anchor->header.stamp.isZero())
            {
                const std::uint64_t anchorNanoseconds = anchor->header.stamp.toNSec();
                rgb = nearestWithin(rgbCache_, anchorNanoseconds, 50000000);
                ir = nearestWithin(irCache_, anchorNanoseconds, 40000000);
                if (!cameraOnly_)
                {
                    lidar = nearestWithin(lidarCache_, anchorNanoseconds, 10000000);
                }
            }
        }

        const ros::Time anchorStamp = anchor->header.stamp;
        if (anchorStamp.isZero())
        {
            ROS_WARN("Skipping Odometry anchor with zero timestamp.");
            continue;
        }
        if (!lastAnchorTimestamp_.isZero())
        {
            if (anchorStamp == lastAnchorTimestamp_)
            {
                ROS_WARN_STREAM("Skipping duplicate Odometry anchor " << anchorStamp << ".");
                continue;
            }
            if (anchorStamp < lastAnchorTimestamp_)
            {
                tracker_->reset();
                ROS_WARN("Odometry timestamp moved backward; starting a new tracking session.");
            }
        }
        lastAnchorTimestamp_ = anchorStamp;
        if (!rgb || !ir || (!cameraOnly_ && !lidar))
        {
            std::ostringstream reason;
            reason << "Skipping Odometry anchor " << anchorStamp << ": no matching";
            bool first = true;
            if (!cameraOnly_ && !lidar)
            {
                reason << " LiDAR (< 10 ms from Odometry)";
                first = false;
            }
            if (!rgb)
            {
                reason << (first ? " " : ", ") << "RGB (< 50 ms from Odometry)";
                first = false;
            }
            if (!ir)
            {
                reason << (first ? " " : ", ") << "IR (< 40 ms from Odometry)";
            }
            reason << " message.";
            ROS_WARN_STREAM(reason.str());
            continue;
        }
        if (anchor->header.frame_id != "camera_init")
        {
            ROS_WARN_STREAM("Skipping Odometry anchor " << anchorStamp << ": frame_id is '"
                            << anchor->header.frame_id << "', expected 'camera_init'.");
            continue;
        }

        BodyPose worldFromBody;
        try
        {
            worldFromBody = poseFromOdometry(*anchor);
        }
        catch (const std::exception& e)
        {
            ROS_WARN_STREAM("Skipping Odometry anchor " << anchorStamp << ": invalid pose: " << e.what());
            continue;
        }

        try
        {
            processImagePair(rgb, ir, lidar, anchorStamp, worldFromBody);
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
            ROS_ERROR_STREAM("Odometry anchor " << anchorStamp << " processing failed: " << e.what());
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

void YoloNode::rgbCallback(const sensor_msgs::ImageConstPtr& message)
{
    std::lock_guard<std::mutex> lock(inputMutex_);
    rgbCache_.push_back(message);
    if (rgbCache_.size() > syncQueueSize_)
    {
        rgbCache_.pop_front();
    }
}

void YoloNode::irCallback(const sensor_msgs::ImageConstPtr& message)
{
    std::lock_guard<std::mutex> lock(inputMutex_);
    irCache_.push_back(message);
    if (irCache_.size() > syncQueueSize_)
    {
        irCache_.pop_front();
    }
}

void YoloNode::lidarCallback(const sensor_msgs::PointCloud2ConstPtr& message)
{
    std::lock_guard<std::mutex> lock(inputMutex_);
    lidarCache_.push_back(message);
    if (lidarCache_.size() > syncQueueSize_)
    {
        lidarCache_.pop_front();
    }
}

void YoloNode::odometryCallback(const nav_msgs::OdometryConstPtr& message)
{
    const auto receivedAt = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(inputMutex_);
        odometryQueue_.push_back(PendingOdometry{message, receivedAt});
        if (odometryQueue_.size() > syncQueueSize_)
        {
            odometryQueue_.pop_front();
        }
    }
    inputCondition_.notify_one();
}

BatchedPointCloudResult YoloNode::filterAndPublishPointClouds(
    const std::vector<Detection>& detections,
    const sensor_msgs::PointCloud2ConstPtr& cloudMsg,
    const BodyPose& worldFromBody,
    std::vector<BodyFrustum>& frustums)
{
    frustums.resize(detections.size());
    std::vector<bool> valid(detections.size(), false);
    BatchedPointCloudResult empty;
    empty.target_point_indices.resize(detections.size());
    if (!cloudMsg) return empty;

    // 射线在 camera_init 下显示；过滤仍使用 body 视锥体，显示长度不影响过滤范围。
    visualization_msgs::MarkerArray markers;
    visualization_msgs::Marker clear;
    clear.header = cloudMsg->header;
    clear.header.frame_id = "camera_init";
    clear.action = visualization_msgs::Marker::DELETEALL;
    markers.markers.push_back(clear);

    try
    {
        for (std::size_t i = 0; i < detections.size(); ++i)
        {
            if (buildBodyFrustum(detections[i].box, kRgb_, rBodyRgb_, tBodyRgb_, frustums[i]))
            {
                valid[i] = true;
                visualization_msgs::Marker marker;
                marker.header = clear.header;
                marker.ns = "frustum_rays";
                marker.id = static_cast<int>(markers.markers.size() - 1);
                marker.type = visualization_msgs::Marker::LINE_LIST;
                marker.action = visualization_msgs::Marker::ADD;
                marker.pose.orientation.w = 1.0;
                marker.scale.x = 0.03;
                marker.color.g = 1.0;
                marker.color.a = 1.0;
                marker.lifetime = ros::Duration(0);  // 暂停 rosbag 时保留，下一帧通过 DELETEALL 更新。
                const cv::Vec3d originWorld = worldFromBody.transform(frustums[i].origin);
                for (const auto& ray : frustums[i].rays)
                {
                    geometry_msgs::Point start, end;
                    start.x = originWorld[0];
                    start.y = originWorld[1];
                    start.z = originWorld[2];
                    const cv::Vec3d tip = worldFromBody.transform(frustums[i].origin + 30.0 * ray);
                    end.x = tip[0];
                    end.y = tip[1];
                    end.z = tip[2];
                    marker.points.push_back(start);
                    marker.points.push_back(end);
                }
                markers.markers.push_back(marker);
            }
        }
    }
    catch (const std::exception& e)
    {
        ROS_WARN_STREAM_THROTTLE(2.0, "Cannot build body frustums: " << e.what());
    }
    frustumRaysPub_.publish(markers);

    BatchedPointCloudResult result;
    try
    {
        result = filterPointCloudBatch(*cloudMsg, frustums);
        for (std::size_t i = 0; i < result.filtered_clouds.size(); ++i)
        {
            if (valid[i]) filteredPointCloudPub_.publish(result.filtered_clouds[i]);
        }
    }
    catch (const std::exception& e)
    {
        ROS_WARN_STREAM_THROTTLE(2.0, "Cannot filter PointCloud2: " << e.what());
        result.target_point_indices.resize(detections.size());
    }
    return result;
}

void YoloNode::processImagePair(const sensor_msgs::ImageConstPtr& rgbMsg,
                                const sensor_msgs::ImageConstPtr& irMsg,
                                const sensor_msgs::PointCloud2ConstPtr& lidarMsg,
                                const ros::Time& frameStamp,
                                const BodyPose& worldFromBody)
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

    std::vector<BodyFrustum> frustums;
    const BatchedPointCloudResult pointCloud =
        filterAndPublishPointClouds(detectionsOnRgb, lidarMsg, worldFromBody, frustums);

    std::vector<ByteTrack3DDetection> trackerInput;
    trackerInput.reserve(detectionsOnRgb.size());
    std::vector<AxisAlignedBounds> worldBounds(detectionsOnRgb.size());
    std::vector<bool> validWorldBounds(detectionsOnRgb.size(), false);
    for (std::size_t i = 0; i < detectionsOnRgb.size(); ++i)
    {
        ByteTrack3DDetection detection;
        detection.box = detectionsOnRgb[i].box;
        detection.score = detectionsOnRgb[i].conf;
        detection.class_id = detectionsOnRgb[i].classId;

        if (lidarMsg && i < pointCloud.target_point_indices.size())
        {
            try
            {
                const auto localized = lidarLocalizer_->localize(pointCloud.candidate_points,
                                                                  pointCloud.source_offsets,
                                                                  pointCloud.target_point_indices[i]);
                if (localized.valid)
                {
                    const cv::Vec3d world = worldFromBody.transform(localized.centroid_body);
                    if (std::isfinite(world[0]) && std::isfinite(world[1]))
                    {
                        detection.position = cv::Point2d(world[0], world[1]);
                        std::vector<cv::Vec3d> worldClusterPoints;
                        worldClusterPoints.reserve(localized.cluster_point_indices.size());
                        for (const std::size_t candidateIndex : localized.cluster_point_indices)
                        {
                            worldClusterPoints.push_back(worldFromBody.transform(pointCloud.candidate_points.at(candidateIndex)));
                        }
                        worldBounds[i] = axisAlignedBounds(worldClusterPoints);
                        validWorldBounds[i] = true;
                    }
                }
            }
            catch (const std::exception& e)
            {
                ROS_WARN_STREAM_THROTTLE(2.0, "LiDAR localization failed for detection " << i << ": " << e.what());
            }
        }
        trackerInput.push_back(detection);
    }
    const auto tracked = tracker_->update(trackerInput, frameStamp.toSec());
    std::vector<TrackAnnotation> trackAnnotations(detectionsOnRgb.size());
    std::vector<const ByteTrack3DResult*> tracksByDetection(detectionsOnRgb.size(), nullptr);
    for (const auto& track : tracked)
    {
        if (track.detection_index >= trackAnnotations.size()) continue;
        tracksByDetection[track.detection_index] = &track;
        auto& annotation = trackAnnotations[track.detection_index];
        annotation.track_id = track.track_id;
        annotation.world_position_valid = std::isfinite(track.position.x) && std::isfinite(track.position.y);
        if (annotation.world_position_valid) annotation.world_position = track.position;
    }
    std::ostringstream trackLog;
    trackLog << std::fixed << std::setprecision(3);
    bool hasTrackedTargets = false;
    for (std::size_t i = 0; i < tracksByDetection.size(); ++i)
    {
        const auto* track = tracksByDetection[i];
        if (!track) continue;

        if (hasTrackedTargets) trackLog << '\n';
        trackLog << "id=" << track->track_id;
        hasTrackedTargets = true;
        if (std::isfinite(track->position.x) && std::isfinite(track->position.y) &&
            std::isfinite(track->velocity.x) && std::isfinite(track->velocity.y))
        {
            trackLog << " X=" << track->position.x << " m Y=" << track->position.y
                     << " m Vx=" << track->velocity.x << " m/s Vy=" << track->velocity.y << " m/s";
        }
        else
        {
            trackLog << " X=NA Y=NA Vx=NA Vy=NA";
        }
    }
    if (hasTrackedTargets) ROS_INFO_STREAM(trackLog.str());
    publishClusterBoundingBoxes(worldBounds, validWorldBounds, trackAnnotations, frameStamp);

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

    utils::visualizeDetection(rgbImage, detectionsOnRgb, classNames_, perDetectionWeightedPoints,
                              trackAnnotations);
    if (showWindow_)
    {
        cv::imshow("two_stream_annotated", rgbImage);
        cv::waitKey(1);
    }
    publishCvImage(rgbImage, rgbMsg->header.stamp);

}

void YoloNode::publishClusterBoundingBoxes(const std::vector<AxisAlignedBounds>& bounds,
                                           const std::vector<bool>& valid,
                                           const std::vector<TrackAnnotation>& tracks,
                                           const ros::Time& stamp)
{
    if (cameraOnly_) return;

    visualization_msgs::MarkerArray markers;
    visualization_msgs::Marker clear;
    clear.header.stamp = stamp;
    clear.header.frame_id = "camera_init";
    clear.action = visualization_msgs::Marker::DELETEALL;
    markers.markers.push_back(clear);

    const std::size_t count = std::min(bounds.size(), std::min(valid.size(), tracks.size()));
    for (std::size_t i = 0; i < count; ++i)
    {
        if (!valid[i]) continue;
        const AxisAlignedBounds& box = bounds[i];
        const cv::Vec3d center = (box.maximum + box.minimum) * 0.5;

        visualization_msgs::Marker boxMarker;
        boxMarker.header = clear.header;
        boxMarker.ns = "cluster_boxes";
        boxMarker.id = static_cast<int>(2 * i);
        boxMarker.type = visualization_msgs::Marker::LINE_LIST;
        boxMarker.action = visualization_msgs::Marker::ADD;
        boxMarker.pose.orientation.w = 1.0;
        boxMarker.scale.x = 0.02;  // Line width only; corner coordinates remain exact.
        boxMarker.color.r = 0.1;
        boxMarker.color.g = 0.55;
        boxMarker.color.b = 1.0;
        boxMarker.color.a = 1.0;
        // Connect each corner to its three neighbours, emitting each edge once.
        // Flat clusters retain their exact bounds without expanding the box.
        for (int corner = 0; corner < 8; ++corner)
        {
            for (int axis = 0; axis < 3; ++axis)
            {
                if (corner & (1 << axis)) continue;
                for (const int endpoint : {corner, corner | (1 << axis)})
                {
                    geometry_msgs::Point point;
                    point.x = (endpoint & 1) ? box.maximum[0] : box.minimum[0];
                    point.y = (endpoint & 2) ? box.maximum[1] : box.minimum[1];
                    point.z = (endpoint & 4) ? box.maximum[2] : box.minimum[2];
                    boxMarker.points.push_back(point);
                }
            }
        }
        boxMarker.lifetime = ros::Duration(0);
        markers.markers.push_back(boxMarker);

        if (tracks[i].track_id <= 0) continue;
        visualization_msgs::Marker label;
        label.header = clear.header;
        label.ns = "cluster_box_ids";
        label.id = static_cast<int>(2 * i + 1);
        label.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        label.action = visualization_msgs::Marker::ADD;
        label.pose.position.x = center[0];
        label.pose.position.y = center[1];
        label.pose.position.z = box.maximum[2] + 0.25;
        label.pose.orientation.w = 1.0;
        label.scale.z = 0.3;
        label.color.r = 1.0;
        label.color.g = 1.0;
        label.color.b = 1.0;
        label.color.a = 1.0;
        label.text = std::to_string(tracks[i].track_id);
        label.lifetime = ros::Duration(0);
        markers.markers.push_back(label);
    }
    clusterBoundingBoxesPub_.publish(markers);
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
