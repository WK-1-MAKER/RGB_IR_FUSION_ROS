#include "ros_yolo/yolo_node.h"

#include <boost/bind.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
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

YoloNode::YoloNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
    : nh_(nh), pnh_(pnh), it_(nh_)
{
    pnh_.param<std::string>("model_path", modelPath_, "");
    pnh_.param<std::string>("class_names_path", classNamesPath_, "");
    pnh_.param<std::string>("rgb_topic", rgbTopic_, "/camera/color/image_raw");
    pnh_.param<std::string>("ir_topic", irTopic_, "/pub_ir");
    pnh_.param<std::string>("output_topic", outputTopic_, "/two_stream/image_annotated");

    pnh_.param<bool>("use_gpu", useGPU_, true);
    pnh_.param<bool>("show_window", showWindow_, false);

    pnh_.param<double>("conf_threshold", confThreshold_, 0.30);
    pnh_.param<double>("iou_threshold", iouThreshold_, 0.40);

    pnh_.param<std::string>("config_path", config_path_, "");
    pnh_.param<std::string>("model_dir", model_dir_, "");

    compute_H();
    compute_F();

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

    configs_ = std::make_unique<Configs>(config_path_, model_dir_);
    superpoint_ = std::make_shared<SuperPoint>(configs_->superpoint_config);
    superpointLightglue_ = std::make_shared<SuperPointLightGlue>(configs_->superpoint_lightglue_config);

    if (!superpoint_->build()) {
        throw std::runtime_error("Failed to build SuperPoint TensorRT engine.");
    }

    if (!superpointLightglue_->build()) {
        throw std::runtime_error("Failed to build SuperPoint-LightGlue TensorRT engine.");
    }

    rgbSub_.subscribe(nh_, rgbTopic_, 1);
    irSub_.subscribe(nh_, irTopic_, 1);
    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), rgbSub_, irSub_);
    sync_->registerCallback(boost::bind(&YoloNode::syncImageCallback, this, _1, _2));
    if (showWindow_)
    {
        annotatedSub_ = it_.subscribe(outputTopic_, 1, &YoloNode::annotatedImageCallback, this);
    }
    ROS_INFO_STREAM("ros_yolo subscribed RGB: " << rgbTopic_);
    ROS_INFO_STREAM("ros_yolo subscribed IR : " << irTopic_);
    ROS_INFO_STREAM("ros_yolo output topic  : " << outputTopic_);
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

void YoloNode::compute_F()
{
    const cv::Matx33d tCross(0.0, -t_[2], t_[1],
                          t_[2], 0.0, -t_[0],
                          -t_[1], t_[0], 0.0);
    const cv::Matx33d E_ = tCross * r_;
    F_ = kIr_.inv(cv::DECOMP_SVD).t() * E_ * kRgb_.inv(cv::DECOMP_SVD);
}

void YoloNode::compute_H()
{
    const double refDepth = 3.0;
    const cv::Vec3d n(0.0, 0.0, 1.0);
    const cv::Matx33d tOuter(
        t_[0] * n[0], t_[0] * n[1], t_[0] * n[2],
        t_[1] * n[0], t_[1] * n[1], t_[1] * n[2],
        t_[2] * n[0], t_[2] * n[1], t_[2] * n[2]);

    H_ = kIr_ * (r_ - tOuter * (1.0 / refDepth)) * kRgb_.inv(cv::DECOMP_SVD);
    //ROS_INFO("Homography initialized for RGB->IR alignment.");
}

void YoloNode::updateRectification(const cv::Size& imageSize)
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
        << r_(0, 0), r_(0, 1), r_(0, 2),
           r_(1, 0), r_(1, 1), r_(1, 2),
           r_(2, 0), r_(2, 1), r_(2, 2));
    const cv::Mat translation = (cv::Mat_<double>(3, 1) << t_(0), t_(1), t_(2));

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
void YoloNode::rectifyMatchedPoints(const std::vector<cv::Point2f>& pointsRgb,
                                    const std::vector<cv::Point2f>& pointsIr,
                                    std::vector<cv::Point2f>& rectifiedRgb,
                                    std::vector<cv::Point2f>& rectifiedIr) const
{
    rectifiedRgb.clear();
    rectifiedIr.clear();

    if (!rectificationReady_)
    {
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

void YoloNode::syncImageCallback(const sensor_msgs::ImageConstPtr& rgbMsg,
                                 const sensor_msgs::ImageConstPtr& irMsg)
{
    cv::Mat rgbImage;
    cv::Mat irImage;
    //try 
    //{
        rgbImage = toBgrImage(rgbMsg);
        irImage = toBgrImage(irMsg);
    //}
    /* catch (const cv_bridge::Exception& e)
    {
        ROS_ERROR_STREAM("Sync cv_bridge exception: " << e.what());
        return;
    } */

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
        updateRectification(rgbImage.size());
    }
    catch (const cv::Exception& e)
    {
        rectificationReady_ = false;
        kRgbTri_ = kRgb_;
        kIrTri_ = kIr_;
        rTri_ = r_;
        tTri_ = t_;
        rRectRgb_ = cv::Matx33d::eye();
        ROS_WARN_STREAM_THROTTLE(1.0, "Rectification disabled for this frame (OpenCV): " << e.what());
    }
    catch (const std::exception& e)
    {
        rectificationReady_ = false;
        kRgbTri_ = kRgb_;
        kIrTri_ = kIr_;
        rTri_ = r_;
        tTri_ = t_;
        rRectRgb_ = cv::Matx33d::eye();
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

    // 剔除不满足极线约束的匹配点对（只过滤 matches，保持 keypoints 索引不变）
    correct_points(F_, keypoints0, keypoints1, lightglue_matches_, 5.0);

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

    //--------------------三角测量深度估计----------------------------
    std::vector<cv::Point2f> matchedRgbPoints;
    std::vector<cv::Point2f> matchedIrPoints;
    matchedRgbPoints.reserve(lightglue_matches_.size());
    matchedIrPoints.reserve(lightglue_matches_.size());
    for (const auto& match : lightglue_matches_)
    {
        if (match.queryIdx < 0 || match.trainIdx < 0 ||
            match.queryIdx >= static_cast<int>(keypoints0.size()) ||
            match.trainIdx >= static_cast<int>(keypoints1.size()))
        {
            continue;
        }

        matchedRgbPoints.push_back(keypoints0[match.queryIdx].pt);
        matchedIrPoints.push_back(keypoints1[match.trainIdx].pt);
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
        cv::Point3f point3d(static_cast<float>(pointLeftVec[0]),
                            static_cast<float>(pointLeftVec[1]),
                            static_cast<float>(pointLeftVec[2]));
        if (!std::isfinite(point3d.x) || !std::isfinite(point3d.y) || !std::isfinite(point3d.z))
        {
            continue;
        }
        if (point3d.z <= 0.0f || point3d.z > 30.0f)
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
                // 原始坐标系下的3D位置和矫正坐标系下的重投影误差
                pointsPerDetection[static_cast<size_t>(bestDetIdx)].push_back({point3d, pointReprojError});
            }
        }
    }

    //std::cout << "Triangulated " << triangulatedPoints.size() << " 3D points from " << pairCount << " matched pairs." << std::endl;

    std::vector<std::string> perTargetLogs;
    perTargetLogs.reserve(detectionsOnRgb.size());
    std::vector<cv::Point3f> perDetectionWeightedPoints(
        detectionsOnRgb.size(),
        cv::Point3f(std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::quiet_NaN()));
    
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

        perDetectionWeightedPoints[detIdx] = weightedPoint;

        std::string targetClassName = "unknown";
        const int classId = detectionsOnRgb[detIdx].classId;
        if (classId >= 0 && classId < static_cast<int>(classNames_.size()))
        {
            targetClassName = classNames_[classId];
        }

        std::ostringstream oss;
        oss << "class=" << targetClassName << ", weighted3d=["
            << weightedPoint.x << ", " << weightedPoint.y << ", " << weightedPoint.z
            << "]";
        perTargetLogs.push_back(oss.str());
    }

    // 打印每个检测目标的类别和对应的加权三维位置估计信息
    if (!perTargetLogs.empty())
    {
        std::ostringstream frameLog;
        frameLog << "Frame targets(" << perTargetLogs.size() << "): ";
        for (size_t i = 0; i < perTargetLogs.size(); ++i)
        {
            if (i > 0)
            {
                frameLog << " | ";
            }
            frameLog << perTargetLogs[i];
        }
        ROS_INFO_STREAM(frameLog.str());
    }
    //-------------------------------------------------------------

    utils::visualizeDetection(rgbImage, detectionsOnRgb, classNames_, perDetectionWeightedPoints);
    publishCvImage(rgbImage, rgbMsg->header.stamp);

}

std::vector<Detection> YoloNode::remapDetectionsToOriginalRgb(const std::vector<Detection>& detections,
                                                              const cv::Size& rgbSize) const
{
    std::vector<Detection> remapped;
    remapped.reserve(detections.size());

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

void YoloNode::annotatedImageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    if (!showWindow_)
    {
        return;
    }

    try
    {
        cv::Mat annotated = toBgrImage(msg);
        if (annotated.empty())
        {
            return;
        }
        cv::imshow("two_stream_annotated", annotated);
        cv::waitKey(1);
    }
    catch (const cv_bridge::Exception& e)
    {
        ROS_ERROR_STREAM("Annotated cv_bridge exception: " << e.what());
    }
}

void YoloNode::correct_points(const cv::Matx33d& F,
                              const std::vector<cv::KeyPoint>& points0,
                              const std::vector<cv::KeyPoint>& points1,
                              std::vector<cv::DMatch>& matches,
                              double threshold)
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

        cv::Matx31d epipolar_line_ir = F * pt0;
        cv::Matx31d epipolar_line_rgb = F.t() * pt1;
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

cv::Point2f YoloNode::normalize_pixel(const cv::Matx33d& K, const cv::Point2f& pt) const
{
    cv::Matx31d normalized_pt = K.inv(cv::DECOMP_SVD) * cv::Matx31d(pt.x, pt.y, 1.0);
    return cv::Point2f(normalized_pt(0) / normalized_pt(2), normalized_pt(1) / normalized_pt(2));
}

cv::Point3f YoloNode::triangulate_svd(const cv::Matx34d& P1,
                                      const cv::Matx34d& P2,
                                      const cv::Point2f& ptn1,
                                      const cv::Point2f& ptn2) const
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
        const float nan = std::numeric_limits<float>::quiet_NaN();
        return cv::Point3f(nan, nan, nan);
    }

    return cv::Point3f(static_cast<float>(xh[0] / xh[3]),
                       static_cast<float>(xh[1] / xh[3]),
                       static_cast<float>(xh[2] / xh[3]));
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

cv::Point3f YoloNode::triangulate(const cv::Point2f& pt1, const cv::Point2f& pt2, int iters) const
{
    const cv::Matx34d P1_ext(1.0, 0.0, 0.0, 0.0,
                             0.0, 1.0, 0.0, 0.0,
                             0.0, 0.0, 1.0, 0.0);
    const cv::Matx34d P2_ext(rTri_(0, 0), rTri_(0, 1), rTri_(0, 2), tTri_(0),
                             rTri_(1, 0), rTri_(1, 1), rTri_(1, 2), tTri_(1),
                             rTri_(2, 0), rTri_(2, 1), rTri_(2, 2), tTri_(2));
    cv::Point2f ptn1 = normalize_pixel(kRgbTri_, pt1);
    cv::Point2f ptn2 = normalize_pixel(kIrTri_, pt2);
    cv::Point3f point3d_svd = triangulate_svd(P1_ext, P2_ext, ptn1, ptn2);
    if (!std::isfinite(point3d_svd.x) || !std::isfinite(point3d_svd.y) || !std::isfinite(point3d_svd.z))
    {
        return point3d_svd;
    }

    const cv::Matx34d P1 = kRgbTri_ * P1_ext;
    const cv::Matx34d P2 = kIrTri_ * P2_ext;
    cv::Point3f point3d_optimized = point3d_svd;
    cv::Vec4d last_error(0.0, 0.0, 0.0, 0.0);

    while (iters--) {
        cv::Point2f reproj_error_rgb = reprojection_error(point3d_optimized, P1, pt1);
        cv::Point2f reproj_error_ir = reprojection_error(point3d_optimized, P2, pt2);
        if (!std::isfinite(reproj_error_rgb.x) || !std::isfinite(reproj_error_rgb.y) ||
            !std::isfinite(reproj_error_ir.x) || !std::isfinite(reproj_error_ir.y))
        {
            break;
        }
        cv::Vec4d error = cv::Vec4d(reproj_error_rgb.x, reproj_error_rgb.y, reproj_error_ir.x, reproj_error_ir.y);
        cv::Matx43d J;
        const float eps = 1e-6f;
        for (int i = 0; i < 3; ++i) {
            cv::Vec3d dx(0.0, 0.0, 0.0);
            dx[i] = eps;
            J(0, i) = (reprojection_error(cv::Point3f(point3d_optimized.x + dx[0], point3d_optimized.y + dx[1], point3d_optimized.z + dx[2]), P1, pt1).x - error[0]) / eps;
            J(1, i) = (reprojection_error(cv::Point3f(point3d_optimized.x + dx[0], point3d_optimized.y + dx[1], point3d_optimized.z + dx[2]), P1, pt1).y - error[1]) / eps;
            J(2, i) = (reprojection_error(cv::Point3f(point3d_optimized.x + dx[0], point3d_optimized.y + dx[1], point3d_optimized.z + dx[2]), P2, pt2).x - error[2]) / eps;
            J(3, i) = (reprojection_error(cv::Point3f(point3d_optimized.x + dx[0], point3d_optimized.y + dx[1], point3d_optimized.z + dx[2]), P2, pt2).y - error[3]) / eps;
        }
        cv::Matx33d JtJ = J.t() * J;
        cv::Matx31d Jte = J.t() * error;
        cv::Matx31d delta = JtJ.inv(cv::DECOMP_SVD) * Jte;
        point3d_optimized = cv::Point3f(point3d_optimized.x - static_cast<float>(delta(0)),
                                    point3d_optimized.y - static_cast<float>(delta(1)),
                                    point3d_optimized.z - static_cast<float>(delta(2)));
        cv::Vec4d err_diff = error - last_error;
        double err_norm = std::sqrt(error[0] * error[0] + error[1] * error[1] + error[2] * error[2] + error[3] * error[3]);
        double err_diff_norm = std::sqrt(err_diff[0] * err_diff[0] + err_diff[1] * err_diff[1] + err_diff[2] * err_diff[2] + err_diff[3] * err_diff[3]);
        double last_err_norm = std::sqrt(last_error[0] * last_error[0] + last_error[1] * last_error[1] + last_error[2] * last_error[2] + last_error[3] * last_error[3]);
        if ((last_err_norm > 1e-10 && err_diff_norm / last_err_norm < 0.001) || err_norm < 0.5) {
            break;
        }
        last_error = error;
    }
    return point3d_optimized;  
}

void YoloNode::publishCvImage(const cv::Mat& image, const ros::Time& stamp)
{
    std_msgs::Header header;
    header.stamp = stamp;
    sensor_msgs::ImagePtr outMsg = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();
    imagePub_.publish(outMsg);
}
}  // namespace ros_yolo
