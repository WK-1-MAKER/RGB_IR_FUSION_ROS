#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include <opencv2/opencv.hpp>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "ros_yolo/detector.h"
#include "ros_yolo/utils.h"

namespace ros_yolo
{
class YoloNode
{
public:
    YoloNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
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

        initializeCalibration();

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

        rgbSub_ = it_.subscribe(rgbTopic_, 1, &YoloNode::rgbImageCallback, this);
        irSub_ = it_.subscribe(irTopic_, 1, &YoloNode::irImageCallback, this);
        if (showWindow_)
        {
            annotatedSub_ = it_.subscribe(outputTopic_, 1, &YoloNode::annotatedImageCallback, this);
        }
        ROS_INFO_STREAM("ros_yolo subscribed RGB: " << rgbTopic_);
        ROS_INFO_STREAM("ros_yolo subscribed IR : " << irTopic_);
        ROS_INFO_STREAM("ros_yolo output topic  : " << outputTopic_);
    }

private:
    static cv::Mat toBgrImage(const sensor_msgs::ImageConstPtr& msg)
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

    void initializeCalibration()
    {
        kRgb_ = cv::Matx33d(606.97900390625, 0.0, 320.3143615722656,
                            0.0, 607.1318359375, 247.97427368164062,
                            0.0, 0.0, 1.0);

        kIr_ = cv::Matx33d(593.946114205786, 0.0, 322.242307522658,
                           0.0, 592.545920338569, 264.298408236226,
                           0.0, 0.0, 1.0);

        r_ = cv::Matx33d::eye();
        t_ = cv::Vec3d(-0.036, 0.03, -0.009);

        const double refDepth = 3.0;
        const cv::Vec3d n(0.0, 0.0, 1.0);
        const cv::Matx33d tOuter(
            t_[0] * n[0], t_[0] * n[1], t_[0] * n[2],
            t_[1] * n[0], t_[1] * n[1], t_[1] * n[2],
            t_[2] * n[0], t_[2] * n[1], t_[2] * n[2]);

        hRgbToIr_ = kIr_ * (r_ - tOuter * (1.0 / refDepth)) * kRgb_.inv(cv::DECOMP_SVD);
        //ROS_INFO("Homography initialized for RGB->IR alignment.");
    }

    void irImageCallback(const sensor_msgs::ImageConstPtr& msg)
    {
        try
        {
            cv::Mat irImage = toBgrImage(msg);
            if (irImage.empty())
            {
                ROS_WARN("Received empty IR image frame.");
                return;
            }

            std::lock_guard<std::mutex> lock(irMutex_);
            latestIrImage_ = irImage;
            latestIrStamp_ = msg->header.stamp;
        }
        catch (const cv_bridge::Exception& e)
        {
            ROS_ERROR_STREAM("IR cv_bridge exception: " << e.what());
        }
    }

    void rgbImageCallback(const sensor_msgs::ImageConstPtr& msg)
    {
        cv::Mat rgbImage;
        try
        {
            rgbImage = toBgrImage(msg);
        }
        catch (const cv_bridge::Exception& e)
        {
            ROS_ERROR_STREAM("RGB cv_bridge exception: " << e.what());
            return;
        }

        if (rgbImage.empty())
        {
            ROS_WARN("Received empty RGB image frame.");
            return;
        }

        cv::Mat irImage;
        {
            std::lock_guard<std::mutex> lock(irMutex_);
            if (!latestIrImage_.empty())
            {
                irImage = latestIrImage_.clone();
            }
        }

        if (irImage.empty())
        {
            ROS_WARN_THROTTLE(2.0, "No IR frame cached yet, skip RGB frame.");
            return;
        }

        cv::Mat rgbAligned;
        cv::warpPerspective(rgbImage,
                            rgbAligned,
                            cv::Mat(hRgbToIr_),
                            irImage.size(),
                            cv::INTER_LINEAR,
                            cv::BORDER_CONSTANT,
                            cv::Scalar(0, 0, 0));

        if (rgbAligned.empty())
        {
            ROS_WARN_THROTTLE(2.0, "RGB-IR alignment failed, skip frame.");
            return;
        }

        std::vector<Detection> detections = detector_->detect(
            rgbAligned,
            irImage,
            static_cast<float>(confThreshold_),
            static_cast<float>(iouThreshold_));

        std::vector<Detection> detectionsOnRgb = remapDetectionsToOriginalRgb(detections, rgbImage.size());

        utils::visualizeDetection(rgbImage, detectionsOnRgb, classNames_);
        publishCvImage(rgbImage, msg->header.stamp);

    }

    std::vector<Detection> remapDetectionsToOriginalRgb(const std::vector<Detection>& detections,
                                                        const cv::Size& rgbSize) const
    {
        std::vector<Detection> remapped;
        remapped.reserve(detections.size());

        const cv::Mat hIrToRgb = cv::Mat(hRgbToIr_.inv(cv::DECOMP_SVD));
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

    void annotatedImageCallback(const sensor_msgs::ImageConstPtr& msg)
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

    void publishCvImage(const cv::Mat& image, const ros::Time& stamp)
    {
        std_msgs::Header header;
        header.stamp = stamp;
        sensor_msgs::ImagePtr outMsg = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();
        imagePub_.publish(outMsg);
    }

private:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber rgbSub_;
    image_transport::Subscriber irSub_;
    image_transport::Subscriber annotatedSub_;
    image_transport::Publisher imagePub_;

    std::unique_ptr<YoloDetector> detector_;
    std::vector<std::string> classNames_;

    std::string modelPath_;
    std::string classNamesPath_;
    std::string rgbTopic_;
    std::string irTopic_;
    std::string outputTopic_;

    std::mutex irMutex_;
    cv::Mat latestIrImage_;
    ros::Time latestIrStamp_;

    cv::Matx33d kRgb_;
    cv::Matx33d kIr_;
    cv::Matx33d r_;
    cv::Vec3d t_;
    cv::Matx33d hRgbToIr_;

    bool useGPU_{};
    bool showWindow_{};

    double confThreshold_{};
    double iouThreshold_{};
};
}  // namespace ros_yolo

int main(int argc, char** argv)
{
    ros::init(argc, argv, "ros_yolo_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    try
    {
        ros_yolo::YoloNode node(nh, pnh);
        ros::spin();
    }
    catch (const std::exception& e)
    {
        ROS_ERROR_STREAM("Failed to start ros_yolo_node: " << e.what());
        return -1;
    }

    return 0;
}
