#include <ros/ros.h>
#include "ros_yolo/yolo_node.h"

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