#include "ros_yolo/triangulation_options.h"

#include <gtest/gtest.h>

TEST(TriangulationOptionsTest, UsesTheLaunchParameterName)
{
    EXPECT_STREQ(ros_yolo::TriangulationOptions::kParamName, "traingulate");
}

TEST(TriangulationOptionsTest, IsEnabledByDefault)
{
    const ros_yolo::TriangulationOptions options;
    EXPECT_TRUE(options.enabled);
}

TEST(TriangulationOptionsTest, AllowsExplicitDisable)
{
    ros_yolo::TriangulationOptions options;
    options.enabled = false;
    EXPECT_FALSE(options.enabled);
}
