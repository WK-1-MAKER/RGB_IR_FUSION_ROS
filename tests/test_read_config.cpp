#include "lightglue/read_config.h"

#include <gtest/gtest.h>

#include <fstream>
#include <string>

namespace
{
const char* kConfigPath = "/tmp/two_stream_test_config.yaml";

void writeConfig(const std::string& body)
{
    std::ofstream out(kConfigPath);
    out << body;
}
}  // namespace

TEST(ReadConfigTest, ParsesCameraCalibrationConfig)
{
    writeConfig(R"yaml(
superpoint:
  max_keypoints: 3000
  keypoint_threshold: 0.01
  remove_borders: 4
  input_tensor_names: ["image"]
  output_tensor_names: ["scores", "descriptors"]
  onnx_file: "superpoint_v1.onnx"
  engine_file: "superpoint_v1.engine"
  dla_core: -1

superpoint_lightglue:
  image_width: 320
  image_height: 240
  filter_threshold: 0.1
  depth_confidence: -1
  width_confidence: -1
  input_tensor_names: ["keypoints0", "keypoints1", "descriptors0", "descriptors1"]
  output_tensor_names: ["scores"]
  onnx_file: "superpoint_lightglue.onnx"
  engine_file: "superpoint_lightglue.engine"
  dla_core: -1

cameras:
  rgb:
    intrinsic: [606.97900390625, 0.0, 320.3143615722656,
                0.0, 607.1318359375, 247.97427368164062,
                0.0, 0.0, 1.0]
    distortion: [0.0, 0.0, 0.0, 0.0, 0.0]
    rotation: [1.0, 0.0, 0.0,
               0.0, 1.0, 0.0,
               0.0, 0.0, 1.0]
    translation: [0.0, 0.0, 0.0]
  ir:
    intrinsic: [593.946114205786, 0.0, 322.242307522658,
                0.0, 592.545920338569, 264.298408236226,
                0.0, 0.0, 1.0]
    distortion: [0.0, 0.0, 0.0, 0.0, 0.0]
    rotation: [1.0, 0.0, 0.0,
               0.0, 1.0, 0.0,
               0.0, 0.0, 1.0]
    translation: [-0.036, 0.03, -0.009]
)yaml");

    const Configs configs(kConfigPath, "/tmp/models");

    ASSERT_EQ(configs.camera_config.rgb.intrinsic.size(), 9u);
    ASSERT_EQ(configs.camera_config.rgb.distortion.size(), 5u);
    ASSERT_EQ(configs.camera_config.rgb.rotation.size(), 9u);
    ASSERT_EQ(configs.camera_config.rgb.translation.size(), 3u);
    EXPECT_DOUBLE_EQ(configs.camera_config.rgb.intrinsic[0], 606.97900390625);
    EXPECT_DOUBLE_EQ(configs.camera_config.ir.intrinsic[0], 593.946114205786);
    EXPECT_DOUBLE_EQ(configs.camera_config.ir.translation[0], -0.036);
    EXPECT_DOUBLE_EQ(configs.camera_config.ir.translation[1], 0.03);
    EXPECT_DOUBLE_EQ(configs.camera_config.ir.translation[2], -0.009);
}

TEST(ReadConfigTest, RejectsWrongCameraMatrixLength)
{
    writeConfig(R"yaml(
superpoint:
  max_keypoints: 3000
  keypoint_threshold: 0.01
  remove_borders: 4
  input_tensor_names: ["image"]
  output_tensor_names: ["scores", "descriptors"]
  onnx_file: "superpoint_v1.onnx"
  engine_file: "superpoint_v1.engine"
  dla_core: -1

superpoint_lightglue:
  image_width: 320
  image_height: 240
  filter_threshold: 0.1
  depth_confidence: -1
  width_confidence: -1
  input_tensor_names: ["keypoints0", "keypoints1", "descriptors0", "descriptors1"]
  output_tensor_names: ["scores"]
  onnx_file: "superpoint_lightglue.onnx"
  engine_file: "superpoint_lightglue.engine"
  dla_core: -1

cameras:
  rgb:
    intrinsic: [1.0, 0.0, 0.0]
    distortion: [0.0, 0.0, 0.0, 0.0, 0.0]
    rotation: [1.0, 0.0, 0.0,
               0.0, 1.0, 0.0,
               0.0, 0.0, 1.0]
    translation: [0.0, 0.0, 0.0]
  ir:
    intrinsic: [593.946114205786, 0.0, 322.242307522658,
                0.0, 592.545920338569, 264.298408236226,
                0.0, 0.0, 1.0]
    distortion: [0.0, 0.0, 0.0, 0.0, 0.0]
    rotation: [1.0, 0.0, 0.0,
               0.0, 1.0, 0.0,
               0.0, 0.0, 1.0]
    translation: [-0.036, 0.03, -0.009]
)yaml");

    EXPECT_THROW(
        {
            const Configs configs(kConfigPath, "/tmp/models");
            (void) configs;
        },
        std::runtime_error);
}

TEST(ReadConfigTest, PackageConfigContainsCameraCalibration)
{
    const std::string configPath = std::string(TWO_STREAM_SOURCE_DIR) + "/config/config.yaml";
    const std::string modelDir = std::string(TWO_STREAM_SOURCE_DIR) + "/weights";

    const Configs configs(configPath, modelDir);

    ASSERT_EQ(configs.camera_config.rgb.intrinsic.size(), 9u);
    ASSERT_EQ(configs.camera_config.ir.intrinsic.size(), 9u);
    ASSERT_EQ(configs.camera_config.rgb.distortion.size(), 5u);
    ASSERT_EQ(configs.camera_config.ir.distortion.size(), 5u);
    EXPECT_DOUBLE_EQ(configs.camera_config.rgb.translation[0], 0.0);
    EXPECT_DOUBLE_EQ(configs.camera_config.rgb.translation[1], 0.0);
    EXPECT_DOUBLE_EQ(configs.camera_config.rgb.translation[2], 0.0);
    EXPECT_DOUBLE_EQ(configs.camera_config.ir.translation[0], -0.036);
    EXPECT_DOUBLE_EQ(configs.camera_config.ir.translation[1], 0.03);
    EXPECT_DOUBLE_EQ(configs.camera_config.ir.translation[2], -0.009);
}
