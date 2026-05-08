#ifndef READ_CONFIG_H_
#define READ_CONFIG_H_

#include <yaml-cpp/yaml.h>

#include <iostream>

#include "utils.h"

struct SuperPointConfig {
  int max_keypoints{};
  double keypoint_threshold{};
  int remove_borders{};
  int dla_core{};
  std::vector<std::string> input_tensor_names;
  std::vector<std::string> output_tensor_names;
  std::string onnx_file;
  std::string engine_file;
};

struct SuperPointLightGlueConfig {
  int image_width{};
  int image_height{};
  double filter_threshold{};
  double depth_confidence{};
  double width_confidence{};
  int dla_core{};
  std::vector<std::string> input_tensor_names;
  std::vector<std::string> output_tensor_names;
  std::string onnx_file;
  std::string engine_file;
};

struct Configs {
  std::string model_dir;

  SuperPointConfig superpoint_config;
  SuperPointLightGlueConfig superpoint_lightglue_config;

  Configs(const std::string &config_file, const std::string &model_dir) {
    std::cout << "Config file is " << config_file << std::endl;
    if (!FileExists(config_file)) {
      std::cerr << "Config file " << config_file << " doesn't exist." << std::endl;
      return;
    }
    YAML::Node file_node = YAML::LoadFile(config_file);

    YAML::Node superpoint_node = file_node["superpoint"];
    superpoint_config.max_keypoints = superpoint_node["max_keypoints"].as<int>();
    superpoint_config.keypoint_threshold = superpoint_node["keypoint_threshold"].as<double>();
    superpoint_config.remove_borders = superpoint_node["remove_borders"].as<int>();
    superpoint_config.dla_core = superpoint_node["dla_core"].as<int>();
    YAML::Node superpoint_input_tensor_names_node = superpoint_node["input_tensor_names"];
    size_t superpoint_num_input_tensor_names = superpoint_input_tensor_names_node.size();
    for (size_t i = 0; i < superpoint_num_input_tensor_names; i++) {
      superpoint_config.input_tensor_names.push_back(superpoint_input_tensor_names_node[i].as<std::string>());
    }
    YAML::Node superpoint_output_tensor_names_node = superpoint_node["output_tensor_names"];
    size_t superpoint_num_output_tensor_names = superpoint_output_tensor_names_node.size();
    for (size_t i = 0; i < superpoint_num_output_tensor_names; i++) {
      superpoint_config.output_tensor_names.push_back(superpoint_output_tensor_names_node[i].as<std::string>());
    }
    std::string superpoint_onnx_file = superpoint_node["onnx_file"].as<std::string>();
    std::string superpoint_engine_file = superpoint_node["engine_file"].as<std::string>();
    superpoint_config.onnx_file = ConcatenateFolderAndFileName(model_dir, superpoint_onnx_file);
    superpoint_config.engine_file = ConcatenateFolderAndFileName(model_dir, superpoint_engine_file);

    YAML::Node lightglue_node = file_node["superpoint_lightglue"];
    superpoint_lightglue_config.image_width = lightglue_node["image_width"].as<int>();
    superpoint_lightglue_config.image_height = lightglue_node["image_height"].as<int>();
    superpoint_lightglue_config.filter_threshold = lightglue_node["filter_threshold"].as<double>();
    superpoint_lightglue_config.depth_confidence = lightglue_node["depth_confidence"].as<double>();
    superpoint_lightglue_config.width_confidence = lightglue_node["width_confidence"].as<double>();
    superpoint_lightglue_config.dla_core = lightglue_node["dla_core"].as<int>();
    YAML::Node lightglue_input_tensor_names_node = lightglue_node["input_tensor_names"];
    size_t lightglue_num_input_tensor_names = lightglue_input_tensor_names_node.size();
    for (size_t i = 0; i < lightglue_num_input_tensor_names; i++) {
      superpoint_lightglue_config.input_tensor_names.push_back(lightglue_input_tensor_names_node[i].as<std::string>());
    }
    YAML::Node lightglue_output_tensor_names_node = lightglue_node["output_tensor_names"];
    size_t lightglue_num_output_tensor_names = lightglue_output_tensor_names_node.size();
    for (size_t i = 0; i < lightglue_num_output_tensor_names; i++) {
      superpoint_lightglue_config.output_tensor_names.push_back(lightglue_output_tensor_names_node[i].as<std::string>());
    }
    std::string lightglue_onnx_file = lightglue_node["onnx_file"].as<std::string>();
    std::string lightglue_engine_file = lightglue_node["engine_file"].as<std::string>();
    superpoint_lightglue_config.onnx_file = ConcatenateFolderAndFileName(model_dir, lightglue_onnx_file);
    superpoint_lightglue_config.engine_file = ConcatenateFolderAndFileName(model_dir, lightglue_engine_file);
  }
};

#endif  // READ_CONFIG_H_
