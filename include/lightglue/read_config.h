#ifndef READ_CONFIG_H_
#define READ_CONFIG_H_

#include <yaml-cpp/yaml.h>

#include <iostream>
#include <stdexcept>

#include "utils.h"

namespace read_config_detail {

inline YAML::Node RequireNode(const YAML::Node &parent, const std::string &key,
                              const std::string &context) {
  const YAML::Node node = parent[key];
  if (!node) {
    throw std::runtime_error("Missing required config section/key '" + key +
                             "' in " + context);
  }
  return node;
}

inline std::vector<std::string> RequireStringList(const YAML::Node &parent,
                                                  const std::string &key,
                                                  const std::string &context,
                                                  size_t min_count) {
  const YAML::Node node = RequireNode(parent, key, context);
  if (!node.IsSequence()) {
    throw std::runtime_error("Config key '" + key + "' in " + context +
                             " must be a sequence");
  }

  std::vector<std::string> values;
  values.reserve(node.size());
  for (size_t i = 0; i < node.size(); ++i) {
    values.push_back(node[i].as<std::string>());
  }

  if (values.size() < min_count) {
    throw std::runtime_error("Config key '" + key + "' in " + context +
                             " must contain at least " +
                             std::to_string(min_count) + " item(s)");
  }
  return values;
}

inline std::string RequireStringValue(const YAML::Node &parent,
                                      const std::string &key,
                                      const std::string &context) {
  const YAML::Node node = RequireNode(parent, key, context);
  const std::string value = node.as<std::string>();
  if (value.empty()) {
    throw std::runtime_error("Config key '" + key + "' in " + context +
                             " must not be empty");
  }
  return value;
}

}  // namespace read_config_detail

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
      throw std::runtime_error("Config file does not exist: " + config_file);
    }
    this->model_dir = model_dir;

    YAML::Node file_node;
    try {
      file_node = YAML::LoadFile(config_file);
    } catch (const YAML::Exception &e) {
      throw std::runtime_error("Failed to load config file '" + config_file +
                               "': " + e.what());
    }

    try {
      YAML::Node superpoint_node =
          read_config_detail::RequireNode(file_node, "superpoint", "root config");
      superpoint_config.max_keypoints =
          read_config_detail::RequireNode(superpoint_node, "max_keypoints",
                                          "superpoint").as<int>();
      superpoint_config.keypoint_threshold =
          read_config_detail::RequireNode(superpoint_node, "keypoint_threshold",
                                          "superpoint").as<double>();
      superpoint_config.remove_borders =
          read_config_detail::RequireNode(superpoint_node, "remove_borders",
                                          "superpoint").as<int>();
      superpoint_config.dla_core =
          read_config_detail::RequireNode(superpoint_node, "dla_core",
                                          "superpoint").as<int>();
      superpoint_config.input_tensor_names =
          read_config_detail::RequireStringList(superpoint_node,
                                                "input_tensor_names",
                                                "superpoint", 1);
      superpoint_config.output_tensor_names =
          read_config_detail::RequireStringList(superpoint_node,
                                                "output_tensor_names",
                                                "superpoint", 2);
      std::string superpoint_onnx_file =
          read_config_detail::RequireStringValue(superpoint_node, "onnx_file",
                                                 "superpoint");
      std::string superpoint_engine_file =
          read_config_detail::RequireStringValue(superpoint_node, "engine_file",
                                                 "superpoint");
      superpoint_config.onnx_file =
          ConcatenateFolderAndFileName(model_dir, superpoint_onnx_file);
      superpoint_config.engine_file =
          ConcatenateFolderAndFileName(model_dir, superpoint_engine_file);

      YAML::Node lightglue_node = read_config_detail::RequireNode(
          file_node, "superpoint_lightglue", "root config");
      superpoint_lightglue_config.image_width =
          read_config_detail::RequireNode(lightglue_node, "image_width",
                                          "superpoint_lightglue").as<int>();
      superpoint_lightglue_config.image_height =
          read_config_detail::RequireNode(lightglue_node, "image_height",
                                          "superpoint_lightglue").as<int>();
      superpoint_lightglue_config.filter_threshold =
          read_config_detail::RequireNode(lightglue_node, "filter_threshold",
                                          "superpoint_lightglue").as<double>();
      superpoint_lightglue_config.depth_confidence =
          read_config_detail::RequireNode(lightglue_node, "depth_confidence",
                                          "superpoint_lightglue").as<double>();
      superpoint_lightglue_config.width_confidence =
          read_config_detail::RequireNode(lightglue_node, "width_confidence",
                                          "superpoint_lightglue").as<double>();
      superpoint_lightglue_config.dla_core =
          read_config_detail::RequireNode(lightglue_node, "dla_core",
                                          "superpoint_lightglue").as<int>();
      superpoint_lightglue_config.input_tensor_names =
          read_config_detail::RequireStringList(lightglue_node,
                                                "input_tensor_names",
                                                "superpoint_lightglue", 4);
      superpoint_lightglue_config.output_tensor_names =
          read_config_detail::RequireStringList(lightglue_node,
                                                "output_tensor_names",
                                                "superpoint_lightglue", 1);
      std::string lightglue_onnx_file =
          read_config_detail::RequireStringValue(lightglue_node, "onnx_file",
                                                 "superpoint_lightglue");
      std::string lightglue_engine_file =
          read_config_detail::RequireStringValue(lightglue_node, "engine_file",
                                                 "superpoint_lightglue");
      superpoint_lightglue_config.onnx_file =
          ConcatenateFolderAndFileName(model_dir, lightglue_onnx_file);
      superpoint_lightglue_config.engine_file =
          ConcatenateFolderAndFileName(model_dir, lightglue_engine_file);
    } catch (const YAML::Exception &e) {
      throw std::runtime_error("Invalid config file '" + config_file +
                               "': " + e.what());
    }
  }
};

#endif  // READ_CONFIG_H_
