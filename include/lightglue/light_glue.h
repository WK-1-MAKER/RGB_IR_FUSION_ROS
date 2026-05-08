//
// Created by haoyuefan on 2021/9/22.
//

#ifndef SUPER_GLUE_H_
#define SUPER_GLUE_H_

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <Eigen/Core>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

#include "3rdparty/tensorrtbuffer/include/buffers.h"
#include "read_config.h"

using tensorrt_buffer::TensorRTUniquePtr;

class SuperPointLightGlue {
 public:
  explicit SuperPointLightGlue(const SuperPointLightGlueConfig &superpoint_lightglue_config);

  bool build();

  bool infer(const Eigen::Matrix<double, 258, Eigen::Dynamic> &features0, const Eigen::Matrix<double, 258, Eigen::Dynamic> &features1, Eigen::Matrix<int, Eigen::Dynamic, 2> &matches_index,
             Eigen::Matrix<double, Eigen::Dynamic, 1> &matches_score);

  int matching_points(const Eigen::Matrix<double, 258, Eigen::Dynamic> &features0, const Eigen::Matrix<double, 258, Eigen::Dynamic> &features1, std::vector<cv::DMatch> &matches,
                      bool outlier_rejection = false);

  Eigen::Matrix<double, 258, Eigen::Dynamic> normalize_keypoints(const Eigen::Matrix<double, 258, Eigen::Dynamic> &features, int width, int height);

  void save_engine();

  bool deserialize_engine();

 private:
  SuperPointLightGlueConfig superpoint_lightglue_config_;

  nvinfer1::Dims keypoints_0_dims_{};
  nvinfer1::Dims descriptors_0_dims_{};
  nvinfer1::Dims keypoints_1_dims_{};
  nvinfer1::Dims descriptors_1_dims_{};

  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IExecutionContext> context_;

  bool construct_network(TensorRTUniquePtr<nvinfer1::IBuilder> &builder, TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network, TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config,
                         TensorRTUniquePtr<nvonnxparser::IParser> &parser) const;

  bool process_input(const tensorrt_buffer::BufferManager &buffers, const Eigen::Matrix<double, 258, Eigen::Dynamic> &features0, const Eigen::Matrix<double, 258, Eigen::Dynamic> &features1);

  bool process_output(const tensorrt_buffer::BufferManager &buffers, Eigen::Matrix<int, Eigen::Dynamic, 2> &matches_index, Eigen::Matrix<double, Eigen::Dynamic, 1> &matches_score);

  void filter_matches(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &scores, Eigen::Matrix<int, Eigen::Dynamic, 2> &matches_index, Eigen::Matrix<double, Eigen::Dynamic, 1> &matches_score);
};

typedef std::shared_ptr<SuperPointLightGlue> SuperPointLightGluePtr;

#endif  // SUPER_GLUE_H_
