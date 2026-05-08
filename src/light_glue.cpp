//
// Created by haoyuefan on 2021/9/22.
//

#include "lightglue/light_glue.h"

#include <cfloat>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <utility>

using namespace tensorrt_log;
using namespace tensorrt_buffer;

SuperPointLightGlue::SuperPointLightGlue(const SuperPointLightGlueConfig &superpoint_lightglue_config) : superpoint_lightglue_config_(superpoint_lightglue_config), engine_(nullptr) {
  setReportableSeverity(Logger::Severity::kINTERNAL_ERROR);
}

bool SuperPointLightGlue::build() {
  if (deserialize_engine()) {
    return true;
  }

  auto builder = TensorRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
  if (!builder) {
    return false;
  }

  const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = TensorRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
  if (!network) {
    return false;
  }

  auto config = TensorRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!config) {
    return false;
  }

  auto parser = TensorRTUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
  if (!parser) {
    return false;
  }

  auto profile = builder->createOptimizationProfile();
  if (!profile) {
    return false;
  }
  profile->setDimensions(superpoint_lightglue_config_.input_tensor_names[0].c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(1, 1, 2));
  profile->setDimensions(superpoint_lightglue_config_.input_tensor_names[0].c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(1, 512, 2));
  profile->setDimensions(superpoint_lightglue_config_.input_tensor_names[0].c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(1, 2000, 2));

  profile->setDimensions(superpoint_lightglue_config_.input_tensor_names[1].c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(1, 1, 2));
  profile->setDimensions(superpoint_lightglue_config_.input_tensor_names[1].c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(1, 512, 2));
  profile->setDimensions(superpoint_lightglue_config_.input_tensor_names[1].c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(1, 2000, 2));

  profile->setDimensions(superpoint_lightglue_config_.input_tensor_names[2].c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(1, 1, 256));
  profile->setDimensions(superpoint_lightglue_config_.input_tensor_names[2].c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(1, 512, 256));
  profile->setDimensions(superpoint_lightglue_config_.input_tensor_names[2].c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(1, 2000, 256));

  profile->setDimensions(superpoint_lightglue_config_.input_tensor_names[3].c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3(1, 1, 256));
  profile->setDimensions(superpoint_lightglue_config_.input_tensor_names[3].c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3(1, 512, 256));
  profile->setDimensions(superpoint_lightglue_config_.input_tensor_names[3].c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3(1, 2000, 256));

  config->addOptimizationProfile(profile);

  auto constructed = construct_network(builder, network, config, parser);
  if (!constructed) {
    return false;
  }

  auto profile_stream = makeCudaStream();
  if (!profile_stream) {
    return false;
  }
  config->setProfileStream(*profile_stream);

  TensorRTUniquePtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
  if (!plan) {
    return false;
  }

  TensorRTUniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(gLogger.getTRTLogger())};
  if (!runtime) {
    return false;
  }

  engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
  if (!engine_) {
    return false;
  }

  save_engine();

  ASSERT(network->getNbInputs() == 4);
  keypoints_0_dims_ = network->getInput(0)->getDimensions();
  keypoints_1_dims_ = network->getInput(1)->getDimensions();
  descriptors_0_dims_ = network->getInput(2)->getDimensions();
  descriptors_1_dims_ = network->getInput(3)->getDimensions();
  assert(keypoints_0_dims_.d[1] == -1);
  assert(keypoints_1_dims_.d[1] == -1);
  assert(descriptors_0_dims_.d[1] == -1);
  assert(descriptors_1_dims_.d[1] == -1);
  return true;
}

bool SuperPointLightGlue::construct_network(TensorRTUniquePtr<nvinfer1::IBuilder> &builder, TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network,
                                            TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config, TensorRTUniquePtr<nvonnxparser::IParser> &parser) const {
  auto parsed = parser->parseFromFile(superpoint_lightglue_config_.onnx_file.c_str(), static_cast<int>(gLogger.getReportableSeverity()));
  if (!parsed) {
    return false;
  }
  config->setFlag(nvinfer1::BuilderFlag::kFP16);
  enableDLA(builder.get(), config.get(), superpoint_lightglue_config_.dla_core);
  return true;
}

bool SuperPointLightGlue::infer(const Eigen::Matrix<double, 258, Eigen::Dynamic> &features0, const Eigen::Matrix<double, 258, Eigen::Dynamic> &features1,
                                Eigen::Matrix<int, Eigen::Dynamic, 2> &matches_index, Eigen::Matrix<double, Eigen::Dynamic, 1> &matches_score) {
  if (!context_) {
    context_ = TensorRTUniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (!context_) {
      return false;
    }
  }

  assert(engine_->getNbBindings() == 5);

  const int keypoints_0_index = engine_->getBindingIndex(superpoint_lightglue_config_.input_tensor_names[0].c_str());
  const int keypoints_1_index = engine_->getBindingIndex(superpoint_lightglue_config_.input_tensor_names[1].c_str());
  const int descriptors_0_index = engine_->getBindingIndex(superpoint_lightglue_config_.input_tensor_names[2].c_str());
  const int descriptors_1_index = engine_->getBindingIndex(superpoint_lightglue_config_.input_tensor_names[3].c_str());

  context_->setBindingDimensions(keypoints_0_index, nvinfer1::Dims3(1, features0.cols(), 2));
  context_->setBindingDimensions(keypoints_1_index, nvinfer1::Dims3(1, features1.cols(), 2));
  context_->setBindingDimensions(descriptors_0_index, nvinfer1::Dims3(1, features0.cols(), 256));
  context_->setBindingDimensions(descriptors_1_index, nvinfer1::Dims3(1, features1.cols(), 256));

  keypoints_0_dims_ = context_->getBindingDimensions(keypoints_0_index);
  keypoints_1_dims_ = context_->getBindingDimensions(keypoints_1_index);
  descriptors_0_dims_ = context_->getBindingDimensions(descriptors_0_index);
  descriptors_1_dims_ = context_->getBindingDimensions(descriptors_1_index);

  BufferManager buffers(engine_, 0, context_.get());

  ASSERT(superpoint_lightglue_config_.input_tensor_names.size() == 4);
  if (!process_input(buffers, features0, features1)) {
    return false;
  }

  buffers.copyInputToDevice();

  bool status = context_->executeV2(buffers.getDeviceBindings().data());
  if (!status) {
    return false;
  }
  buffers.copyOutputToHost();

  if (!process_output(buffers, matches_index, matches_score)) {
    return false;
  }

  return true;
}

bool SuperPointLightGlue::process_input(const BufferManager &buffers, const Eigen::Matrix<double, 258, Eigen::Dynamic> &features0, const Eigen::Matrix<double, 258, Eigen::Dynamic> &features1) {
  auto *keypoints_0_buffer = static_cast<float *>(buffers.getHostBuffer(superpoint_lightglue_config_.input_tensor_names[0]));
  auto *keypoints_1_buffer = static_cast<float *>(buffers.getHostBuffer(superpoint_lightglue_config_.input_tensor_names[1]));
  auto *descriptors_0_buffer = static_cast<float *>(buffers.getHostBuffer(superpoint_lightglue_config_.input_tensor_names[2]));
  auto *descriptors_1_buffer = static_cast<float *>(buffers.getHostBuffer(superpoint_lightglue_config_.input_tensor_names[3]));

  for (int colk0 = 0; colk0 < features0.cols(); ++colk0) {
    for (int rowk0 = 0; rowk0 < 2; ++rowk0) {
      keypoints_0_buffer[colk0 * 2 + rowk0] = features0(rowk0, colk0);
    }
  }

  for (int colk1 = 0; colk1 < features1.cols(); ++colk1) {
    for (int rowk1 = 0; rowk1 < 2; ++rowk1) {
      keypoints_1_buffer[colk1 * 2 + rowk1] = features1(rowk1, colk1);
    }
  }

  for (int cold0 = 0; cold0 < features0.cols(); ++cold0) {
    for (int rowd0 = 2; rowd0 < features0.rows(); ++rowd0) {
      descriptors_0_buffer[cold0 * 256 + (rowd0 - 2)] = features0(rowd0, cold0);
    }
  }

  for (int cold1 = 0; cold1 < features1.cols(); ++cold1) {
    for (int rowd1 = 2; rowd1 < features1.rows(); ++rowd1) {
      descriptors_1_buffer[cold1 * 256 + (rowd1 - 2)] = features1(rowd1, cold1);
    }
  }

  return true;
}

void SuperPointLightGlue::filter_matches(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &scores, Eigen::Matrix<int, Eigen::Dynamic, 2> &matches_index,
                                         Eigen::Matrix<double, Eigen::Dynamic, 1> &matches_score) {
  // obtain matches from a log assignment matrix [M+1 x N+1]
  std::vector<std::pair<int, double>> row_max;
  row_max.resize(scores.rows() - 1);
  for (int row = 0; row < scores.rows() - 1; ++row) {
    double max_value = -FLT_MAX;
    for (int col = 0; col < scores.cols() - 1; ++col) {
      if (scores(row, col) > max_value) {
        row_max[row] = std::make_pair(col, scores(row, col));
        max_value = scores(row, col);
      }
    }
  }

  std::vector<std::pair<int, double>> col_max;
  col_max.resize(scores.cols() - 1);

  for (int col = 0; col < scores.cols() - 1; ++col) {
    double max_value = -FLT_MAX;
    for (int row = 0; row < scores.rows() - 1; ++row) {
      if (scores(row, col) > max_value) {
        col_max[col] = std::make_pair(row, scores(row, col));
        max_value = scores(row, col);
      }
    }
  }
  std::vector<int> matches_index0_vec;
  std::vector<int> matches_index1_vec;
  std::vector<double> matches_score_vec;
  for (int row = 0; row < row_max.size(); ++row) {
    if (row == col_max[row_max[row].first].first) {
      double score_exp = std::exp(row_max[row].second);
      if (score_exp > superpoint_lightglue_config_.filter_threshold) {
        matches_index0_vec.push_back(row);
        matches_index1_vec.push_back(row_max[row].first);
        matches_score_vec.push_back(score_exp);
      }
    }
  }
  matches_index.resize(matches_index0_vec.size(), 2);
  matches_score.resize(matches_score_vec.size(), 1);
  for (int i = 0; i < matches_index0_vec.size(); ++i) {
    matches_index(i, 0) = matches_index0_vec[i];
    matches_index(i, 1) = matches_index1_vec[i];
    matches_score(i) = matches_score_vec[i];
  }
}

bool SuperPointLightGlue::process_output(const BufferManager &buffers, Eigen::Matrix<int, Eigen::Dynamic, 2> &matches_index, Eigen::Matrix<double, Eigen::Dynamic, 1> &matches_score) {
  auto *output_scores = static_cast<float *>(buffers.getHostBuffer(superpoint_lightglue_config_.output_tensor_names[0]));
  int scores_rows = keypoints_0_dims_.d[1] + 1;
  int scores_cols = keypoints_1_dims_.d[1] + 1;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> scores_matrix;
  scores_matrix.resize(scores_rows, scores_cols);
  for (int row = 0; row < scores_rows; ++row) {
    for (int col = 0; col < scores_cols; ++col) {
      scores_matrix(row, col) = output_scores[row * scores_cols + col];
    }
  }
  filter_matches(scores_matrix, matches_index, matches_score);
  return true;
}

void SuperPointLightGlue::save_engine() {
  if (superpoint_lightglue_config_.engine_file.empty()) return;
  if (engine_ != nullptr) {
    nvinfer1::IHostMemory *data = engine_->serialize();
    std::ofstream file(superpoint_lightglue_config_.engine_file, std::ios::binary);
    if (!file) return;
    file.write(reinterpret_cast<const char *>(data->data()), data->size());
  }
}

bool SuperPointLightGlue::deserialize_engine() {
  std::ifstream file(superpoint_lightglue_config_.engine_file, std::ios::binary);
  if (file.is_open()) {
    file.seekg(0, std::ifstream::end);
    size_t size = file.tellg();
    file.seekg(0, std::ifstream::beg);
    char *model_stream = new char[size];
    file.read(model_stream, size);
    file.close();
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(gLogger);
    if (runtime == nullptr) {
      delete[] model_stream;
      return false;
    }
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(model_stream, size));
    if (engine_ == nullptr) {
      delete[] model_stream;
      return false;
    }
    delete[] model_stream;
    return true;
  }
  return false;
}

int SuperPointLightGlue::matching_points(const Eigen::Matrix<double, 258, Eigen::Dynamic> &features0, const Eigen::Matrix<double, 258, Eigen::Dynamic> &features1, std::vector<cv::DMatch> &matches,
                                         bool outlier_rejection) {
  matches.clear();
  auto norm_features0 = normalize_keypoints(features0, superpoint_lightglue_config_.image_width, superpoint_lightglue_config_.image_height);
  auto norm_features1 = normalize_keypoints(features1, superpoint_lightglue_config_.image_width, superpoint_lightglue_config_.image_height);

  Eigen::VectorXi indices0, indices1;
  Eigen::Matrix<int, Eigen::Dynamic, 2> matches_index;
  Eigen::Matrix<double, Eigen::Dynamic, 1> matches_score;
  infer(norm_features0, norm_features1, matches_index, matches_score);

  std::vector<cv::Point> points0, points1;
  std::vector<int> point_indexes;
  for (size_t i = 0; i < matches_index.rows(); i++) {
    points0.emplace_back(features0(0, matches_index(i, 0)), features0(1, matches_index(i, 0)));
    points1.emplace_back(features1(0, matches_index(i, 1)), features1(1, matches_index(i, 1)));
    matches.emplace_back(matches_index(i, 0), matches_index(i, 1), matches_score(i));
  }

  if (outlier_rejection) {
    std::vector<uchar> inliers;
    cv::findFundamentalMat(points0, points1, cv::FM_RANSAC, 3, 0.99, inliers);
    int j = 0;
    for (int i = 0; i < matches.size(); i++) {
      if (inliers[i]) {
        matches[j++] = matches[i];
      }
    }
    matches.resize(j);
  }

  return matches.size();
}

Eigen::Matrix<double, 258, Eigen::Dynamic> SuperPointLightGlue::normalize_keypoints(const Eigen::Matrix<double, 258, Eigen::Dynamic> &features, int width, int height) {
  Eigen::Matrix<double, 258, Eigen::Dynamic> norm_features;
  norm_features.resize(258, features.cols());
  norm_features = features;
  for (int col = 0; col < features.cols(); ++col) {
    norm_features(0, col) = (features(0, col) - width / 2) / (std::max(width, height) * 0.5);
    norm_features(1, col) = (features(1, col) - height / 2) / (std::max(width, height) * 0.5);
  }
  return norm_features;
}
