# YOLOv8 ONNX/TensorRT 双后端部署实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变 ROS 节点、图像预处理、YOLO 后处理和三维定位架构的前提下，让 `YoloDetector` 根据 `model_path` 同时支持 ONNX Runtime 和预构建 TensorRT engine。

**Architecture:** 保持 `YoloDetector` 的公开接口不变，在其内部增加后端枚举和 TensorRT 资源。`.onnx` 继续沿用现有 ONNX Runtime 路径；`.engine` 只反序列化现成 engine，复用预处理产生的 CPU float blob，通过持久化的 TensorRT buffer/stream 执行，再把统一的 float 输出交给现有 `decodeYolov8Detections()`。`YoloNode`、相机几何和特征定位模块不改。

**Tech Stack:** ROS1/catkin、C++14、ONNX Runtime GPU 1.18.0、TensorRT 8.6.1.6、CUDA 11.8、OpenCV 4、yaml-cpp、GoogleTest。

---

## 文件结构

- 修改 `include/ros_yolo/detector.h`
  - 增加后端枚举、扩展名选择函数声明和 TensorRT 资源成员。
  - 保持 `YoloDetector(const std::string&, bool, const cv::Size&)` 与 `detect()` 签名不变。
- 修改 `src/detector.cpp`
  - 保留当前 ONNX Runtime 初始化、预处理和 ONNX 执行逻辑。
  - 增加 engine 反序列化、binding 校验、持久化 buffer/stream 和 TensorRT 执行。
  - 将 TensorRT 输出汇入现有通用 YOLOv8 解码函数。
- 修改 `tests/test_detector_postprocess.cpp`
  - 增加纯函数后端选择测试，保持已有 ONNX 后处理测试。
- 新建 `tests/test_detector_tensorrt.cpp`
  - 在有 GPU 且 engine 文件存在时验证 TensorRT 构造和一次双输入推理；资源不可用时明确跳过。
- 修改 `CMakeLists.txt`
  - 注册 TensorRT detector 测试并链接已经存在的 ONNX Runtime、CUDA、TensorRT 库。
- 修改 `launch/yolo_topic.launch`
  - 保留现有 `model_path` 参数，增加 launch arg 以便命令行在 ONNX/engine 间切换。
- 修改 `README.md`
  - 记录两种模型路径、engine 不自动构建、硬件兼容性和启动示例。

仓库规则禁止开发过程中自动执行 `git add`、`git commit` 或 `git push`；实现阶段只保留工作区修改和验证结果，由人工管理提交。

---

### 任务 1：增加可测试的后端选择逻辑

**文件：**

- 修改：`include/ros_yolo/detector.h:1-48`
- 修改：`src/detector.cpp:1-35`
- 测试：`tests/test_detector_postprocess.cpp`

- [ ] **步骤 1：先写失败测试**

在 `tests/test_detector_postprocess.cpp` 的匿名命名空间测试前加入：

```cpp
TEST(YoloDetectorBackendTest, SelectsOnnxFromExtension)
{
    EXPECT_EQ(inferBackendFromModelPath("/tmp/aifi-best.onnx"), InferenceBackend::ONNX);
    EXPECT_EQ(inferBackendFromModelPath("/tmp/AIFI-BEST.ONNX"), InferenceBackend::ONNX);
}

TEST(YoloDetectorBackendTest, SelectsTensorRtFromExtension)
{
    EXPECT_EQ(inferBackendFromModelPath("/tmp/aifi-best.engine"), InferenceBackend::TENSORRT);
    EXPECT_EQ(inferBackendFromModelPath("/tmp/AIFI-BEST.ENGINE"), InferenceBackend::TENSORRT);
}

TEST(YoloDetectorBackendTest, RejectsUnsupportedExtension)
{
    EXPECT_THROW(inferBackendFromModelPath("/tmp/aifi-best.pt"), std::invalid_argument);
    EXPECT_THROW(inferBackendFromModelPath("/tmp/aifi-best"), std::invalid_argument);
}
```

在测试文件中补充 `#include <stdexcept>`，并确保测试仍使用 `namespace ros_yolo`。

- [ ] **步骤 2：运行测试确认当前失败**

从工作区根目录运行：

```bash
cd /home/SENSETIME/wenkai/RGB_IR_FUSION_ROS_YOLOv8
catkin_make --pkg two_stream -DCMAKE_BUILD_TYPE=Release
./devel/lib/two_stream/test_detector_postprocess
```

预期：编译失败，提示 `InferenceBackend` 或 `inferBackendFromModelPath` 尚未声明。

- [ ] **步骤 3：加入最小声明和实现**

在 `detector.h` 的 `namespace ros_yolo` 开头加入：

```cpp
enum class InferenceBackend
{
    ONNX,
    TENSORRT
};

InferenceBackend inferBackendFromModelPath(const std::string& modelPath);
```

在 `detector.cpp` 的匿名命名空间中加入小写后缀辅助函数，并在 `namespace ros_yolo` 中实现：

```cpp
namespace
{
std::string lowerCopy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}
}

InferenceBackend inferBackendFromModelPath(const std::string& modelPath)
{
    const std::string lowerPath = lowerCopy(modelPath);
    if (lowerPath.size() >= 5 && lowerPath.compare(lowerPath.size() - 5, 5, ".onnx") == 0)
    {
        return InferenceBackend::ONNX;
    }
    if (lowerPath.size() >= 7 && lowerPath.compare(lowerPath.size() - 7, 7, ".engine") == 0)
    {
        return InferenceBackend::TENSORRT;
    }
    throw std::invalid_argument("Model path must end with .onnx or .engine: " + modelPath);
}
```

为 `std::tolower` 增加 `<cctype>`，并保留已有 `<algorithm>`、`<stdexcept>`。

- [ ] **步骤 4：重新运行选择测试**

运行：

```bash
cd /home/SENSETIME/wenkai/RGB_IR_FUSION_ROS_YOLOv8
catkin_make --pkg two_stream -DCMAKE_BUILD_TYPE=Release
./devel/lib/two_stream/test_detector_postprocess
```

预期：后端选择测试和已有 ONNX 后处理测试全部通过。

---

### 任务 2：扩展检测器的 TensorRT 资源和 engine 初始化

**文件：**

- 修改：`include/ros_yolo/detector.h:19-48`
- 修改：`src/detector.cpp:34-115`

- [ ] **步骤 1：先增加 engine 元数据测试**

新建 `tests/test_detector_tensorrt.cpp`：

```cpp
#include "ros_yolo/detector.h"

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>
#include <fstream>

#ifndef TWO_STREAM_ENGINE_PATH
#define TWO_STREAM_ENGINE_PATH ""
#endif

TEST(YoloDetectorTensorRtTest, LoadsProvidedEngineWhenRuntimeIsAvailable)
{
    const std::string enginePath = TWO_STREAM_ENGINE_PATH;
    std::ifstream engineFile(enginePath, std::ios::binary);
    if (enginePath.empty() || !engineFile.good())
    {
        GTEST_SKIP() << "TensorRT engine is not present: " << enginePath;
    }

    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
    {
        GTEST_SKIP() << "No CUDA device is available";
    }

    ros_yolo::YoloDetector detector(enginePath, true, cv::Size(640, 640));
    cv::Mat rgb(640, 640, CV_8UC3, cv::Scalar::all(0));
    cv::Mat ir(640, 640, CV_8UC3, cv::Scalar::all(0));

    EXPECT_NO_THROW({
        const auto detections = detector.detect(rgb, ir, 0.30F, 0.40F);
        (void)detections;
    });
}
```

- [ ] **步骤 2：把测试注册到 CMake**

在 `CMakeLists.txt` 的 `if(CATKIN_ENABLE_TESTING)` 中加入：

```cmake
  catkin_add_gtest(test_detector_tensorrt
    tests/test_detector_tensorrt.cpp
    src/detector.cpp
    src/yolov8_decode.cpp
    src/utils.cpp
  )
  if(TARGET test_detector_tensorrt)
    target_compile_definitions(test_detector_tensorrt
      PRIVATE TWO_STREAM_ENGINE_PATH="${PROJECT_SOURCE_DIR}/models/aifi-best-fp16.engine"
    )
    target_link_libraries(test_detector_tensorrt
      ${catkin_LIBRARIES}
      ${OpenCV_LIBS}
      ${ONNXRUNTIME_LIB}
      ${CUDA_LIBRARIES}
      ${TENSORRT_NVINFER_LIB}
      ${TENSORRT_NVONNXPARSER_LIB}
      -Wl,--no-as-needed
      ${TENSORRT_NVINFER_PLUGIN_LIB}
      -Wl,--as-needed
      ${TENSORRT_NVPARSERS_LIB}
      gtest_main
    )
  endif()
```

测试会在 engine 未放入包内 `models/` 时跳过，不影响没有 engine 的普通 ONNX 构建。

由于 `test_detector_postprocess` 也会编译修改后的 `src/detector.cpp`，还要在其
现有 `target_link_libraries(test_detector_postprocess ...)` 中追加同一组
`${CUDA_LIBRARIES}`、`${TENSORRT_NVINFER_LIB}`、`${TENSORRT_NVONNXPARSER_LIB}`、
`${TENSORRT_NVINFER_PLUGIN_LIB}` 和 `${TENSORRT_NVPARSERS_LIB}`；否则后端选择测试
虽然能编译源码，但链接阶段会缺少 TensorRT/CUDA 符号。

- [ ] **步骤 3：运行测试确认构造尚未实现**

运行：

```bash
cd /home/SENSETIME/wenkai/RGB_IR_FUSION_ROS_YOLOv8
catkin_make --pkg two_stream -DCMAKE_BUILD_TYPE=Release
./devel/lib/two_stream/test_detector_tensorrt
```

预期：如果 engine 尚未复制到 `models/`，测试显示 skipped；如果 engine 已存在，
则因 TensorRT 资源尚未接入而编译或运行失败。

- [ ] **步骤 4：在 detector 头文件增加 TensorRT 成员**

在 `detector.h` 增加 TensorRT/CUDA 头文件和成员：

```cpp
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <memory>

#include "3rdparty/tensorrtbuffer/include/buffers.h"

// private:
InferenceBackend backend_{InferenceBackend::ONNX};
std::shared_ptr<nvinfer1::ICudaEngine> trtEngine_;
std::shared_ptr<nvinfer1::IExecutionContext> trtContext_;
std::unique_ptr<tensorrt_buffer::BufferManager> trtBuffers_;
cudaStream_t trtStream_{nullptr};
std::vector<int64_t> trtOutputShape_;
int trtRgbBinding_{-1};
int trtIrBinding_{-1};
int trtOutputBinding_{-1};

void initializeTensorRt(const std::string& modelPath);
std::vector<Detection> detectTensorRt(const std::vector<float>& rgbInput,
                                      const std::vector<float>& irInput,
                                      const cv::Size& resizedShape,
                                      const cv::Size& originalShape,
                                      float confThreshold,
                                      float iouThreshold);
```

同时声明 `~YoloDetector()`，确保 `trtStream_` 在对象销毁时释放。保留现有
ONNX `Ort::Env`、`Ort::Session` 和字符串 name 成员。

- [ ] **步骤 5：实现 engine 读取、反序列化和 binding 校验**

在 `detector.cpp` 中实现 `initializeTensorRt()`，核心顺序固定为：

```cpp
std::ifstream file(modelPath, std::ios::binary | std::ios::ate);
if (!file)
    throw std::runtime_error("Cannot open TensorRT engine: " + modelPath);
const std::streamsize size = file.tellg();
file.seekg(0, std::ios::beg);
std::vector<char> bytes(static_cast<size_t>(size));
if (!file.read(bytes.data(), size))
    throw std::runtime_error("Cannot read TensorRT engine: " + modelPath);

TensorRTUniquePtr<nvinfer1::IRuntime> runtime{
    nvinfer1::createInferRuntime(tensorrt_log::gLogger)};
if (!runtime)
    throw std::runtime_error("TensorRT runtime creation failed");
trtEngine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
    runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
if (!trtEngine_)
    throw std::runtime_error("TensorRT engine deserialization failed: " + modelPath);
trtContext_ = std::shared_ptr<nvinfer1::IExecutionContext>(
    trtEngine_->createExecutionContext());
if (!trtContext_)
    throw std::runtime_error("TensorRT execution context creation failed");
```

按 `images_rgb`、`images_ir`、`output` 查找 binding，要求三个 binding 都存在，
输入/输出数据类型为 `kFLOAT`，输入 rank 为 4，输出 rank 为 3。将实际输出维度
保存到 `trtOutputShape_`，并用 `cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)`
创建 stream。对于当前静态 engine，构造 `BufferManager(trtEngine_, 0,
trtContext_.get())`，让其一次性分配所有 host/device buffer。

如果任意校验失败，抛出包含 binding 名称、维度或数据类型的 `std::runtime_error`。

---

### 任务 3：接入 TensorRT 推理并保持统一后处理

**文件：**

- 修改：`src/detector.cpp:117-226`
- 修改：`include/ros_yolo/detector.h:30-60`

- [ ] **步骤 1：先让 TensorRT 测试覆盖真实双输入调用**

在任务 2 的测试中保留 640×640 的黑色 RGB/IR 图像和：

```cpp
const auto detections = detector.detect(rgb, ir, 0.30F, 0.40F);
```

测试只断言调用不抛异常，不断言随机/全零图像的检测数量，避免把模型数值行为
耦合到资源测试。

- [ ] **步骤 2：在构造函数中按后端分支初始化**

在 `YoloDetector::YoloDetector(...)` 开头设置：

```cpp
backend_ = inferBackendFromModelPath(modelPath);
if (backend_ == InferenceBackend::TENSORRT)
{
    initializeTensorRt(modelPath);
    inputImageShape_ = cv::Size2f(inputSize);
    return;
}
```

随后保留现有 ONNX Runtime 初始化代码不变。不要在 TensorRT 分支创建
`Ort::Session`，也不要调用任何 ONNX parser/build API。

- [ ] **步骤 3：实现 TensorRT 单帧执行**

`detectTensorRt()` 复用 `BufferManager` 分配的 device buffer，但直接调用并检查
CUDA 拷贝 API，避免项目通用 `CHECK` 宏在错误时直接终止进程。按以下顺序执行：

```cpp
const size_t rgbBytes = rgbInput.size() * sizeof(float);
const size_t irBytes = irInput.size() * sizeof(float);
if (rgbBytes != trtBuffers_->size("images_rgb") || irBytes != trtBuffers_->size("images_ir"))
    throw std::runtime_error("TensorRT input size does not match engine bindings");
if (cudaMemcpyAsync(trtBuffers_->getDeviceBuffer("images_rgb"), rgbInput.data(),
                    rgbBytes, cudaMemcpyHostToDevice, trtStream_) != cudaSuccess ||
    cudaMemcpyAsync(trtBuffers_->getDeviceBuffer("images_ir"), irInput.data(),
                    irBytes, cudaMemcpyHostToDevice, trtStream_) != cudaSuccess)
    throw std::runtime_error("TensorRT input cudaMemcpyAsync failed");

if (!trtContext_->enqueueV2(trtBuffers_->getDeviceBindings().data(), trtStream_, nullptr))
    throw std::runtime_error("TensorRT enqueueV2 failed");

const size_t outputBytes = trtBuffers_->size("output");
std::vector<float> output(outputBytes / sizeof(float));
if (cudaMemcpyAsync(output.data(), trtBuffers_->getDeviceBuffer("output"),
                    outputBytes, cudaMemcpyDeviceToHost, trtStream_) != cudaSuccess)
    throw std::runtime_error("TensorRT output cudaMemcpyAsync failed");
if (cudaStreamSynchronize(trtStream_) != cudaSuccess)
    throw std::runtime_error("TensorRT CUDA stream synchronization failed");

return decodeYolov8Detections(output, trtOutputShape_, resizedShape, originalShape,
                              confThreshold, iouThreshold);
```

如果实际 engine 使用动态输入，先通过 context 设置与现有 `rgbInputShape`、
`irInputShape` 相同的维度，再创建/重建对应的 buffer；当前提供的静态 640×640
engine 不需要这条分支。

- [ ] **步骤 4：在 `detect()` 中只替换执行段**

保留当前从 `rgbImage`/`irImage` 到 `rgbInputTensorValues`、`irInputTensorValues`
的预处理代码。两个 vector 准备完成后加入：

```cpp
if (backend_ == InferenceBackend::TENSORRT)
{
    const cv::Size resizedShape(static_cast<int>(rgbInputShape[3]),
                                static_cast<int>(rgbInputShape[2]));
    return detectTensorRt(rgbInputTensorValues, irInputTensorValues,
                          resizedShape, rgbImage.size(),
                          confThreshold, iouThreshold);
}
```

ONNX 分支继续执行现有 `Ort::Value` 创建和 `session_.Run()` 代码。不得修改
`preprocessing()`、`postprocessing()` 中的图像和坐标逻辑。

- [ ] **步骤 5：实现析构函数并验证资源释放**

在 `detector.cpp` 中加入：

```cpp
YoloDetector::~YoloDetector()
{
    if (trtStream_ != nullptr)
    {
        cudaStreamDestroy(trtStream_);
        trtStream_ = nullptr;
    }
}
```

确保 `BufferManager`、execution context 和 engine 成员在 stream 销毁后按 RAII
释放；ONNX-only 构造的 detector 不创建 stream。

- [ ] **步骤 6：运行检测器测试**

```bash
cd /home/SENSETIME/wenkai/RGB_IR_FUSION_ROS_YOLOv8
catkin_make --pkg two_stream -DCMAKE_BUILD_TYPE=Release
./devel/lib/two_stream/test_detector_postprocess
./devel/lib/two_stream/test_detector_tensorrt
```

预期：ONNX 后处理和后端选择测试通过；engine 文件/GPU 不可用时 TensorRT 集成
测试显示 skipped，资源可用时构造并完成一次推理。

---

### 任务 4：配置 launch 和部署文档

**文件：**

- 修改：`launch/yolo_topic.launch:2-20`
- 修改：`README.md:1-30`

- [ ] **步骤 1：增加可覆盖的 launch arg**

在 `<launch>` 内加入：

```xml
<arg name="model_path" default="$(find two_stream)/models/aifi-best.onnx"/>
```

把原来的固定参数改为：

```xml
<param name="model_path" value="$(arg model_path)"/>
```

这样默认继续使用 ONNX，同时可以直接执行：

```bash
roslaunch two_stream yolo_topic.launch \
  model_path:=/home/SENSETIME/wenkai/YOLOV8_IR_RGB/runs/train/transformer-aifi-s/weights/aifi-best-fp16.engine
```

- [ ] **步骤 2：补充 README 的双后端说明**

在 README 的部署状态部分明确写出：

```text
model_path 以 .onnx 结尾时使用 ONNX Runtime；以 .engine 结尾时使用 TensorRT。
TensorRT 路径只加载已有 engine，不会在节点启动时从 ONNX 构建。
engine 与 GPU、TensorRT、CUDA 环境相关；换机器时可直接选择 ONNX。
```

同时列出 ONNX 默认启动和 TensorRT 覆盖启动命令，并注明当前 engine 的固定
输入为双路 `1x3x640x640`。

- [ ] **步骤 3：检查 launch 和文档差异**

```bash
git -C /home/SENSETIME/wenkai/RGB_IR_FUSION_ROS_YOLOv8/src/two_stream diff -- launch/yolo_topic.launch README.md
```

确认没有覆盖用户当前对 `model_path` 的未提交修改，只是在其基础上增加可覆盖入口。

---

### 任务 5：构建、双模型集成验证和性能记录

**文件：**

- 验证：`tests/test_yolov8_decode.cpp`
- 验证：`tests/test_detector_postprocess.cpp`
- 验证：`tests/test_detector_tensorrt.cpp`
- 验证：`launch/yolo_topic.launch`

- [ ] **步骤 1：运行所有现有 two_stream 测试**

```bash
cd /home/SENSETIME/wenkai/RGB_IR_FUSION_ROS_YOLOv8
catkin_make --pkg two_stream run_tests -DCMAKE_BUILD_TYPE=Release
```

预期：现有解码、后处理、配置和几何测试通过；TensorRT 测试在缺少 engine/GPU
时仅跳过，不使 ONNX 构建失败。

- [ ] **步骤 2：验证 engine 元数据和独立推理**

```bash
env LD_LIBRARY_PATH=/home/SENSETIME/wenkai/3rd/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib:/usr/local/cuda-11.8/lib64 \
  /home/SENSETIME/wenkai/3rd/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/bin/trtexec \
  --loadEngine=/home/SENSETIME/wenkai/YOLOV8_IR_RGB/runs/train/transformer-aifi-s/weights/aifi-best-fp16.engine \
  --shapes=images_rgb:1x3x640x640,images_ir:1x3x640x640 \
  --warmUp=500 --duration=3 --iterations=100 --noDataTransfers
```

确认 binding 名称、输入输出维度以及 TensorRT 纯 GPU latency 均符合 spec；记录
结果但不要把它当作完整 ROS FPS。

- [ ] **步骤 3：验证 ONNX launch 路径**

```bash
source devel/setup.bash
roslaunch two_stream yolo_topic.launch \
  model_path:=$(rospack find two_stream)/models/aifi-best.onnx
```

在有 ROS master 和图像话题的部署环境中确认节点按原有 ONNX Runtime 路径启动，
并输出原有检测/定位结果。

- [ ] **步骤 4：验证 TensorRT launch 路径**

```bash
source devel/setup.bash
roslaunch two_stream yolo_topic.launch \
  model_path:=/home/SENSETIME/wenkai/YOLOV8_IR_RGB/runs/train/transformer-aifi-s/weights/aifi-best-fp16.engine
```

确认节点打印 TensorRT 后端初始化成功，接收同一对 RGB/IR 图像并完成检测、特征
匹配、定位和标注发布。

- [ ] **步骤 5：做同帧结果和端到端耗时对比**

使用相同输入图像和阈值，记录 ONNX 与 TensorRT 的检测框、类别、置信度、三维
结果及回调耗时。FP16 允许小幅数值误差，但不得出现系统性漏检、类别变化或坐标
映射变化。

---

## 计划自审

- 预处理不变：任务 3 步骤 4 只在现有输入 vector 准备完成后切换执行分支。
- 后处理不变：两个后端都调用现有 `decodeYolov8Detections()`。
- 不现场构建 engine：任务 2 只读取并反序列化序列化文件，任务 4/5 仅切换路径。
- 跨机器回退：launch 默认 ONNX，可通过同一个 `model_path` 参数选择外部 engine。
- 资源复用：任务 2 创建持久化 `BufferManager` 和 CUDA stream，任务 3 每帧只拷贝和执行。
- 错误边界明确：不支持扩展名、binding 不匹配、反序列化失败和 CUDA 执行失败均报错。
- 工作区规则：不包含自动 `git add`、`git commit` 或 `git push` 操作。
