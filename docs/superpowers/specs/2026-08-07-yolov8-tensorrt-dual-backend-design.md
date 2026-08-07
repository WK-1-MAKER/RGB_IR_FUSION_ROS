# YOLOv8 ONNX/TensorRT 双后端部署设计

## 背景

`two_stream` ROS1 软件包当前对外提供统一的 `YoloDetector` 接口，但内部
使用 ONNX Runtime CUDA 执行双输入 YOLOv8 模型。项目已经为 SuperPoint 和
LightGlue 链接 TensorRT。现在已有一个预构建的 TensorRT FP16 engine，其双流
接口如下：

```text
images_rgb: [1, 3, 640, 640]
images_ir:  [1, 3, 640, 640]
output:     [1, 5, 8400]
```

部署需要同时支持这两类模型：在 TensorRT/CUDA/GPU 环境兼容的机器上使用
engine；在其他机器上直接选择 FP32 ONNX 模型，不在运行时重新构建 engine。

## 目标

- 保持 `YoloDetector` 的构造函数和 `detect()` 对外接口不变。
- 通过现有 launch 参数 `model_path` 选择后端：`.onnx` 使用 ONNX Runtime，
  `.engine` 使用 TensorRT。
- 保持现有图像预处理、RGB/IR 输入顺序、YOLOv8 解码/NMS、坐标映射、ROS 节点
  和三维定位行为不变。
- TensorRT 只进行 engine 反序列化，节点启动时绝不从 ONNX 构建 engine。
- 在多帧之间复用 device buffer 和 CUDA stream，避免每帧重复分配 TensorRT 内存。
- 让两个后端输出统一的 CPU 端 float 数据和 shape，使现有
  `decodeYolov8Detections()` 继续作为唯一的后处理入口。
- 当模型文件缺失、损坏或与当前运行时不兼容时，给出清晰错误。

## 非目标

- 不改变 ROS topic、同步策略、相机几何、特征匹配、三角测量或输出发布。
- 不改变 letterbox、归一化、颜色转换、置信度/NMS 阈值或检测框坐标映射。
- 不增加运行时 engine 构建、ONNX 到 TensorRT 的解析流程，也不对失败的
  `.engine` 自动静默回退到 `.onnx`。
- 除非后续证明仅靠扩展名选择不足，否则不增加新的后端参数；`model_path`
  是模型和后端选择的唯一来源。

## 方案比较

### 方案一：仅使用 TensorRT

用 TensorRT 替换 ONNX Runtime，并移除 ONNX 支持。运行时结构最简单，但无法
在没有兼容 TensorRT/GPU 环境的机器上部署，也不利于 ONNX 与 TensorRT 的对比验证。

### 方案二：按扩展名选择双后端（推荐）

保留当前 ONNX Runtime 路径，在同一个 `YoloDetector` 接口内部增加 TensorRT
路径。构造函数根据模型文件的扩展名选择并校验后端。这样 launch 的使用方式
不变，只修改 `model_path` 就能切换模型，也能保留 ONNX 作为跨机器回退方案。

### 方案三：使用 ONNX Runtime TensorRT Execution Provider

让 ONNX Runtime 使用 TensorRT Execution Provider。该方式不能直接消费现成的
序列化 `.engine`，可能会自行构建或缓存另一个 engine，还需要额外的 provider
配置。本次迁移需要确定性地使用给定 engine，因此不采用此方案。

## 架构设计

`YoloNode` 继续按当前方式构造和调用 `YoloDetector`，不感知具体后端。
`YoloDetector` 一次只持有一个后端：

```text
YoloNode
  └── YoloDetector（对外接口不变）
        ├── *.onnx：ONNX Runtime session
        └── *.engine：TensorRT runtime/engine/context/buffer
```

现有预处理函数继续生成两个 CPU float blob。后端差异从两个 blob 准备完成后
开始：

```text
RGB/IR cv::Mat
  → 现有预处理
  → 后端推理
  → CPU float 输出和 shape
  → 现有 YOLOv8 解码/NMS
  → 现有节点几何和定位流程
```

检测器头文件可以继续保留 ONNX Runtime 声明，以支持 ONNX 路径和现有后处理
测试。TensorRT 成员只在选择 `.engine` 时初始化。检测 engine 的加载不需要
TensorRT ONNX parser。

## 后端选择和 launch 行为

构造 `YoloDetector` 时检查 `model_path`：

- `.onnx`（不区分大小写）：完全保持现有 ONNX Runtime CUDA/CPU 行为，包括
  `use_gpu` 参数处理。
- `.engine`（不区分大小写）：初始化 TensorRT，并要求当前机器具有可用 CUDA
  设备。为保持 launch 兼容性，继续接受 `use_gpu`，但 TensorRT 本身必然使用 GPU。
- 其他扩展名：抛出错误，并列出支持的后缀。

launch 可以指向包内 engine，也可以指向外部 engine 路径。换到其他机器时，
只需把 `model_path` 改为 ONNX 文件，无需构建 engine。

## TensorRT 数据流

构造 TensorRT 后端时执行以下步骤：

1. 从 `model_path` 读取序列化 engine 字节。
2. 创建 `IRuntime`，反序列化得到 `ICudaEngine`，再创建 execution context。
3. 按 `images_rgb`、`images_ir`、`output` 名称查找 binding，不依赖 binding 顺序。
4. 校验两个四维 FP32/FP16 输入 binding 和一个 FP32/FP16 输出 binding。当前静态 engine
   应为输入 `[1,3,640,640]`、输出 `[1,5,8400]`；如果遇到动态维度，则在执行前
   使用现有预处理得到的 shape 设置维度。
5. 一次性分配输入/输出 device buffer，并创建一个 CUDA stream。

每次 `detect()` 调用时，后端继续接收现有 RGB/IR FP32 CPU blob；若 engine 输入为
FP16，则只在执行层转换为 FP16 后拷贝到 GPU。执行 context 并同步 stream 后，把
输出拷贝到 CPU vector；FP16 输出先转换为 FP32。输出 shape 使用 engine 报告的
实际 shape（当前为 `{1, 5, 8400}`）传给现有通用解码器。

ONNX 路径继续调用 `session_.Run()` 和现有的
`decodeCurrentYolov8Tensor()` 适配函数。两个后端都会在进入后处理前汇合。

## 错误处理

- 模型路径不存在或不可读：报告路径和选择的后端。
- TensorRT 反序列化失败：提示 engine 可能与当前 TensorRT 版本、CUDA 运行时
  或 GPU 架构不兼容。
- binding 缺失或 rank/类型不兼容：报告期望和实际的 binding 名称、维度及数据类型。
- CUDA 内存分配、拷贝或 enqueue 失败：报告具体失败操作，并交给现有节点错误
  边界抛出异常。
- `.engine` 加载失败时不自动尝试 ONNX；选择 ONNX 必须通过 launch 显式修改
  `model_path`。

## 测试和验收标准

### 单元测试

- 保留当前 YOLO 解码和 ONNX 后处理测试。
- 增加 `.onnx`、`.engine` 和不支持扩展名的后端选择测试，不需要构造 ROS 节点。
- 在存在 CUDA 设备和给定 engine 时增加 TensorRT metadata/load 测试；没有
  TensorRT/GPU 资源时应能明确跳过。

### 集成验证

- 在不改变软件包架构的前提下构建现有 catkin 目标。
- 分别使用 FP32 `.onnx` 和 FP16 `.engine` 启动节点。
- 对完全相同的 RGB/IR 图像比较检测数量、类别 ID、检测框和置信度，允许 FP16
  带来的数值误差。
- 确认后续三维定位和标注图像输出仍然正常。
- 单独测量 ROS 回调端到端延迟和 TensorRT 纯推理延迟，不能把 `trtexec` 的
  kernel-only 结果直接当作完整 ROS 流程速度。

## 兼容性和风险

- 序列化 TensorRT engine 与硬件和运行时相关。给定 engine 已在 TensorRT 8.6.1、
  GTX 1660 Ti（计算能力 7.5）上验证；换到其他机器时可以选择 ONNX，或在节点
  外部重新生成兼容 engine。
- FP32 和 FP16 ONNX 文件大小不同是正常现象。在宣称数值等价前，仍应确认它们
  来自相同 checkpoint 和网络配置。
- TensorRT 加速后，CPU 预处理和 Host↔Device 拷贝可能成为主要耗时。复用 buffer
  可以避免额外的逐帧内存分配，同时不改变周边处理流程。
