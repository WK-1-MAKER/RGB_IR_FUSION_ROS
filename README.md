# two_stream

`two_stream` 是一个 ROS1 (`catkin`) C++ 包，用 ONNX Runtime（支持 CUDA/CPU）运行 YOLO 推理（话题订阅模式）。

## 功能

- 使用 ONNX Runtime 加载 YOLO ONNX 模型
- 支持 `use_gpu=true/false`
- 支持双话题输入（RGB + IR）
- 发布带检测框的图像到 `output_topic`

## 目录说明

- `src/yolo_node.cpp`：ROS 节点入口（参数、订阅发布）
- `src/detector.cpp`：ONNX 推理核心
- `src/utils.cpp`：预处理、后处理、绘制
- `launch/yolo_topic.launch`：图像话题推理

## 关键参数

- `model_path`：双输入 ONNX 模型路径
- `class_names_path`：类别名称文件路径
- `use_gpu`：是否使用 GPU
- `rgb_topic`：RGB 输入图像话题
- `ir_topic`：IR 输入图像话题
- `output_topic`：输出标注图话题
- `conf_threshold`、`iou_threshold`

## 构建

先修改 `CMakeLists.txt` 中的 `ONNXRUNTIME_DIR` 为你的实际路径。

```bash
cd /home/wk/multispectral-object-detection-main/ROS
catkin_make -DCMAKE_BUILD_TYPE=Release
```

## 运行

终端 1：

```bash
cd /home/wk/multispectral-object-detection-main/ROS
source devel/setup.bash
roscore
```

终端 2（话题模式）：

```bash
cd /home/wk/multispectral-object-detection-main/ROS
source devel/setup.bash
roslaunch two_stream yolo_topic.launch \
	model_path:=/ABS/PATH/to/twostream.onnx \
	class_names_path:=/ABS/PATH/to/coco.names \
	rgb_topic:=/camera/color/image_raw \
	ir_topic:=/camera/ir/image_raw \
	output_topic:=/two_stream/image_annotated
```

## 当前最小可跑版本说明

- 节点使用“当前 RGB 帧 + 最新缓存 IR 帧”进行推理（未启用严格时间同步）。
- 如果暂时没有收到 IR 帧，RGB 帧会被跳过。

## onnxruntime版本和tensorrt版本
- onnxruntime-gpu-1.18.0
- tensorrt-8.6.1.6
- CUDA-11.8
