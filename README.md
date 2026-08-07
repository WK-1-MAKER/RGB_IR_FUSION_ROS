# two_stream

`two_stream` 是一个 ROS1（catkin）C++ 包，用于部署可见光/红外双流 YOLOv8
目标检测和定位模型。检测器支持 ONNX Runtime 与 TensorRT 两种推理后端，
SuperPoint 和 LightGlue 保持原有实现不变。

## 编译

```bash
cd /home/SENSETIME/wenkai/RGB_IR_FUSION_ROS_YOLOv8
source /opt/ros/noetic/setup.bash
catkin_make -DCMAKE_BUILD_TYPE=Release
```

依赖版本：

- ONNX Runtime GPU 1.18.0
- TensorRT 8.6.1.6
- CUDA 11.8

## 选择检测后端

启动文件仍使用同一个 `model_path` 参数，根据文件扩展名选择后端：

- `.onnx`：使用 ONNX Runtime，适合作为跨机器回退方案；
- `.engine`：使用 TensorRT 直接反序列化已有 engine，不会在节点运行时构建 engine。

默认使用包内 ONNX 模型：

```bash
roslaunch two_stream yolo_topic.launch
```

使用包内 FP16 TensorRT engine：

```bash
roslaunch two_stream yolo_topic.launch \
  model_path:=$(rospack find two_stream)/models/aifi-best-fp16.engine
```

也可以把 `model_path` 指向包外的绝对路径。TensorRT engine 与 GPU 架构、
TensorRT/CUDA 版本相关；如果 engine 无法在目标机器上反序列化，请显式切换到
`.onnx` 模型，程序不会静默回退，也不会现场重建 engine。

## 部署范围

两种后端共用原有图像预处理、YOLOv8 rank-3 输出解码、NMS、ROS 话题和三维定位
流程，仅模型执行层不同。当前部署面向本仓库的双输入模型，TensorRT engine 必须包含
`images_rgb`、`images_ir` 和 `output` 三个 binding，输入输出支持 FP32 或 FP16。
