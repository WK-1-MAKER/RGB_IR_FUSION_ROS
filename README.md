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


## 检测框视锥体过滤点云

节点独立订阅 RGB、IR、Odometry 和（非 `camera_only` 模式下的）`PointCloud2`。
每条 Odometry 消息触发一次处理：从各传感器缓存中按 `header.stamp` 选择最近帧，
只有所有必需传感器都通过严格时间门限时才进入检测和跟踪。回调只缓存消息和排队
Odometry；推理与跟踪在唯一处理线程中串行执行。

启动文件提供以下私有参数：

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `lidar_topic` | `/fusion/points` | 输入 `sensor_msgs/PointCloud2` |
| `filtered_point_cloud_topic` | `/fusion/points_filtered` | 逐目标输出过滤点云 |
| `odometry_topic` | `/Odometry` | 处理时间锚点，要求 `frame_id=camera_init` |
| `camera_only` | `false` | 开启后只订阅 RGB、IR、Odometry |
| `sync_queue_size` | `10` | 每路传感器缓存和 Odometry 队列容量 |

传感器时间戳必须与 Odometry 使用相同时间基准。匹配门限为 RGB `<50 ms`、IR
`<40 ms`、LiDAR `<20 ms`；差值等于门限也会拒绝。任一路无合格帧时跳过该
Odometry 锚点并打印原因，成功匹配不打印同步日志。缓存不按时间清理，队列或缓存满时
直接丢弃最旧消息，不打印溢出日志。跟踪时间和 LiDAR 位姿
均使用该锚点的时间戳和 pose。

检测框先从配准图像映射回已去畸变 RGB 图像，再用 RGB 内参反投影四角。
对每条单位射线 `d_rgb`，使用配置中的 RGB 到 body 外参得到
`d_body = R_body_rgb * d_rgb`，公共起点为 `t_body_rgb`。
由相邻射线构造四个侧面，保留侧面以内且位于 RGB 相机前方的点。
输入点云必须已在配置的 body 坐标系中，节点不执行 TF 变换。

每个检测框在输出话题上单独发布一条点云消息，无命中点时发布空点云；
没有检测框时不发布。输出保持输入的时间戳、frame_id、fields 和保留点的完整
原始字节（包括 intensity、ring 等），整理为 `height=1` 的非组织点云。
支持 FLOAT32/FLOAT64 的 x/y/z、行填充和大小端数据。
多个框的视锥体重叠时，同一点可出现在多个输出消息中；标准 PointCloud2
暂不携带目标 ID。该功能独立于 `triangulate` 开关。

当前仅进行角度过滤，没有远裁面，同方向的目标后方背景仍可能保留；
点云聚类和基于点云的目标定位留待后续实现。
