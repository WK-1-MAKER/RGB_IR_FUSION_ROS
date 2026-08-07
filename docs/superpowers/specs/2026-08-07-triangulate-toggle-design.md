# `traingulate` 后处理开关设计

## 目标

通过 launch 参数 `traingulate` 控制三维定位后处理。默认值为 `true`，保持现有行为；设为 `false` 时只执行图像预处理和 YOLO 检测，并发布检测标注图像。该参数可在 launch 文件中修改，也可通过 `roslaunch ... traingulate:=false` 覆盖。

## 行为

- `traingulate=true`：构建并使用 SuperPoint、LightGlue，执行匹配、三角测量、body 坐标转换和三维位置输出。
- `traingulate=false`：不构建 SuperPoint/LightGlue engine；每帧跳过 SuperPoint、LightGlue、三角测量和三维位置输出。
- 无论开关取值，YOLO 检测、检测框映射、检测标注图像可视化和标注图像话题发布保持可用。

## 实现边界

在 `YoloNode` 构造函数读取私有 ROS 参数 `~traingulate`，并仅在开启时初始化两个特征推理对象。处理线程中将现有特征匹配到三维位置聚合的逻辑置于条件分支内；关闭时向检测可视化传入空的三维位置列表，使检测框仍正常显示。

## 验证

新增轻量级配置测试验证参数名、默认开启和关闭状态；重新构建节点，运行配置、缓冲区、相机几何、YOLO 解码和检测后处理测试。
