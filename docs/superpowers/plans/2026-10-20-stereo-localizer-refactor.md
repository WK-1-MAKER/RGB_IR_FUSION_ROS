# Stereo Localizer Refactor Plan

> 保持目标检测、匹配、极线约束、校正回退、三角化、坐标变换和目标聚合行为不变，仅移动代码归属。

**Goal:** 将双目几何与定位流程从 `YoloNode` 拆成可独立测试的 `StereoLocalizer`。

**Architecture:** 节点保留图像配准 H、检测、SuperPoint/LightGlue 特征生成和 ROS 输入输出。`StereoLocalizer` 接收相机标定、匹配关键点/索引、RGB 检测框，完成 F 极线筛选、校正/回退、SVD+已有优化、坐标变换和每框聚合，返回筛选后的匹配和 body 坐标位置。

**Tech Stack:** ROS1 C++14、OpenCV、现有 GTest。

- [x] 核实原算法边界、状态及回退路径。
- [x] 新增 `stereo_localizer.h/.cpp` 和针对公共接口的测试。
- [x] 将主循环改为通过 localizer 的窄接口调用。
- [x] 清理 YoloNode 中不再使用的几何状态和方法。
- [x] 编译节点、运行几何和点云测试并检查 diff。
