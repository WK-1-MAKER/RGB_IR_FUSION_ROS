# LiDAR 视锥体过滤实现计划

用户已确认：订阅 `/fusion/points`（参数名 `lidar_topic`），发布
`/fusion/points_filtered`；RGB、IR、LiDAR 在构造函数中通过三路
ApproximateTime 同步，检测后使用同组点云。输入点云坐标即 body，不使用 TF。

- [x] 将同步帧扩展为 RGB、IR、LiDAR 三元组；缓冲整体替换，避免串帧。
- [x] 从已去畸变 RGB 检测框四角反投影，方向只乘旋转，光心取外参平移。
      保留四条单位射线、四侧面和相机前向；不设置远裁面。
- [x] 独立 PointCloud2 过滤模块，逐个检测框输出点云，保留 header、fields、
      point_step 和每个保留点的原始字节。支持 FLOAT32/FLOAT64、行填充和字节序。
- [x] 接到检测之后，独立于 triangulate 开关；增加 launch 参数和使用说明。
- [x] 验证旋转和平移、框四边、前后方向、投影等价性、点云字段完整性、
      多框独立过滤、同步帧整体替换。编译节点并运行相关测试、diff 检查。

不自动提交代码。仅角度过滤会保留目标后方同方向背景，后续聚类另行实现。
