# 视觉项目交接说明

> 对接基线：2026-09-02 / 多项目契约 1.3.4

## 1. 本项目负责什么

视觉项目负责接收彩色图、对齐深度和相机内参，运行人脸、多人姿态、手势、目标
检测与人脸识别，发布完整 `human_candidates[]` 和兼容 `active_target`，并提供目标
查询和人脸注册 Service。物体模型由动作系统按任务会话开启，视觉只发布检测事实；
它不判断行为优先级、不发布 `/cmd_vel`、不调用 Nav2。

主要入口：

- `realsense2_camera`：RealSense 彩色图像与 IMU 驱动。
- `marsdog-vision-interaction`：视觉推理与 Service。
- `marsdog-vision-viewer` / `vision_debug.launch.py`：浏览器识别仪表盘、
  OpenCV 兼容窗口和 Debug Image。

## 2. 对外接口

| 方向 | 接口 | 类型 | QoS/说明 |
|---|---|---|---|
| 订阅 | `/camera/camera/color/image_raw` | `sensor_msgs/msg/Image` | BEST_EFFORT, KEEP_LAST 1 |
| 订阅 | `/camera/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/msg/Image` | BEST_EFFORT, KEEP_LAST 1；人体、动物和物体米制距离 |
| 订阅 | `/camera/camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` | BEST_EFFORT, KEEP_LAST 1；反投影内参 |
| 发布 | `/perception/visual_event` | `std_msgs/msg/String` JSON | BEST_EFFORT, KEEP_LAST 5，默认 10 Hz |
| 发布 | `/perception/vision/object_detections` | `std_msgs/msg/String` JSON v2 | BEST_EFFORT, KEEP_LAST 5，正式启动为 0 Hz，任务默认 2 Hz |
| 调试发布 | `/perception/vision/gesture_debug` | `std_msgs/msg/String` JSON | 精确动作名、候选分数和时序命中，仅供调试 |
| 发布 | `/perception/vision/enrollment_event` | `std_msgs/msg/String` JSON | RELIABLE, KEEP_LAST 10 |
| 提供 | `/perception/vision/task` | `marsdog_vision_interaction/srv/VisionTask` | 人物/物体查询和人脸管理 |
| 提供 | `/api/v1/faces...` | FastAPI multipart/JSON/JPEG | 固定5个人、每人5张的样本级增删改查；默认 `127.0.0.1:8092` |
| Viewer 发布 | `/perception/vision/debug_image` | `sensor_msgs/msg/Image` | 调试用途 |

## 3. 当前相机与坐标约定

当前使用 RealSense 的 640×480@30 彩色流，并启用对齐到彩色坐标系的深度流与
彩色 CameraInfo。红外流仍可关闭。缺少深度不影响识别，但所有人体都必须保持
`range_valid=false`，动作系统不得平移靠近。

`config/vision.yaml` 当前设置：

```yaml
stereo_enabled: false
stereo_view: left
stereo_min_aspect_ratio: 2.2
inference_frame_stride: 2
single_target_mode: true
max_num_poses: 5
horizontal_fov_deg: 69.0
depth_fusion:
  enabled: true
```

所有模型处理完整 640×480 彩色画面。输出坐标相对该画面归一化到
`[0,1]`，所以画面正中心是 `x=0.5`。完整人脸/姿态/手部流水线隔帧执行，
30 Hz 相机下推理上限约 15 Hz；`/perception/visual_event` 仍按 10 Hz 发布。

这是下游跟踪稳定的硬契约。修改相机布局时必须同步验证 Viewer、`body_center` 和 Action 控制器。

## 4. 人体候选和 `active_target` 使用规则

下游运动控制必须同时满足：

```text
track_id > 0
vision_epoch 与 target_id 均非空且匹配当前锁定目标
tracking_state == "tracking"
last_seen_age_ms 在允许范围内
bbox/body_center 为有限归一化数值
需要平移时 range_valid == true 且 distance_m 为有限米制距离
```

重要字段：

- `bbox=[x,y,w,h]`：人体框；只能用于画面几何，不能作为米制停车距离。
- `body_center=[x,y]`：优先使用躯干关键点中心，降低四肢摆动造成的抽搐。
- `face_center=[x,y]`：有人脸时可辅助居中。
- `track_id`：视觉目标 ID，不等同于人脸 tracker ID。
- `vision_epoch + target_id`：跨帧锁定键；重启视觉后 epoch 改变，身份名不能代替 ID。
- `bearing_deg`：相对相机光轴角度；`distance_m` 仅在 `range_valid=true` 时可用。
- `identity`/`is_registered`：人脸识别结果。
- `tracking_state`：`tracking`、`temporarily_lost` 或 `lost`。

`human_candidates[]` 最多输出 5 个独立人体 Track；行为树负责“人优先于动物、熟人
优先于陌生人、唤醒后转向正前方优先”等策略。`active_target` 仍只输出一个兼容
主目标，沿用已注册身份优先、其次面积与置信度的旧规则，不能代替多人候选查询。

人体距离使用躯干中心稳健 ROI 的对齐深度中值和 CameraInfo 反投影，并定义为
相机水平 `x-z` 平面的 `hypot(x,z)`，与底盘可运动平面一致，不计相机到躯干的
垂直高度差。编码、内参、
尺寸、frame、时间同步、有效样本或 0.5 秒新鲜度任一不合格均 fail closed：
`range_valid=false,distance_m=null`，不使用人体框高度回退估距。

陌生人脸存在时，视觉统一发布 `EVT_VISION_STRANGER`。视觉不订阅
`/emotion/state` 或 `/internal_need/state`，也不根据机器人内部状态细分视觉事实。
需要结合情绪判断陌生人行为时，由行为树分别消费视觉事件和情绪状态后完成融合。

## 5. VisionTask

Service 定义见 `srv/VisionTask.srv`。当前任务：

| 任务 | 用途 |
|---|---|
| `check_person` | 行为树情绪/Social 路由，返回 `present/count` |
| `query_targets` | 返回人体/动物/物体目标；人体含原子快照 ID，查询时实时重算 age |
| `detect_objects` | 单帧同步查询，返回 `objects[]`，不启动数据流 |
| `set_object_detection` | 动作系统按 `session_id` 开启、续租、更新或关闭检测流 |
| `get_object_detection_state` | 查询当前检测会话、频率、目标标签和剩余租约 |
| `recognize_face` | 对当前最大人脸识别 |
| `start_face_enrollment` | 固定身份槽位；默认保持自然正对摄像头，自动连续采集三张注册 |
| `cancel_face_enrollment` | 取消注册 |
| `upload_face` | Base64 图片注册 |
| `list_faces` | 查询已注册人脸 |
| `list_face_records` | 查询身份、角色、样本数和稳定样本编号 |
| `list_face_samples` / `get_face_sample` | 查询指定身份的全部/单张样本元数据 |
| `replace_face_sample` / `delete_face_sample` | 按 `sample_id=1～5` 替换或删除一张 |
| `delete_face` | 删除注册 |

当 Service 不可用时，行为树会回退到最新 `/perception/visual_event` 缓存；缓存超过约 0.5 秒或目标不是 `tracking` 时按无人处理。

人脸身份仅允许 `owner`、`family_member_1`、`family_member_2`、
`family_member_3`、`family_member_4`。每个身份最多5张 JPG 模板。FastAPI 的
Swagger 页面为 `/docs`，接口形状与声纹样本 CRUD 对齐：POST 新增、GET 列表/单条/
图片、PUT 原位替换、DELETE 单条删除。当前 HTTP 接口暂不鉴权，远程监听仅允许
用于可信隔离局域网。自由姓名旧数据不会自动删除，但不再载入运行时识别索引。

`query_targets.target_types` 支持 `human/person/animal/object`。`animal` 为 cat/dog，
三类玩具属于 `object` 且 `object_kind=toy`。非人体 Track 只复用已经运行的物体检测
结果，本查询不启动或抢占 session；其 ID 是 `epoch:object:<id>`，不是 label。

## 6. 启动与验证

```bash
# 在视觉仓库根目录执行，并预先加载本机 ROS2 环境
uv sync --extra models --extra dev

colcon build --base-paths . --packages-select marsdog_vision_interaction
source install/setup.bash
ros2 launch marsdog_vision_interaction vision.launch.py
```

视觉 launch 不启动相机。RealSense 驱动需提前单独启动，并至少发布彩色图、
对齐深度和 CameraInfo，例如：

```bash
ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true enable_depth:=true align_depth.enable:=true
```

统一启动正式视觉节点和 Viewer：

```bash
ros2 launch marsdog_vision_interaction vision_debug.launch.py
```

该入口默认 `start_vision_node:=true`，不要与 `vision.launch.py` 同时启动。若设备已有
正式视觉节点，只附加页面时传入 `start_vision_node:=false`。

浏览器打开 `http://127.0.0.1:8765`。绿色为人体/姿态骨架，蓝色为人脸，
紫色为物体，橙色为手部，红色为当前目标。正式视觉节点默认不运行物体模型。
页面按钮支持单次检测以及启动/停止持续检测。持续调试固定使用
`vision-debug-web` session 并定期续租；Action 已持有其他 session 时会拒绝启动，
页面不会抢占。动作系统开启寻物 session 后，页面也会显示其 Topic 结果。
GesturePose
面板中的“粗姿态”来自 `pose_state`，精确标签和25项候选分数来自独立调试
Topic，不会改变行为树使用的正式 `/perception/visual_event` 契约。
普通姿态、跌倒和 Stop 手势只有在当前主目标仍为 `tracking`、属于固定人脸库且
达到 `confirmed_known` 时才进入正式 `events[]`；陌生人、`candidate_known` 或无人脸
状态只保留结构化姿态与 GesturePose 调试结果，不上报姿态事件。
“Vision 已发布事件记录”在 Viewer 的 Topic 回调中记录事件开始、持续、结束，
保留序号和目标证据，并支持复制、导出和清空。它仅验证 Vision 发布边界；测试
行为树候选选择或 Action 执行结果时仍须分别检查对应节点日志。

检查：

```bash
ros2 topic hz /camera/camera/color/image_raw
ros2 topic hz /camera/camera/aligned_depth_to_color/image_raw
ros2 topic echo /camera/camera/color/camera_info --once
ros2 topic hz /perception/visual_event
ros2 topic echo /perception/visual_event --once
ros2 topic hz /perception/vision/object_detections
ros2 topic echo /perception/vision/object_detections --once
uv run pytest
```

要启用 `/perception/vision/task`，需要 colcon 构建并 source 本包；日志出现 `service=unavailable` 表示 Topic 可用但生成的 Service 类型不可用。

## 7. 模型与降级

期望启动日志为 `5/5 models loaded`：YuNet、MediaPipe Pose、MediaPipe Hand、ByteTrack、SFace。缺少 `mediapipe` 或 `supervision` 时会降级，进程不一定退出；负责人必须从日志和输出字段判断能力是否完整。

目标检测模型为 lazy load，第一次数据流推理或 `detect_objects` 可能明显更慢。
动作系统必须把模型预热阶段计入搜索状态，不能在收到第一条新鲜结果前向前移动。
数据流采用单会话租约：相同 `session_id` 才能续租或关闭，不同 session 会被拒绝；
租约过期后视觉发布 `status=stopped` 并清空旧框。

姿态/手势由 `providers/gesture_pose_engine.py` 的确定性时序规则识别，每个稳定
`track_id` 使用独立状态。默认阈值来自参考实现，未完成黄金样本校准前不要直接
调阈值。静态躺卧只输出姿态；`EVT_VISION_FALL` 需要直立基线、快速转变和持续
躺卧，并带 30 秒事件冷却。

真实 Provider 加载失败时不会切换到固定人体或固定物体。只有配置文件明确写出
`type: mock` 才产生 Mock 数据。`detect_objects` 的模型、相机帧不可用时 Service
返回 `success=false` 和原因。

## 8. 修改接口时必须回归

- 单目正中心输出 `body_center.x≈0.5`。
- 同一人连续移动时 `track_id` 不应高频切换。
- 两人同框时 `human_candidates[]` 含两个不同 `target_id`，且 snapshot sequence 递增。
- 无人后 `tracking_state` 正确进入丢失状态，不持续输出陈旧跟踪。
- 断开/破坏深度或 CameraInfo 后，0.5 秒内所有人体回退 `range_valid=false`。
- 断开外部相机后 0.5 秒内清空场景数组，目标年龄持续增长而不是被刷新。
- 静态躺卧不触发跌倒；直立到躺卧序列只产生一次跌倒边沿。
- JSON 始终包含完整顶层字段，即使数组为空。
- Topic QoS 保持 BEST_EFFORT depth 5，与 BT/Action 匹配。
- 陌生人脸始终只发布 `EVT_VISION_STRANGER`；`vision_interaction` 的订阅列表中不得
  出现 `/emotion/state`，情绪组合判断属于行为树。
- 物体 Topic 的 `schema_version=2`，Action 只消费 `source=stream` 且匹配
  `stream.session_id` 的结果，同时校验 `header.stamp`、`status`，并在停止、
  错误或超时后立即停止使用旧框。
- 修改字段时同步更新 `messages/visual_event.py`、`docs/ROS2_CONTRACT.md`、Viewer 和跨项目 manifest。

## 9. 明确不属于本项目的问题

- 何时启动/停止跟随：语音会话 + 行为树。
- 跟随速度、角速度、死区和 `/cmd_vel`：动作系统。
- 寻物搜索、目标确认、靠近、丢失恢复和停车：动作系统。
- 陌生人视觉事实与情绪状态的组合判断、情绪/需求行为选择和排队：行为树。
- 视觉只保证提供稳定、及时、坐标定义明确的事实数据。
