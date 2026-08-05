# 视觉项目交接说明

> 对接基线：2026-08-04 / 多项目契约 1.0.0

## 1. 本项目负责什么

视觉项目负责接收摄像头图像，运行人脸、人体、手势、目标检测与人脸识别，选择单个稳定 `active_target`，并提供视觉查询和人脸注册 Service。它不判断行为优先级，不根据语音直接控制底盘。

主要入口：

- `realsense2_camera`：RealSense 彩色图像与 IMU 驱动。
- `marsdog-vision-interaction`：视觉推理与 Service。
- `marsdog-vision-viewer`：带框调试界面和 Debug Image。

## 2. 对外接口

| 方向 | 接口 | 类型 | QoS/说明 |
|---|---|---|---|
| 订阅 | `/camera/camera/color/image_raw` | `sensor_msgs/msg/Image` | BEST_EFFORT, KEEP_LAST 1 |
| 发布 | `/perception/visual_event` | `std_msgs/msg/String` JSON | BEST_EFFORT, KEEP_LAST 5，默认 10 Hz |
| 发布 | `/perception/vision/enrollment_event` | `std_msgs/msg/String` JSON | RELIABLE, KEEP_LAST 10 |
| 提供 | `/perception/vision/task` | `marsdog_vision_interaction/srv/VisionTask` | 人物/物体查询和人脸管理 |
| Viewer 发布 | `/perception/vision/debug_image` | `sensor_msgs/msg/Image` | 调试用途 |

## 3. 当前相机与坐标约定

当前使用 RealSense 的 640×480@30 彩色流，同时启用陀螺仪、加速度计和
IMU 线性插值；深度流和红外流关闭。

`config/vision.yaml` 当前设置：

```yaml
stereo_enabled: false
stereo_view: left
stereo_min_aspect_ratio: 2.2
single_target_mode: true
```

所有模型处理完整 640×480 彩色画面。输出坐标相对该画面归一化到
`[0,1]`，所以画面正中心是 `x=0.5`。

这是下游跟踪稳定的硬契约。修改相机布局时必须同步验证 Viewer、`body_center` 和 Action 控制器。

## 4. `active_target` 使用规则

下游运动控制必须同时满足：

```text
track_id > 0
tracking_state == "tracking"
last_seen_age_ms 在允许范围内
bbox/body_center 为有限归一化数值
```

重要字段：

- `bbox=[x,y,w,h]`：人体框；`h` 被跟随控制器用作距离近似。
- `body_center=[x,y]`：优先使用躯干关键点中心，降低四肢摆动造成的抽搐。
- `face_center=[x,y]`：有人脸时可辅助居中。
- `track_id`：视觉目标 ID，不等同于人脸 tracker ID。
- `identity`/`is_registered`：人脸识别结果。
- `tracking_state`：`tracking`、`temporarily_lost` 或 `lost`。

同一画面只输出一个主人体目标。目标优先级为已注册身份优先，其次面积和置信度，并使用 IoU、躯干中心距离和切换迟滞保持稳定。

## 5. VisionTask

Service 定义见 `srv/VisionTask.srv`。当前任务：

| 任务 | 用途 |
|---|---|
| `check_person` | 行为树情绪/Social 路由，返回 `present/count` |
| `detect_objects` | Hunger/Exploration 路由，返回 `objects[]` |
| `recognize_face` | 对当前最大人脸识别 |
| `start_face_enrollment` | 开始连续采集注册 |
| `cancel_face_enrollment` | 取消注册 |
| `upload_face` | Base64 图片注册 |
| `list_faces` | 查询已注册人脸 |
| `delete_face` | 删除注册 |

当 Service 不可用时，行为树会回退到最新 `/perception/visual_event` 缓存；缓存超过约 0.5 秒或目标不是 `tracking` 时按无人处理。

## 6. 启动与验证

```bash
cd /home/cat/xbb/MarsDogVisionInteraction
source /opt/ros/humble/setup.bash
uv sync --extra models --extra dev

colcon build --base-paths . --packages-select marsdog_vision_interaction
source install/setup.bash
ros2 launch marsdog_vision_interaction vision.launch.py
```

视觉 launch 不启动相机。RealSense 驱动需提前单独启动并发布
`/camera/camera/color/image_raw`。

另开终端启动 Viewer：

```bash
uv run marsdog-vision-viewer --ros-args \
  -p camera_topic:=/camera/camera/color/image_raw \
  -p visual_topic:=/perception/visual_event
```

检查：

```bash
ros2 topic hz /camera/camera/color/image_raw
ros2 topic hz /perception/visual_event
ros2 topic echo /perception/visual_event --once
uv run pytest
```

要启用 `/perception/vision/task`，需要 colcon 构建并 source 本包；日志出现 `service=unavailable` 表示 Topic 可用但生成的 Service 类型不可用。

## 7. 模型与降级

期望启动日志为 `5/5 models loaded`：YuNet、MediaPipe Pose、MediaPipe Hand、ByteTrack、SFace。缺少 `mediapipe` 或 `supervision` 时会降级，进程不一定退出；负责人必须从日志和输出字段判断能力是否完整。

目标检测模型为 lazy load，第一次 `detect_objects` 可能明显更慢。不要用第一次调用延迟作为稳定性能指标。

## 8. 修改接口时必须回归

- 单目正中心输出 `body_center.x≈0.5`。
- 同一人连续移动时 `track_id` 不应高频切换。
- 无人后 `tracking_state` 正确进入丢失状态，不持续输出陈旧跟踪。
- JSON 始终包含完整顶层字段，即使数组为空。
- Topic QoS 保持 BEST_EFFORT depth 5，与 BT/Action 匹配。
- 修改字段时同步更新 `messages/visual_event.py`、`docs/ROS2_CONTRACT.md`、Viewer 和跨项目 manifest。

## 9. 明确不属于本项目的问题

- 何时启动/停止跟随：语音会话 + 行为树。
- 跟随速度、角速度、死区和 `/cmd_vel`：动作系统。
- 情绪/需求行为选择和排队：行为树。
- 视觉只保证提供稳定、及时、坐标定义明确的事实数据。
