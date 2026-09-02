# 四项目构建、启动与联调手册

## 1. 联调前提

所有终端先按本机安装方式加载 ROS2 环境，并设置可移植的工作区/仓库变量：

```bash
export ROS_DOMAIN_ID=0
export ROS2_WS="${ROS2_WS:-$HOME/ros2_ws}"
export VISION_REPO="${VISION_REPO:-$ROS2_WS/src/MarsDogVisionInteraction}"
export VOICE_REPO="${VOICE_REPO:-$ROS2_WS/src/MarsDogVoiceInteraction}"
```

多机部署时，各主机的 `ROS_DOMAIN_ID`、RMW 实现和网络发现策略必须一致。先用 `ros2 node list` 确认主机间可发现，再排查业务代码。

公共 Action 类型必须先构建：

```bash
cd "$ROS2_WS"
colcon build --symlink-install --packages-select marsdog_interfaces
source install/setup.bash
```

如果直接 `uv run` 视觉或语音节点时日志显示 `service=unavailable`，说明 `.srv` 尚未由 ROS2 构建生成。需要通过 colcon 构建该包并 source 对应 `install/setup.bash`；只有 Topic 推理不受影响，但行为树的视觉 Service 会降级为 Topic 缓存。

## 2. 推荐启动顺序

### 2.1 摄像头

当前使用 RealSense，输出 640×480@30 彩色流并启用 IMU：

```bash
ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true enable_depth:=true align_depth.enable:=true \
  enable_infra1:=false enable_infra2:=false \
  enable_gyro:=true enable_accel:=true unite_imu_method:=2 \
  enable_sync:=true rgb_camera.color_profile:=640x480x30 \
  rgb_camera.enable_auto_exposure:=true
```

验证：

```bash
ros2 topic hz /camera/camera/color/image_raw
ros2 topic echo /camera/camera/color/image_raw --once --field width
ros2 topic echo /camera/camera/color/image_raw --once --field height
ros2 topic hz /camera/camera/aligned_depth_to_color/image_raw
```

预期约 30 Hz、宽 640、高 480。

### 2.2 视觉节点

```bash
cd "$VISION_REPO"
uv run marsdog-vision-interaction \
  --ros-args -p config_path:="$PWD/config/vision.yaml"
```

期望日志包含 `5/5 models loaded` 和
`Face FastAPI ready: http://127.0.0.1:8092/docs`；少模型时节点仍可降级运行，
但对应字段为空。人脸接口固定为主人和家人1～4五个身份、每人最多5张；本机打开
`http://127.0.0.1:8092/docs` 可做样本级新增、查询、替换、删除和图片查看。

需要局域网管理时：

```bash
ros2 launch marsdog_vision_interaction vision.launch.py \
  face_api_host:=0.0.0.0
```

当前所有人脸 CRUD 接口暂不鉴权。仅在可信隔离局域网绑定远程地址，并优先使用
`ssh -L 8092:127.0.0.1:8092`，避免暴露生物数据管理端口。

统一视觉调试（同时启动正式视觉节点和 Viewer）：

```bash
source "$ROS2_WS/install/setup.bash"
ros2 launch marsdog_vision_interaction vision_debug.launch.py
```

不要再同时执行上面的 `vision.launch.py`。如果正式视觉节点已经由系统启动，则改为：

```bash
ros2 launch marsdog_vision_interaction vision_debug.launch.py \
  start_vision_node:=false
```

浏览器打开 `http://127.0.0.1:8765`，确认相机和视觉事件状态均为绿色，再逐项
核对人体骨架、姿态/动作、人脸身份和手势。物体面板通过按钮执行单次检测或
启动/停止 `vision-debug-web` 持续 session；若 Action 已持有其他 session，页面
不会抢占。Viewer 从 `/perception/vision/object_detections` 显示物体框。远程机器使用 SSH
`-L 8765:127.0.0.1:8765` 转发，不建议直接
把含人脸画面的端口暴露到局域网。

### 2.3 语音节点

```bash
cd "$VOICE_REPO"
uv run marsdog-voice-interaction \
  --ros-args -p config_path:="$PWD/config/voice.yaml"
```

验证唤醒事件：

```bash
ros2 topic echo /perception/audio_event
```

### 2.4 动作系统

先构建并 source `marsdog_interfaces` 和动作包，再启用 AGV：

```bash
source "$ROS2_WS/install/setup.bash"
ros2 launch marsdog_action_executor action_executor.launch.py \
  agv_enabled:=true \
  attention_tracking_enabled:=true
```

只有需要语义点位导航时才增加：

```text
navigation_enabled:=true
```

这要求 Nav2 `/navigate_to_pose` 可用。唤醒声源旋转默认使用 Nav2 `/spin`；无 Nav2 时可暂时设置 `wake_orientation_enabled:=false`。

### 2.5 行为树

动作 Action Server 可用后再启动：

```bash
source "$ROS2_WS/install/setup.bash"
ros2 launch marsdog_behavior behavior_tree.launch.py
```

行为树找不到公共 Action 类型时会退化到 Mock executor。联调真车时日志必须出现：

```text
Executor: ActionClientAdapter → /execute_behavior
```

## 3. 一分钟接口检查

```bash
ros2 node list
ros2 topic info -v /perception/audio_event
ros2 topic info -v /perception/visual_event
ros2 topic info -v /emotion/state
ros2 topic info -v /behavior/attention_tracking
ros2 action info /execute_behavior
ros2 service type /perception/vision/task
ros2 service type /perception/voice/task
```

应至少看到：

- `voice_interaction`
- `vision_interaction`
- `action_executor_node`
- `behavior_tree_node`
- Voice → BT 的 `/perception/audio_event` 连接
- Vision → BT/Action 的 `/perception/visual_event` 连接
- Emotion → BT 的 `/emotion/state` 连接；Vision 不应订阅该 Topic
- BT → Action 的 `/behavior/attention_tracking` 连接
- `/execute_behavior` 一台 Server；不能同时启动两个动作执行器

## 4. 分链路冒烟测试

### 4.1 视觉事实

```bash
ros2 topic echo /perception/visual_event --once
```

检查 `active_target.tracking_state`、`body_center[0]`、`bbox[3]`。当前 640×480
RealSense 单目画面下，站在正前方时 `body_center[0]` 应接近 0.5。

陌生人边界可同时观察：

```bash
ros2 topic echo /perception/visual_event
ros2 node info /vision_interaction
```

未登记人脸出现时只应看到 `EVT_VISION_STRANGER`，且 Vision 的订阅列表中不得
出现 `/emotion/state`。情绪节点启动、停止或状态变化均不得改变 Vision 的陌生人
事件名；Behavior Tree 负责把该视觉事实与 `/emotion/state` 组合成最终行为候选。

### 4.2 视觉 Service

```bash
ros2 service call /perception/vision/task \
  marsdog_vision_interaction/srv/VisionTask \
  "{task_id: 'check-001', task_type: 'check_person', params_json: '{}'}"
```

物体流生命周期冒烟测试：

```bash
ros2 topic echo /perception/vision/object_detections

ros2 service call /perception/vision/task \
  marsdog_vision_interaction/srv/VisionTask \
  "{task_id: 'object-start-001', task_type: 'set_object_detection', \
    params_json: '{\"enabled\":true,\"session_id\":\"object-smoke-001\",\
\"rate_hz\":2.0,\"target_labels\":[\"dog toy ball\"],\"lease_sec\":3.0}'}"
```

预期 Topic 的 `schema_version=2`、`source=stream`、
`stream.session_id=object-smoke-001`。使用相同请求在 3 秒前续租；停止续租后预期
收到 `status=stopped,stop_reason=lease_expired`。显式停止可调用：

```bash
ros2 service call /perception/vision/task \
  marsdog_vision_interaction/srv/VisionTask \
  "{task_id: 'object-stop-001', task_type: 'set_object_detection', \
    params_json: '{\"enabled\":false,\"session_id\":\"object-smoke-001\"}'}"
```

### 4.3 行为 Action

```bash
ros2 action send_goal --feedback /execute_behavior \
  marsdog_interfaces/action/ExecuteBehavior \
  "{goal_id: 'smoke-001', behavior_id: 'smoke-001', \
    behavior_name: 'sit_down', priority_level: 1, \
    params_json: '{}', timeout_sec: 10.0}"
```

### 4.4 会话视角跟随

先观察输出：

```bash
ros2 topic echo /behavior/attention_tracking
ros2 topic echo /cmd_vel
```

正常时序：

```text
EVT_VOICE_CALL_NAME
  -> enabled=true, mode=face_body_centering
EVT_VOICE_COMMAND_FOLLOW
  -> enabled=true, mode=follow_owner
EVT_STATE_CHANGED(state=idle)
  -> enabled=false -> /cmd_vel 零速度
```

跟随不前进时依次检查：

1. `active_target.tracking_state` 是否为 `tracking`。
2. 视觉消息是否新于 0.8 秒。
3. 人体框高度 `bbox[3]` 是否小于目标高度 0.68 且差值超过 deadband。
4. 水平误差是否小于 `follow_max_heading_error=0.30`；偏得太多时控制器会先转向，不前进。
5. 是否有正式 Action 正在执行；执行期间后台跟踪会暂停。
6. `agv_enabled` 是否为 true，底盘是否真的订阅同一个 `/cmd_vel`。

### 4.5 充电结算

观察：

```bash
ros2 topic echo /behavior/result_event
```

完整链路：

```text
NEED_ENERGY_* -> BT recharge/restInPlace -> Action
Action SUCCESS(metadata_json.energyValue)
-> BT ACTION_RECHARGE/COMPLETED
-> InternalNeed 更新 Energy 并按阈值发布 RECOVERED
```

到达充电点途中不应被 Lv5 情绪行为抢占；它应留在延迟队列中。只有更高优先级且满足当前中断策略的行为才能抢占。

## 5. 安全约束

- 无障碍物传感器/代价地图保护时，不要在狭窄环境直接运行预录制平移、前进动作。
- 首次调参架空驱动轮或使用急停可触达的低速场地。
- 任何目标丢失、视觉超时、会话结束、Action 取消都必须产生零速度。
- 寻物 Action 必须只接受匹配 `session_id` 的新鲜物体结果，并在所有终态关闭
  视觉租约；视觉节点不得发布 `/cmd_vel`。
- 退出动作节点前先确认 `/cmd_vel` 已归零。
- `/cmd_vel` 不得再接入第二个未仲裁的发布者；可用 `ros2 topic info -v /cmd_vel` 检查。

## 6. 常见故障

| 现象 | 优先检查 |
|---|---|
| `service=unavailable` | `.srv` 未经过 colcon 生成，或未 source install |
| BT 使用 Mock executor | `marsdog_interfaces` 未构建/source，或 Action Server 未启动 |
| 视觉左右跳框 | 检查目标 Track ID、人体关键点和相机是否被错误配置为双目 |
| Viewer 显示窄图/坐标错 | RealSense 当前应为 `stereo_enabled=false`，完整 640×480 参与推理 |
| Web Viewer 没有画面 | 检查 `http://127.0.0.1:8765`、相机 Topic 和页面中的相机消息年龄 |
| Web Viewer 没有物体框 | 正式配置默认关闭；先检查 Action 是否成功开启/续租 `set_object_detection`，再检查 session、Topic、相机和 RKNN 模型 |
| 唤醒后不转向 | `/behavior/attention_tracking`、Nav2 `/spin`、wake frame/角度方向 |
| 跟随只前进一小段 | `follow_owner` 被配置成固定 Twist 动作；正式配置应为 handoff |
| 跟随不前进 | 目标过近、偏角过大、视觉过期、Action 占锁或 AGV 未启用 |
| 左右抽搐 | 目标 ID 频繁变化、躯干中心不稳、死区过小或方向符号错误 |
| 相机断流仍显示人物 | 检查代码/配置是否包含 `camera_stale_timeout_sec=0.5`，目标年龄不得被发布定时器刷新 |
| 静态躺卧触发跌倒 | 应使用时序 GesturePose 引擎；确认不是旧随机 `PoseActionClassifier` |
| Vision 出现陌生人 Alert/Friend 细分事件 | 运行的不是当前版本；Vision 应只发 `EVT_VISION_STRANGER` 且不订阅 `/emotion/state` |
| 陌生人行为未结合情绪或重复/冲突 | 检查 BT 对 `EVT_VISION_STRANGER` 与 `/emotion/state` 的组合逻辑，以及 10 Hz 状态流的 queued/in-flight 去重 |
| 充电一直不结束 | Action 未返回 SUCCESS，或 Result 未携带/转换 `energyValue` |
| 充电被情绪中断 | 运行配置/代码不是带延迟队列和抢占修复的当前版本 |
| 情绪动作后长期静止 | 检查 `emotion_continuation` 配置和 `/emotion/state.triggered` |

## 7. 交付验收清单

- [ ] 四项目能分别安装依赖、单测通过。
- [ ] 所有 Topic 的类型和 QoS 与 `interface_manifest.yaml` 一致。
- [ ] Voice 的同一会话保持相同 `interaction_id`。
- [ ] Vision 在当前相机下只输出单个稳定目标，坐标中心定义正确。
- [ ] Vision 空闲时无物体推理；Action 开启后按 session 发布，停止或租约到期后清空。
- [ ] Vision 对陌生人只发 `EVT_VISION_STRANGER` 且不订阅 `/emotion/state`；组合判断在 BT 验证。
- [ ] BT 在 Action 不可抢占时保留候选，而不是丢弃。
- [ ] Action 只接受行为表中的精确 `behavior_name`。
- [ ] 会话结束 1 秒内 `/cmd_vel` 归零。
- [ ] 目标丢失 0.8 秒后 `/cmd_vel` 归零。
- [ ] 充电 SUCCESS 可在 `/behavior/result_event` 看到实际电量。
- [ ] Debug Viewer 和动作可视化均能显示当前框、行为、ACT 与结果。
- [ ] 修改接口时同步更新权威源码、交接文档和机器清单。
