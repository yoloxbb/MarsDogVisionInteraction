# ROS2 跨项目接口契约

本文描述四个项目当前实现使用的正式接口。消息中的“必须”表示消费者可以据此校验；未列出的附加 JSON 字段应被消费者忽略，以便向后兼容。

## 1. 接口总表

| 接口 | 类型 | QoS | 生产者 | 主要消费者 |
|---|---|---|---|---|
| `/perception/audio_event` | `std_msgs/msg/String` JSON | RELIABLE, KEEP_LAST 10 | Voice | Behavior Tree、情绪系统 |
| `/perception/voice/enrollment_event` | `std_msgs/msg/String` JSON | RELIABLE, KEEP_LAST 10 | Voice | 管理界面 |
| `/perception/voice/task` | `marsdog_voice_interaction/srv/VoiceTask` | Service | Voice | 管理界面/调度端 |
| `/camera/camera/color/image_raw` | `sensor_msgs/msg/Image` | BEST_EFFORT, KEEP_LAST 1 | RealSense | Vision、调试 Viewer |
| `/camera/camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` | BEST_EFFORT, KEEP_LAST 1 | RealSense | 标定/调试消费者 |
| `/perception/visual_event` | `std_msgs/msg/String` JSON | BEST_EFFORT, KEEP_LAST 5 | Vision | Behavior Tree、Action |
| `/perception/vision/object_detections` | `std_msgs/msg/String` JSON v2 | BEST_EFFORT, KEEP_LAST 5 | Vision | Action、Viewer |
| `/perception/vision/enrollment_event` | `std_msgs/msg/String` JSON | RELIABLE, KEEP_LAST 10 | Vision | 管理界面 |
| `/perception/vision/task` | `marsdog_vision_interaction/srv/VisionTask` | Service | Vision | Behavior Tree、管理界面 |
| `/api/v1/faces...` | FastAPI multipart/JSON/JPEG | HTTP；暂不鉴权 | Vision | 本地人脸管理界面 |
| `/emotion/state` | `std_msgs/msg/String` JSON v2 | RELIABLE, KEEP_LAST 10 | Emotion | Behavior Tree |
| `/emotion/signal_event` | `std_msgs/msg/String` JSON | RELIABLE, KEEP_LAST 10 | Emotion | Behavior Tree |
| `/internal_need/state` | `std_msgs/msg/String` JSON v2 | RELIABLE, KEEP_LAST 10 | InternalNeed | Behavior Tree |
| `/internal_need/signal_event` | `std_msgs/msg/String` JSON | RELIABLE, KEEP_LAST 10 | InternalNeed | Behavior Tree |
| `/behavior/attention_tracking` | `std_msgs/msg/String` JSON | RELIABLE, KEEP_LAST 10 | Behavior Tree | Action |
| `/execute_behavior` | `marsdog_interfaces/action/ExecuteBehavior` | Action | Action Server | Behavior Tree Client |
| `/behavior/result_event` | `std_msgs/msg/String` JSON | RELIABLE, KEEP_LAST 10 | Behavior Tree | InternalNeed |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | 默认 depth 10 | Action | AGV base controller |
| `/debug/execute_behavior/{goal,feedback,result}` | `std_msgs/msg/String` JSON | 默认 RELIABLE depth 10 | Action | 调试界面 |
| `/perception/vision/debug_image` | `sensor_msgs/msg/Image` | BEST_EFFORT, KEEP_LAST 5 | Vision Viewer | RViz/调试界面 |

## 2. `/perception/audio_event`

`String.data` 为 `schema_version=1` 的 JSON。完整稳定字段为：

```text
schema_version
header.stamp, header.frame_id
event_type, interaction_id, utterance_id
wake_word, wake_angle, wake_confidence
asr_text, language
speaker_id, speaker_confidence
emotion, action, control
command_id, intent_category, intent_source, intent_confidence
slots[], response_text
is_executable, should_trigger_behavior_tree
danger_type, danger_angle
state, previous_state, state_reason
latency_ms
```

行为树直接处理：

- `EVT_VOICE_CALL_NAME`：启动当前会话的人脸/人体居中。
- `EVT_VOICE_COMMAND_*`：一对一映射为语义 Behavior。
- `EVT_VOICE_COMMAND_FOLLOW`：除发送 `follow_owner` 行为外，把会话跟踪模式切换为 `follow_owner`。
- `EVT_STATE_CHANGED` 且 `state="idle"`：结束匹配 `interaction_id` 的后台跟踪。

典型跟随事件：

```json
{
  "schema_version": 1,
  "header": {"stamp": 1785780000.1, "frame_id": "base_link"},
  "event_type": "EVT_VOICE_COMMAND_FOLLOW",
  "interaction_id": "session-a1",
  "utterance_id": "utt-a1-2",
  "asr_text": "跟着我",
  "action": "FOLLOW",
  "control": "DO",
  "command_id": "CMD_FOLLOW",
  "intent_category": "command",
  "should_trigger_behavior_tree": true
}
```

结束事件：

```json
{
  "schema_version": 1,
  "event_type": "EVT_STATE_CHANGED",
  "interaction_id": "session-a1",
  "state": "idle",
  "previous_state": "listening",
  "state_reason": "interaction_timeout"
}
```

`state_reason` 当前为 `interaction_timeout`、`utterance_limit` 或 `stop_listening`。

## 3. `/perception/visual_event`

`String.data` 为 `schema_version=1` 的 JSON，默认 10 Hz：

当前 30 Hz 相机输入使用 `inference_frame_stride=2`，完整人脸/姿态/手部流水线
最高约 15 Hz；Topic 的 10 Hz 表示发布频率，不等同于模型推理频率。

精确 GesturePose 标签和候选分数通过调试专用
`/perception/vision/gesture_debug` 发布。行为树只消费本节定义的正式
`/perception/visual_event`，不依赖调试 Topic。

```json
{
  "schema_version": 1,
  "header": {"stamp": 1785780000.2, "frame_id": "camera_link"},
  "active_target": {
    "track_id": 46,
    "face_track_id": 12,
    "identity": "owner",
    "identity_confidence": 0.83,
    "identity_state": "known",
    "is_registered": true,
    "bbox": [0.62, 0.08, 0.30, 0.84],
    "face_bbox": [0.70, 0.10, 0.10, 0.16],
    "face_center": [0.75, 0.18],
    "body_center": [0.73, 0.42],
    "pose_state": "standing",
    "pose_action": "",
    "confidence": 0.94,
    "face_confidence": 0.82,
    "selection_reason": "identity (owner)",
    "tracking_state": "tracking",
    "last_seen_age_ms": 33.0
  },
  "faces": [],
  "humans": [],
  "hands": [],
  "tracked_objects": [],
  "events": []
}
```

坐标约束：

- `x/y/w/h` 和中心坐标均相对“推理实际使用的画面”归一化到 `[0,1]`。
- 当前为 RealSense 640×480 单目彩色流，完整画面参与推理，中心为 `x=0.5`。
- 仅在显式开启双目模式且输入满足横拼宽高比时选择单眼画面。
- Service 的人脸注册、识别与物体检测使用和连续感知相同的画面选择规则。

目标有效条件：`tracking_state == "tracking"`、`track_id > 0` 且消息未超过消费者的视觉超时。Action 当前默认视觉超时为 0.8 秒。无有效目标时必须停止底盘。

`speaker_id`、`is_speaking`、`speaker_confidence` 是旧兼容占位，视觉项目不填充跨模态信息。

陌生人脸在 Vision 中始终产生 `EVT_VISION_STRANGER`。Vision 不订阅
`/emotion/state` 或 `/internal_need/state`，也不产生 Alert/Friend 细分视觉事件。
Behavior Tree 分别消费 `/perception/visual_event` 和 `/emotion/state`，在下游完成
陌生人事实与情绪状态的组合判断、候选去重和行为选择。

### 3.1 `/perception/vision/object_detections`

正式配置以 `inference_rate_hz=0` 启动，不持续占用 RKNN。Action 通过
`VisionTask.set_object_detection` 开启带 `session_id` 和租约的数据流；默认
2 Hz、最大 5 Hz。视觉只发布检测数据，搜索、靠近、目标丢失处理和 `/cmd_vel`
全部属于 Action。推理流与单次 Service 串行访问 RKNN，不会并发执行。

```json
{
  "schema_version": 2,
  "header": {"stamp": 1786417000.1, "frame_id": "camera_link"},
  "published_at": 1786417000.2,
  "sequence": 12,
  "source": "stream",
  "status": "ok",
  "stream": {
    "active": true,
    "session_id": "find-object-001",
    "rate_hz": 2.0,
    "confidence": 0.25,
    "target_labels": ["dog toy ball"],
    "lease_remaining_sec": 2.7
  },
  "request": {
    "target_labels": ["dog toy ball"],
    "confidence": 0.25
  },
  "stop_reason": "",
  "inference_latency_ms": 85.3,
  "objects": [],
  "error": ""
}
```

`header.stamp` 对应实际推理相机帧，`published_at` 对应结果发布时间。
`status=ok` 的空数组是有效的“未检测到物体”；`status=error` 时 Action 必须
停止使用旧位置；显式停止或租约失效会发布 `source=control,status=stopped`，
`stop_reason` 分别为 `requested` 或 `lease_expired`。Action 必须同时校验新鲜度、
`source=stream`、`stream.session_id` 和递增 `sequence`，不能只依赖缓存框。

## 4. VoiceTask 与 VisionTask

两个 Service 的传输结构相同，但属于不同 ROS2 package，不能互换：

```text
string task_id
string task_type
string params_json
---
bool success
string task_id
string task_type
string result_json
string error_message
float64 latency_ms
```

### VoiceTask 任务

| `task_type` | 主要参数 | 主要结果 |
|---|---|---|
| `start_speaker_enrollment` | `name`, `required_shots` | `ok`, `step`, `total_steps`, `text` |
| `cancel_speaker_enrollment` | `{}` | `ok` |
| `upload_speaker` | `name`, `audio_base64` | `ok` |
| `verify_speaker` | 可选 `audio_base64` | `speaker_id`, `confidence` |
| `list_speakers` | `{}` | `speakers[]` |
| `delete_speaker` | `name` | `ok` |
| `start_listening` | `{}` | `listening=true` |
| `stop_listening` | `{}` | `listening=false` |

### VisionTask 任务

| `task_type` | 主要参数 | 主要结果 |
|---|---|---|
| `check_person` | `{}` | `ok`, `present`, `count` |
| `detect_objects` | 可选 `confidence`, `target_labels[]` | `ok`, `objects[]`，单帧查询 |
| `set_object_detection` | `enabled`, `session_id`; 开启时可选 `rate_hz`, `confidence`, `target_labels[]`, `lease_sec` | `ok`, `stream` |
| `get_object_detection_state` | `{}` | `ok`, `stream` |
| `recognize_face` | `{}` | `ok`, `user_id`, `confidence`, `matched` |
| `start_face_enrollment` | 固定 `name`, `required_shots`（默认 3，范围1～5） | `ok`, `step`, `total_steps`, `pose`, `prompt` |
| `cancel_face_enrollment` | `{}` | `ok`, `cancelled` |
| `upload_face` | 固定 `name`, `image_base64` | `ok`, `name`, `shots`, `sample_id` |
| `list_faces` | `{}` | `faces[]` |
| `list_face_records` | `{}` | `count`, `allowed_names`, `available_names`, `faces[]` |
| `list_face_samples` | `name` | `shots`, `sample_ids`, `samples[]` |
| `get_face_sample` | `name`, `sample_id` | 单张图片元数据 |
| `replace_face_sample` | `name`, `sample_id`, `image_base64` | `replaced=true` |
| `delete_face_sample` | `name`, `sample_id` | `remaining_sample_ids`, `face_removed` |
| `delete_face` | `name` | `ok` |

人脸固定身份为 `owner/family_member_1～4`，每个身份最多5张，稳定 `sample_id`
为1～5。HTTP 样本 CRUD 见视觉项目 `README.md`；当前接口暂不鉴权，远程绑定仅限
可信隔离局域网，生物数据只保存在视觉设备本地。

`params_json` 必须是 JSON object。过渡期仍兼容 `[ {"key":"...","value":"..."} ]`。
视觉节点测量并返回本次回调的 `latency_ms`；该值包括同步推理时间，但仍不能
替代 Topic 新鲜度或下游硬实时超时判断。

数据流或 Service 的 `detect_objects` 成功结果会在缓存有效期内镜像到
`tracked_objects[]`；停止流会立即清空缓存。真实模型或相机不可用时会发布/返回
失败，不会返回合成物体。目标标签是对 RKNN 已有类别的精确过滤，不会动态扩展
开放词汇。

姿态/手势事件使用关键点时序规则。`EVT_VISION_FALL` 需要直立基线、快速转变
和持续躺卧，静态躺卧不会产生跌倒事件。

## 5. `/behavior/attention_tracking`

行为树发布，Action 订阅：

```json
{
  "schema_version": 1,
  "header": {"stamp": 1785780000.3, "frame_id": "base_link"},
  "interaction_id": "session-a1",
  "enabled": true,
  "mode": "face_body_centering",
  "wake_angle": 25.0,
  "wake_confidence": 0.92,
  "reason": ""
}
```

`mode`：

- `face_body_centering`：只输出角速度，让目标保持在画面中心。
- `follow_owner`：同时输出角速度和前进速度，以人体框高度估计距离。

关闭：

```json
{"schema_version":1,"interaction_id":"session-a1","enabled":false,"mode":"face_body_centering","reason":"interaction_timeout"}
```

Action 必须以 `interaction_id` 和最新 `enabled` 状态为准。Action 正在执行正式 Behavior 时，后台跟踪控制暂停，避免两个控制器同时写 `/cmd_vel`。

## 6. `/execute_behavior`

权威类型：`marsdog_interfaces/action/ExecuteBehavior`。

Goal：

```text
string goal_id
string behavior_id
string behavior_name
int32 priority_level
string params_json
float64 timeout_sec
```

Result：

```text
string goal_id
string behavior_id
string behavior_name
string status
string result
string reason
float64 reward
string emotion_delta_json
string need_delta_json
string metadata_json
```

Feedback：

```text
string goal_id
string behavior_id
string behavior_name
string status
float64 progress
bool safe_to_interrupt
string current_action
string message
```

约束：

- `behavior_name` 大小写敏感，必须精确存在于动作项目 `config/behavior_tree_actions.yaml`。
- `params_json` 和 Result 的三个 `*_json` 字段必须为合法 JSON 字符串。
- `priority_level` 范围为 0–6，数值越小越高。
- `timeout_sec` 是行为总超时，不是单个 Stage 超时。
- Action Result 的业务 `status/result` 才是结算依据，不能只看 ROS2 transport terminal state。
- 当前行为树以 `goal_id` 作为主关联键；过渡客户端若未填 `behavior_id`，消费者应回退到 `goal_id`。

`follow_owner` 的 Action 只负责确认/交接，不能包含固定前进动作；实际跟随由 `/behavior/attention_tracking + /perception/visual_event` 的闭环控制完成。

## 7. `/behavior/result_event`

只有内部需求相关 Behavior 会发布。语音指令、纯情绪表达和 idle 不发布该 Topic。

```json
{
  "event_id": "result-000123",
  "timestamp": 1785780001.0,
  "action_type": "ACTION_RECHARGE",
  "demand_type": "Energy",
  "result_type": "COMPLETED",
  "metadata": {"recoveryMode": "charging", "energyValue": 88}
}
```

终态映射：

| Action/BT 状态 | `result_type` |
|---|---|
| `SUCCESS` / `SUCCEEDED` / `COMPLETED` | `COMPLETED` |
| `FAILURE` / `FAILED` | `FAILED` |
| `TIMEOUT` | `TIMEOUT` |
| `CANCELED` / `CANCELLED` / `INTERRUPTED` | `INTERRUPTED` |

充电契约：

- 只有 `ACTION_RECHARGE + Energy + COMPLETED` 才结算充电。
- `metadata.energyValue` 表示实际电量百分比，不是能量缺口。
- Action 成功时应在 `metadata_json` 返回 BMS 电量；当前无 BMS 时由动作节点参数 `recharge_result_energy_value` 提供，默认 100。
- 行为树会规范化 `energy_value`/`batteryValue` 为 `energyValue`；缺失或非法时按 100 处理。

## 8. Emotion / InternalNeed 输入

行为树只接受 `schema_version="2.0"`：

- `*/state` 是权威当前状态，只更新缓存、失效候选，不制造新的边沿候选。
- `*/signal_event` 是等级/阈值变化事件，用于创建候选。
- 情绪事件名为 `EMO_<NAME>_TRIGGERED`。
- 需求事件名为 `NEED_<DEMAND>_{TRIGGERED|URGENT|OVERFLOW|RECOVERED}`。
- Energy 的 state value 是电量缺口 `100 - 实际电量百分比`。

完整阈值与事件到行为映射由行为树 `docs/event_behavior_table.md` 和 `config/*.yaml` 管理。
