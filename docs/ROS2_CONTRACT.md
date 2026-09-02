x number / `0` | 相对相机光轴角度；右侧为正 |
| `bearing_valid/source` | boolean,string | 角度是否有效以及 `camera_intrinsics/configured_hfov` 来源 |
| `range_valid` | boolean / `false` | 只有对齐深度、内参、时间、ROI 全部有效时为 true |
| `distance_m` | number/null | 人体躯干 ROI 反投影后的相机水平 `x-z` 平面距离，米 |
| `range_source` | string / `none` | 有效米制距离固定为 `aligned_depth`；无效时为 `none` |
| `depth_sync_delta_ms` | number/null | 人体源彩色帧与实际选中深度帧的绝对时间差，毫秒 |
| `pose_3d` | object | `valid,frame_id,x,y,z`；无效时坐标均为 null |

运动控制使用主目标前必须同时满足：

```text
track_id > 0
tracking_state == "tracking"
消息接收时间未超过消费者超时
last_seen_age_ms 在消费者允许范围内
bbox/body_center 是有限的归一化数值
需要平移时还必须 range_valid == true 且 distance_m 有限
```

`human_candidates[]` 使用同一组目标、身份、几何、新鲜度和距离字段。正式配置
`max_num_poses=5`，因此多人场景不再只解码主目标；谁是社交对象或唤醒者由行为树
选择，视觉节点不把“已知人优先”等业务策略写进候选列表。`active_target` 仅保留
旧消费者需要的单目标视图。

深度只接受 `16UC1/mono16`（毫米）或 `32FC1`（米），支持 ROS 行 padding 和
大小端。人体距离取躯干中心附近 ROI 的有效样本中值，并用彩色相机内参反投影；
`distance_m=hypot(pose_3d.x,pose_3d.z)`，表示底盘可运动的相机水平 `x-z` 平面
距离，不计相机高度与人体躯干采样点之间的垂直 `y` 偏移。
节点保留有界、稀疏采样的深度历史，并为人体源彩色帧选择时间戳最近的深度帧；
两者差值仍必须不超过 0.10 秒。编码、尺寸、内参、frame、时间同步、深度范围、
有效样本比例任一不合法，或深度流超过 0.5 秒未更新，均输出
`range_valid=false,distance_m=null`，不得回退成框大小估距。

### 4.4 `faces[]`、`humans[]`、`hands[]` 和 `tracked_objects[]`

`faces[]` 每项：

| 字段 | 类型 / 默认值 | 含义 |
|---|---|---|
| `track_id` | integer / `-1` | 人脸 tracker ID |
| `x,y,w,h` | number / `0` | 归一化人脸框 |
| `confidence` | number / `0` | YuNet 检测分数 |
| `recognized_user` | string / `""` | 识别名称；未确认时为空 |
| `identity_confidence` | number / `0` | SFace 置信量 |
| `identity_state` | string / `"unverified"` | 与 `active_target.identity_state` 相同状态机 |
| `quality` | number / `0` | 当前实现等于人脸检测分数，不是独立清晰度模型 |

`humans[]` 每项：`track_id,x,y,w,h,confidence,pose_state,pose_action,
pose_action_label,keypoints[]`。字段语义与 `active_target` 对应字段相同。

`hands[]` 每项：

| 字段 | 类型 / 默认值 | 含义 |
|---|---|---|
| `handedness` | string / `""` | MediaPipe 返回的 `Left` 或 `Right` |
| `hand_action` | string / `""` | 折叠后的手势兼容名，见 4.7 |
| `hand_action_label` | string / `""` | 中文显示名 |
| `landmarks` | array / `[]` | 21 点手部关键点 |

当前 HandLandmarker 没有手到人体的多目标关联；规则引擎只在单主目标模式使用
手部结果，并把本帧选出的同一个兼容 `hand_action` 写入所有当前手项。因此
`hands[i].hand_action` 不能解释为第 `i` 只手独立分类的结论。

`tracked_objects[]` 是 `/perception/vision/object_detections` 最近成功结果的
短时镜像。除检测框字段外还包含 `vision_epoch,target_id,target_type,track_id,
tracking_state,last_seen_age_ms,range_valid,distance_m` 等稳定轨迹事实；但镜像本身
没有 session、sequence 和错误状态，因而不得单独用于寻物运动控制。

### 4.5 关键点格式

Pose 关键点对象：

```json
{"id": 11, "x": 0.42, "y": 0.31, "z": -0.08,
 "confidence": 0.93, "presence": 0.98}
```

共 33 点，常用 ID：`0` 鼻尖，`11/12` 左/右肩，`13/14` 左/右肘，`15/16`
左/右腕，`23/24` 左/右髋，`25/26` 左/右膝，`27/28` 左/右踝。
`confidence` 对应 MediaPipe visibility。

手部关键点对象为 `{"id", "x", "y", "z"}`，共 21 点。`0` 是腕部，
`1..4` 拇指，`5..8` 食指，`9..12` 中指，`13..16` 无名指，`17..20` 小指。

### 4.6 粗姿态 `pose_state`

`pose_state` 只是单帧粗分类：`standing`、`sitting`、`lying`、`unknown`。
它与 GesturePose 精确动作不是同一字段。特别是：

- `lying` 只表示当前画面像躺卧，不会单独触发跌倒报警。
- 只拍到上半身、关键腿部点不可用时，粗姿态可能不稳定；消费者不得把
  `pose_state` 当作安全事件。
- `EVT_VISION_FALL` 使用独立时序状态机，要求直立基线、快速下降/转倒和持续
  躺卧。

### 4.7 25 个精确动作到正式字段和事件的映射

精确动作名只稳定出现在 `/perception/vision/gesture_debug`。正式 Topic 为兼容
既有下游，将它们折叠为 `pose_action` 或 `hand_action`。所有姿态/手势事件都要求
`active_target.identity` 是固定人脸库中的 `owner` 或 `family_member_1`～
`family_member_4`，且同时满足 `tracking_state == "tracking"`、
`identity_state == "confirmed_known"`。`candidate_known`、
陌生人和未确认身份仍会输出结构化姿态字段及调试 Topic，但不会下发事件。

| 优先级 | 精确 GesturePose 名 | 组 | 正式兼容字段和值 | 派生正式事件 |
|---|---|---|---|---|
| P0 | `fall` | event | `pose_action=fallen_down` | `EVT_VISION_FALL`，需确认固定身份 |
| P0 | `stop_gesture` | event | `hand_action=stop_gesture` | `EVT_VISION_STOP_GESTURE`，需确认固定身份 |
| P1 | `hands_on_hips` | gesture | `pose_action=hands_on_hips` | `EVT_VISION_MASTER_SAD`，需确认身份 |
| P1 | `large_arm_swing` | dynamic | `pose_action=rapid_wave_slap` | `EVT_VISION_MASTER_SAD`，需确认身份 |
| P1 | `pointing` | gesture | `hand_action=finger_pointing` | `EVT_VISION_MASTER_SAD`，需确认身份 |
| P1 | `stomping` | dynamic | `pose_action=stomping` | `EVT_VISION_MASTER_SAD`，需确认身份 |
| P1 | `arms_crossed` | gesture | `pose_action=arms_crossed` | `EVT_VISION_MASTER_SAD`，需确认身份 |
| P2 | `head_down` | posture | `pose_action=head_down_slumped` | `EVT_VISION_MASTER_SAD`，需确认身份 |
| P2 | `shoulders_slumped` | posture | `pose_action=head_down_slumped` | `EVT_VISION_MASTER_SAD`，需确认身份 |
| P2 | `face_covering` | gesture | `hand_action=hands_covering_face` | `EVT_VISION_MASTER_SAD`，需确认身份 |
| P2 | `hands_on_head` | gesture | `hand_action=hands_covering_face` | `EVT_VISION_MASTER_SAD`，需确认身份 |
| P2 | `curled_up` | posture | `pose_action=body_curled_up` | `EVT_VISION_MASTER_SAD`，需确认身份 |
| P2 | `hunched` | posture | `pose_action=hunched_back` | `EVT_VISION_MASTER_SAD`，需确认身份 |
| P3 | `arms_raised` | gesture | `pose_action=arm_raise_wave` | `EVT_VISION_MASTER_HAPPY`，需确认身份 |
| P3 | `waving` | dynamic | `pose_action=arm_raise_wave` | `EVT_VISION_MASTER_HAPPY`，需确认身份 |
| P3 | `victory` | gesture | 无；仅调试 Topic | 无 |
| P3 | `jumping` | dynamic | `pose_action=jump` | `EVT_VISION_MASTER_HAPPY`，需确认身份 |
| P3 | `arms_open` | gesture | `pose_action=lean_forward_arms_open` | `EVT_VISION_MASTER_HAPPY`，需确认身份 |
| P3 | `fast_nod` | dynamic | `pose_action=nodding` | `EVT_VISION_MASTER_HAPPY`，需确认身份 |
| P3 | `clapping` | dynamic | `hand_action=clapping` | `EVT_VISION_MASTER_HAPPY`，需确认身份 |
| P3 | `thumbs_up` | gesture | `hand_action=thumbs_up` | `EVT_VISION_MASTER_HAPPY`，需确认身份 |
| P4 | `standing` | posture | `pose_action=neutral_stand_sit` | `EVT_VISION_MASTER_NEUTRAL`，需确认身份 |
| P4 | `sitting` | posture | `pose_action=neutral_stand_sit` | `EVT_VISION_MASTER_NEUTRAL`，需确认身份 |
| P4 | `lying` | posture | 无；由 `pose_state=lying` 表达 | 无 |
| P4 | `low_motion` | activity | `pose_action=neutral_stand_sit` | `EVT_VISION_MASTER_NEUTRAL`，需确认身份 |

一帧可以同时有一个 `pose_action` 和一个 `hand_action`。P0–P4 是规则输出优先级，
不是 ROS QoS，也不等同于行为树候选优先级。

### 4.8 当前会实际产生的 `events[]`

事件生成顺序固定为：人脸 → 主目标姿态 → 手势 → 物体；同一事件名在一个数组
内去重。`events[]` 只有字符串，没有独立时间、置信度或目标 ID；相关细节必须从
同一份消息的结构化字段读取。

| 事件名 | 当前触发条件 | 同包中应读取的证据 |
|---|---|---|
| `EVT_VISION_MASTER` | `faces[]` 非空，且主目标 `identity` 非空且不为 `unknown` | `active_target.identity`、`faces[]` |
| `EVT_VISION_STRANGER` | `faces[]` 非空，且主目标身份未知；不读取或组合情绪状态 | `faces[]`、`active_target.identity_state` |
| `EVT_VISION_MASTER_HAPPY` | `identity_state=confirmed_known` 的主目标出现表 4.7 的 happy 兼容动作 | `pose_action` 或 `hands[].hand_action` |
| `EVT_VISION_MASTER_SAD` | `identity_state=confirmed_known` 的主目标出现表 4.7 的 sad 兼容动作 | `pose_action` 或 `hands[].hand_action` |
| `EVT_VISION_MASTER_NEUTRAL` | `identity_state=confirmed_known` 的主目标出现 `neutral_stand_sit` | `pose_action`、`pose_state` |
| `EVT_VISION_FALL` | 固定人脸库身份达到 `confirmed_known`，且 `pose_action=fallen_down`；已确认从直立快速转为持续躺卧 | `active_target.identity/identity_state`、`pose_action`；调试时看 `fall_detector` |
| `EVT_VISION_STOP_GESTURE` | 固定人脸库身份达到 `confirmed_known`，且任一 `hands[].hand_action=stop_gesture` | `active_target.identity/identity_state`、`hands[]` |
| `EVT_VISION_FOOD` | `tracked_objects[].label` 精确为 `dog bowl`、`dog food can` 或 `dog treat bag` | `tracked_objects[]` |
| `EVT_VISION_TOY` | `tracked_objects[].label` 精确为 `dog toy ball`、`dog frisbee toy` 或 `dog tug ring toy` | `tracked_objects[]` |

FOOD/TOY 的事件映射按上述英文标签做区分大小写的精确比较；这与物体 session 的
`target_labels` 过滤（不区分大小写）是两个不同步骤。

人脸事件当前以“是否存在任意 `faces[]` + 主目标身份”组合判断，并不逐张人脸绑定
事件。多人物场景的消费者应读取 `active_target`，不要自行假定事件对应
`faces[0]`。

以下常量虽然已在代码中声明，但当前没有任何映射或发布路径，属于预留事件，
行为树不能依赖它们：

| 预留事件 | 当前状态 |
|---|---|
| `EVT_VISION_ANIMAL_CALM` | 仅声明，当前不会产生 |
| `EVT_VISION_ANIMAL_GREET` | 仅声明，当前不会产生 |
| `EVT_VISION_ANIMAL_PLAY` | 仅声明，当前不会产生 |
| `EVT_VISION_ANIMAL_BOUNDARY` | 仅声明，当前不会产生 |

### 4.9 事件持续与去重语义

`/perception/visual_event` 是 10 Hz 状态流，因此事件可能在连续消息中重复：

- 人脸在画面中持续存在时，`EVT_VISION_MASTER` 或 `EVT_VISION_STRANGER` 会重复。
- 平滑后的动作仍有效时，对应情绪或手势事件会重复。
- 跌倒状态机内部的 `fall_event_triggered` 只在确认帧为一次边沿，但正式
  `EVT_VISION_FALL` 来自短暂保持的 `fallen_down` 兼容动作，可能出现在多个
  Topic 包中；同一次跌倒有 30 秒重新触发冷却。
- 物体结果在缓存有效期内镜像时，FOOD/TOY 事件会重复。

因此，下游若需要“一次行为”，必须按自己的候选/in-flight/cooldown 规则去重，
不能把 `events[]` 中每次出现都当作新的物理事件。

`vision_debug.launch.py` 的网页会在 Viewer Topic 回调层把上述状态流压缩为
`ENTER/ACTIVE/EXIT` 生命周期记录，并显示同包的身份、姿态、手势证据及
`vision_epoch/sequence`。这属于进程内调试数据，不修改正式 ROS schema，也不构成
行为树已选中或 Action 已执行的确认；Viewer 重启或页面点击“清空”后历史消失。
默认保存最近200条，可用 `event_history_limit` 在20～1000之间调整。

### 4.10 陌生人事实与下游融合边界

Vision 只依据当前视觉观察判断已登记人或陌生人：

```text
faces[] 非空且 active_target.identity 为固定已知身份
  -> EVT_VISION_MASTER
faces[] 非空且 active_target.identity 为空或 unknown
  -> EVT_VISION_STRANGER
```

Vision 不订阅 `/emotion/state` 或 `/internal_need/state`，不读取 Anxiety、Fear、Joy、
Excite、Calm 等内部状态，也不发布 `EVT_VISION_STRANGER_ALERT` 或
`EVT_VISION_STRANGER_FRIEND`。因此情绪节点是否启动、状态是否新鲜或 JSON 是否
合法，都不能改变 Vision 的陌生人事件名。

Behavior Tree 是组合判断的唯一责任方：分别消费 `/perception/visual_event` 的
`EVT_VISION_STRANGER` 和 `/emotion/state`，在自身候选、优先级、queued/in-flight
去重与冷却生命周期内决定最终陌生人行为。组合结果属于行为决策，不得写回或伪装成
Vision 已观察到的新事件。

## 5. `/perception/vision/gesture_debug`

该 Topic 仅供 Viewer、标定和规则回归使用，不属于行为树稳定输入。根字段包括：

| 字段 | 含义 |
|---|---|
| `schema_version`、`stamp` | 调试 schema 和发布时间 |
| `track_id`、`tracking_state`、`pose_state` | 当前主目标与粗姿态 |
| `legacy_pose_action`、`legacy_hand_actions[]` | 映射到正式 Topic 的兼容名 |
| `face_observed` | YuNet 是否明确看到当前人脸；禁用人脸任务时可为 `null` |
| `primary_action`、`primary_priority`、`state_hint` | 精确主动作、P0–P4 和状态候选 |
| `recognized_actions[]` | 经过时序平滑的动作；含 `name,priority,group,confidence,support_ratio,duration_s` |
| `raw_scores[]` | 25 个规则原始分数；含 `name,priority,group,score`，按分数降序 |
| `fall_phase` | `unknown/monitoring/falling/lying/recovering` |
| `fall_event_triggered` | 跌倒确认帧的一次边沿 |
| `fall_alert_active` | 跌倒确认后的短暂保持状态 |
| `fall_detector` | `phase,armed,lying_score,transition_score,event_triggered,alert_active,cooldown_remaining_s` |
| `hand_features.left/right` | 手是否检出、伸指状态、掌心分数、尺度和运动能量 |
| `temporal_features` | 窗口、人体/头/手运动和双腕开合量 |
| `feature_ms`、`recognition_ms` | 特征与规则耗时 |
| `landmarker` | 模型变体、运行模式、帧计数、有效 FPS、平均/P95 耗时和检测率 |

这些诊断字段允许在不升级正式 v1 schema 的情况下增加。调试消费者必须忽略未知
字段，正式消费者不得用 `raw_scores` 直接触发机器人动作。

## 6. `/perception/vision/object_detections`

### 6.1 生命周期

- `schema_version`：`2`
- 正式 `vision.yaml` 启动频率：0 Hz，即空闲时不运行物体模型
- 按需默认频率：2 Hz；允许范围 `(0,5]` Hz
- 默认租约：3 秒；允许范围 `[0.5,30]` 秒
- `vision_debug.launch.py` 是统一调试入口；页面按钮可调用 `detect_objects` 单次检测，
  或使用固定 `vision-debug-web` session 启停持续检测，不抢占其他 session

动作系统用 `VisionTask.set_object_detection` 开启、续租和关闭唯一数据流。
同一 `session_id` 才能更新或关闭现有流；不同 session 的操作会失败，不能抢占。

### 6.2 schema v2

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
    "lease_remaining_sec": 2.74
  },
  "request": {
    "target_labels": ["dog toy ball"],
    "confidence": 0.25
  },
  "stop_reason": "",
  "inference_latency_ms": 85.3,
  "objects": [{
    "vision_epoch": "...", "target_id": "...:object:3",
    "target_type": "object", "track_id": 3,
    "label": "dog toy ball",
    "x": 0.32, "y": 0.45, "w": 0.12, "h": 0.12,
    "confidence": 0.91,
    "center_x": 0.38, "center_y": 0.51,
    "tracking_state": "tracking", "last_seen_age_ms": 0.0,
    "range_valid": true, "distance_m": 1.42,
    "range_source": "aligned_depth"
  }],
  "error": ""
}
```

| 字段 | 类型 | 语义 |
|---|---|---|
| `header.stamp` | number | 实际参与本次推理的相机帧时间 |
| `published_at` | number | 检测完成并发布的时间 |
| `sequence` | integer | 节点进程内递增的物体结果序号 |
| `source` | string | `stream` 持续流、`service` 单次查询、`control` 停止终态 |
| `status` | string | `ok`、`error`、`stopped` |
| `stream` | object | session 所有者、频率、阈值、标签和剩余租约 |
| `request` | object | 本次推理实际使用的 `target_labels` 和 `confidence` |
| `stop_reason` | string | `requested` 或 `lease_expired`；非停止包为空 |
| `inference_latency_ms` | number | 本次物体模型调用耗时 |
| `objects` | array | 稳定物体轨迹；包含检测框、稳定 ID、新鲜度以及经时间同步的对齐深度距离 |
| `error` | string | 错误原因；正常为空 |

状态约定：

- `status=ok,objects=[]` 是有效的“本帧未检测到目标”，不是故障。
- `status=error` 表示相机、模型或推理失败；旧物体缓存同时清空。
- `status=stopped` 表示显式关闭或租约到期；旧物体缓存同时清空。
- Action 只消费 `source=stream`、`status=ok`、session 匹配且足够新鲜的结果。
  `source=service` 结果不得混入当前寻物运动闭环。
- `target_labels` 最多 32 项，每项最长 128 字符；过滤时不区分大小写但要求
  完整标签精确匹配。它只能过滤 RKNN 导出模型已有类别，不能运行时增加新词汇。
- 当前只提供二维归一化框，不提供真实深度或米制距离。

## 7. `/perception/vision/enrollment_event`

调用 `start_face_enrollment` 成功后，视觉节点在每次视觉发布周期处理最新相机帧，
并可靠发布注册进度。没有活动会话时不发布。

可能的 `status`：

| 状态 | 主要字段 | 含义 |
|---|---|---|
| `searching` | `ok,name,step,total_steps,prompt,done=false` | 未检测到合格人脸 |
| `tracking` | 加 `confidence,progress_pct` | 人脸已进入画面，等待稳定帧数 |
| `captured` | 加 `shots` | 已保存一张，继续下一步 |
| `done` | 加 `shots,done=true` | 注册完成；只发布一次并立即结束会话 |

示例：

```json
{
  "ok": true,
  "name": "owner",
  "step": 2,
  "total_steps": 3,
  "status": "tracking",
  "confidence": 0.95,
  "progress_pct": 66,
  "done": false
}
```

该 Topic 当前没有 `schema_version`、`header` 或 `task_id`。管理端必须用自己发起的
单活动注册会话关联进度；若未来需要并发会话，必须先升级契约。

## 8. `/perception/vision/task`

### 8.1 ROS Service 传输结构

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

- 响应原样回传 `task_id` 和 `task_type`。
- `params_json` 应是 JSON object；过渡期兼容 `[ {"key":"...","value":...} ]`。
- 正常进入任务分发后，`result_json` 是 JSON object；若 `params_json` 本身无法
  解码等异常发生在分发前，`success=false`、`error_message` 有值且
  `result_json` 可能为空串。
- `success` 来自结果中的 `ok`；失败时 `error_message` 同步填入 `error`。
- `latency_ms` 是整个 Service 回调耗时，包含同步模型推理。

### 8.2 全部 `task_type`

| `task_type` | `params_json` | 成功 `result_json` | 副作用/说明 |
|---|---|---|---|
| `check_person` | `{}` | `ok,present,count` | 读取最近视觉观察 |
| `query_targets` | 可选 `target_types,min_confidence,max_age_ms` | `ok,header,vision_epoch,sequence,snapshot_id,snapshot_age_ms,targets[],human_candidates[],animal_candidates[],object_candidates[],active_target` | 只读目标事实；查询时实时重算年龄，不启动物体流 |
| `detect_objects` | 可选 `confidence,target_labels[]` | `ok,objects[]` | 同步单帧推理，并发布 `source=service`；不启动持续流 |
| `set_object_detection` | `enabled,session_id`；开启时可选 `rate_hz,confidence,target_labels[],lease_sec` | `ok,stream`；关闭时还含 `stopped_session_id` | 开启、续租、更新或关闭唯一物体流 |
| `get_object_detection_state` | `{}` | `ok,stream` | 只读当前 session 状态 |
| `recognize_face` | `{}` | `ok,user_id,confidence,matched` | 对当前最大人脸同步识别 |
| `start_face_enrollment` | 固定 `name`，可选 `required_shots`（默认 3，范围1～5） | `ok,name,step,total_steps,pose,prompt` | 在该身份剩余槽位中新建注册会话；自然面对摄像头，无动作要求 |
| `cancel_face_enrollment` | `{}` | `ok,name,cancelled` | 无会话时失败 |
| `upload_face` | 固定 `name,image_base64` | `ok,name,shots,sample_id,sample_key,image_path` | 解码图片、检测并保存最大人脸，复用最小空闲编号 |
| `list_faces` | `{}` | `ok,faces[]` | 返回注册名称，按字典序排序 |
| `list_face_records` | `{}` | `ok,count,max_faces,max_samples_per_face,allowed_names,available_names,faces[]` | 返回固定身份的样本汇总 |
| `list_face_samples` | `name` | `ok,name,role,shots,sample_ids,samples[]` | 返回一个身份的样本级记录 |
| `get_face_sample` | `name,sample_id` | `ok,name,role,sample_id,image_path,image_url,...` | 查询一张人脸样本元数据 |
| `replace_face_sample` | `name,sample_id,image_base64` | `ok,name,shots,sample_id,replaced` | 校验成功后原位替换并同步识别模板 |
| `delete_face_sample` | `name,sample_id` | `ok,name,shots,deleted_sample_id,remaining_sample_ids,face_removed` | 只删除一张；最后一张删除后释放身份槽位 |
| `delete_face` | `name` | `ok,name` | 删除本地人脸样本并同步内存库 |

不支持的 `task_type` 返回：

```json
{"ok": false, "error": "unsupported task_type: <value>"}
```

### 8.3 人脸样本 FastAPI

正式节点默认在 `127.0.0.1:8092` 提供 `/docs` 和以下管理接口。身份是固定枚举
`owner/family_member_1～4`，不是自由姓名；每个身份最多5张，`sample_id` 只能为
1～5。

| 方法和路径 | 请求 | 成功用途 |
|---|---|---|
| `POST /api/v1/faces/{name}/samples` | multipart `image=.jpg/.jpeg/.png` | 新增一张，HTTP 201 |
| `GET /api/v1/faces` | 无 | 列出身份汇总和容量 |
| `GET /api/v1/faces/{name}/samples` | 无 | 列出一人的稳定样本编号 |
| `GET /api/v1/faces/{name}/samples/{sample_id}` | 无 | 查询单张元数据 |
| `GET /api/v1/faces/{name}/samples/{sample_id}/image` | 无 | 返回裁剪后的 `image/jpeg` |
| `PUT /api/v1/faces/{name}/samples/{sample_id}` | multipart `image` | 校验后原位替换 |
| `DELETE /api/v1/faces/{name}/samples/{sample_id}` | 无 | 只删除指定样本 |

删除样本不会给其它文件重编号；新增复用1～5中的最小空闲编号。默认上传上限10 MiB。
POST/PUT 先完成图片解码、人脸检测、质量门控和裁剪，成功后才落盘；PUT 校验失败时
旧文件不变。每次新增、替换和删除成功后都重建当前进程的多模板识别索引。

当前 `/health` 和全部人脸 CRUD 接口暂不鉴权，不定义 token、Cookie 或认证请求头。
绑定非回环地址时必须部署在可信隔离局域网；推荐 SSH 转发，不得将设备本地人脸
图片和 `face_registry.json` 暴露到不可信网络。

| 状态码 | 含义 |
|---:|---|
| `201` | 新增成功 |
| `400` | 空文件 |
| `404` | 身份或样本不存在 |
| `409` | 单人已有5张或存储状态冲突 |
| `413` | 文件超过配置上限 |
| `415` | 扩展名不是 JPG/JPEG/PNG |
| `422` | 身份非法、图片无法解码、未检出合格人脸或参数越界 |
| `503` | 人脸检测模型或操作处理器不可用 |

### 8.4 人体目标快照查询
xi | `phase,armed,lying_score,transition_score,event_triggered,alert_active,cooldown_remaining_s` |
| `hand_features.left/right` | 手是否检出、伸指状态、掌心分数、尺度和运动能量 |
| `temporal_features` | 窗口、人体/头/手运动和双腕开合量 |
| `feature_ms`、`recognition_ms` | 特征与规则耗时 |
| `landmarker` | 模型变体、运行模式、帧计数、有效 FPS、平均/P95 耗时和检测率 |

这些诊断字段允许在不升级正式 v1 schema 的情况下增加。调试消费者必须忽略未知
字段，正式消费者不得用 `raw_scores` 直接触发机器人动作。

## 6. `/perception/vision/object_detections`

### 6.1 生命周期

- `schema_version`：`2`
- 正式 `vision.yaml` 启动频率：0 Hz，即空闲时不运行物体模型
- 按需默认频率：2 Hz；允许范围 `(0,5]` Hz
- 默认租约：3 秒；允许范围 `[0.5,30]` 秒
- `vision_debug.launch.py` 是统一调试入口；页面按钮可调用 `detect_objects` 单次检测，
  或使用固定 `vision-debug-web` session 启停持续检测，不抢占其他 session

动作系统用 `VisionTask.set_object_detection` 开启、续租和关闭唯一数据流。
同一 `session_id` 才能更新或关闭现有流；不同 session 的操作会失败，不能抢占。

### 6.2 schema v2

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
    "lease_remaining_sec": 2.74
  },
  "request": {
    "target_labels": ["dog toy ball"],
    "confidence": 0.25
  },
  "stop_reason": "",
  "inference_latency_ms": 85.3,
  "objects": [{
    "vision_epoch": "...", "target_id": "...:object:3",
    "target_type": "object", "track_id": 3,
    "label": "dog toy ball",
    "x": 0.32, "y": 0.45, "w": 0.12, "h": 0.12,
    "confidence": 0.91,
    "center_x": 0.38, "center_y": 0.51,
    "tracking_state": "tracking", "last_seen_age_ms": 0.0,
    "range_valid": true, "distance_m": 1.42,
    "range_source": "aligned_depth"
  }],
  "error": ""
}
```

| 字段 | 类型 | 语义 |
|---|---|---|
| `header.stamp` | number | 实际参与本次推理的相机帧时间 |
| `published_at` | number | 检测完成并发布的时间 |
| `sequence` | integer | 节点进程内递增的物体结果序号 |
| `source` | string | `stream` 持续流、`service` 单次查询、`control` 停止终态 |
| `status` | string | `ok`、`error`、`stopped` |
| `stream` | object | session 所有者、频率、阈值、标签和剩余租约 |
| `request` | object | 本次推理实际使用的 `target_labels` 和 `confidence` |
| `stop_reason` | string | `requested` 或 `lease_expired`；非停止包为空 |
| `inference_latency_ms` | number | 本次物体模型调用耗时 |
| `objects` | array | 稳定物体轨迹；包含检测框、稳定 ID、新鲜度以及经时间同步的对齐深度距离 |
| `error` | string | 错误原因；正常为空 |

状态约定：

- `status=ok,objects=[]` 是有效的“本帧未检测到目标”，不是故障。
- `status=error` 表示相机、模型或推理失败；旧物体缓存同时清空。
- `status=stopped` 表示显式关闭或租约到期；旧物体缓存同时清空。
- Action 只消费 `source=stream`、`status=ok`、session 匹配且足够新鲜的结果。
  `source=service` 结果不得混入当前寻物运动闭环。
- `target_labels` 最多 32 项，每项最长 128 字符；过滤时不区分大小写但要求
  完整标签精确匹配。它只能过滤 RKNN 导出模型已有类别，不能运行时增加新词汇。
- 当前只提供二维归一化框，不提供真实深度或米制距离。

## 7. `/perception/vision/enrollment_event`

调用 `start_face_enrollment` 成功后，视觉节点在每次视觉发布周期处理最新相机帧，
并可靠发布注册进度。没有活动会话时不发布。

可能的 `status`：

| 状态 | 主要字段 | 含义 |
|---|---|---|
| `searching` | `ok,name,step,total_steps,prompt,done=false` | 未检测到合格人脸 |
| `tracking` | 加 `confidence,progress_pct` | 人脸已进入画面，等待稳定帧数 |
| `captured` | 加 `shots` | 已保存一张，继续下一步 |
| `done` | 加 `shots,done=true` | 注册完成；只发布一次并立即结束会话 |

示例：

```json
{
  "ok": true,
  "name": "owner",
  "step": 2,
  "total_steps": 3,
  "status": "tracking",
  "confidence": 0.95,
  "progress_pct": 66,
  "done": false
}
```

该 Topic 当前没有 `schema_version`、`header` 或 `task_id`。管理端必须用自己发起的
单活动注册会话关联进度；若未来需要并发会话，必须先升级契约。

## 8. `/perception/vision/task`

### 8.1 ROS Service 传输结构

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

- 响应原样回传 `task_id` 和 `task_type`。
- `params_json` 应是 JSON object；过渡期兼容 `[ {"key":"...","value":...} ]`。
- 正常进入任务分发后，`result_json` 是 JSON object；若 `params_json` 本身无法
  解码等异常发生在分发前，`success=false`、`error_message` 有值且
  `result_json` 可能为空串。
- `success` 来自结果中的 `ok`；失败时 `error_message` 同步填入 `error`。
- `latency_ms` 是整个 Service 回调耗时，包含同步模型推理。

### 8.2 全部 `task_type`

| `task_type` | `params_json` | 成功 `result_json` | 副作用/说明 |
|---|---|---|---|
| `check_person` | `{}` | `ok,present,count` | 读取最近视觉观察 |
| `query_targets` | 可选 `target_types,min_confidence,max_age_ms` | `ok,header,vision_epoch,sequence,snapshot_id,snapshot_age_ms,targets[],human_candidates[],animal_candidates[],object_candidates[],active_target` | 只读目标事实；查询时实时重算年龄，不启动物体流 |
| `detect_objects` | 可选 `confidence,target_labels[]` | `ok,objects[]` | 同步单帧推理，并发布 `source=service`；不启动持续流 |
| `set_object_detection` | `enabled,session_id`；开启时可选 `rate_hz,confidence,target_labels[],lease_sec` | `ok,stream`；关闭时还含 `stopped_session_id` | 开启、续租、更新或关闭唯一物体流 |
| `get_object_detection_state` | `{}` | `ok,stream` | 只读当前 session 状态 |
| `recognize_face` | `{}` | `ok,user_id,confidence,matched` | 对当前最大人脸同步识别 |
| `start_face_enrollment` | 固定 `name`，可选 `required_shots`（默认 3，范围1～5） | `ok,name,step,total_steps,pose,prompt` | 在该身份剩余槽位中新建注册会话；自然面对摄像头，无动作要求 |
| `cancel_face_enrollment` | `{}` | `ok,name,cancelled` | 无会话时失败 |
| `upload_face` | 固定 `name,image_base64` | `ok,name,shots,sample_id,sample_key,image_path` | 解码图片、检测并保存最大人脸，复用最小空闲编号 |
| `list_faces` | `{}` | `ok,faces[]` | 返回注册名称，按字典序排序 |
| `list_face_records` | `{}` | `ok,count,max_faces,max_samples_per_face,allowed_names,available_names,faces[]` | 返回固定身份的样本汇总 |
| `list_face_samples` | `name` | `ok,name,role,shots,sample_ids,samples[]` | 返回一个身份的样本级记录 |
| `get_face_sample` | `name,sample_id` | `ok,name,role,sample_id,image_path,image_url,...` | 查询一张人脸样本元数据 |
| `replace_face_sample` | `name,sample_id,image_base64` | `ok,name,shots,sample_id,replaced` | 校验成功后原位替换并同步识别模板 |
| `delete_face_sample` | `name,sample_id` | `ok,name,shots,deleted_sample_id,remaining_sample_ids,face_removed` | 只删除一张；最后一张删除后释放身份槽位 |
| `delete_face` | `name` | `ok,name` | 删除本地人脸样本并同步内存库 |

不支持的 `task_type` 返回：

```json
{"ok": false, "error": "unsupported task_type: <value>"}
```

### 8.3 人体目标快照查询
相机
支持 `target_types=human/person/animal/object`。唤醒交互等需要多人候选的消费者调用：

```json
{
  "task_type": "query_targets",
  "params_json": "{\"target_types\":[\"human\"],\"min_confidence\":0.3,\"max_age_ms\":300}"
}
```

成功结果字段节选如下；实际 `targets[]`、`human_candidates[]` 与
`active_target` 都返回 4.3 定义的完整对象：

```json
{
  "ok": true,
  "schema_version": 1,
  "header": {
    "stamp": 1786417000.1,
    "frame_id": "camera_color_optical_frame"
  },
  "vision_epoch": "cf31...",
  "sequence": 126,
  "snapshot_id": "cf31...:126",
  "snapshot_age_ms": 23.4,
  "targets": [{
    "vision_epoch": "cf31...",
    "target_id": "cf31...:human:7",
    "target_type": "human",
    "track_id": 7,
    "detection_confidence": 0.91,
    "tracking_state": "tracking",
    "last_seen_age_ms": 42.1,
    "bearing_deg": -3.2,
    "range_valid": true,
    "distance_m": 2.17
  }],
  "human_candidates": [{
    "vision_epoch": "cf31...",
    "target_id": "cf31...:human:7",
    "target_type": "human",
    "track_id": 7,
    "detection_confidence": 0.91,
    "tracking_state": "tracking",
    "last_seen_age_ms": 42.1,
    "bearing_deg": -3.2,
    "range_valid": true,
    "distance_m": 2.17
  }],
  "animal_candidates": [],
  "object_candidates": [],
  "active_target": {"target_id": "cf31...:human:7"}
}
```

对人体而言，`header`、`vision_epoch`、`sequence`、`snapshot_id`、
`human_candidates[]` 和 `active_target` 来自同一份已发布快照。Service 返回前会把
缓存后的单调时间差叠加到 `last_seen_age_ms`；超过 0.35 秒的项转为
`temporarily_lost` 并清除任何缓存距离，因此发布定时器或推理线程卡住时不能冻结
重播一个“仍在跟踪”的人。没有任何视觉快照时返回
`ok=false,error="visual snapshot unavailable"`。

`target_id` 只在同一 `vision_epoch` 内稳定。视觉重启后即使 `track_id` 再次为 1，
完整 ID 也不同；消费者必须同时锁定 `vision_epoch + target_id`。身份名只是属性，
禁止作为运动目标 ID。

`animal` 当前只包括 detector 的精确 `cat/dog` 标签；`object` 包括普通物体，三类
玩具 `dog toy ball/dog frisbee toy/dog tug ring toy` 额外带
`object_kind=toy`。这些 Track 只在既有 `detect_objects` 或租约流成功产出后更新，
`query_targets` 本身绝不启动、续租或抢占物体检测 session。相邻结果按同 label 和
IoU 关联，ID 为 `<vision_epoch>:object:<track_id>`，绝不使用 label 充当 ID。
每个非人体项自带其检测源 `header/source_sequence/source_snapshot_id`；顶层
`header/sequence/snapshot_id` 仍描述人体视觉快照。错误或 stopped 物体包立即清空
非人体 Track，0.75 秒未更新转 `temporarily_lost`，2 秒后删除。

### 8.4 物体流请求

开启或续租：

```json
{
  "enabled": true,
  "session_id": "find-object-001",
  "rate_hz": 2.0,
  "confidence": 0.25,
  "target_labels": ["dog toy ball"],
  "lease_sec": 3.0
}
```

同一 `session_id` 重复调用表示续租，并可更新配置。已有 session 续租时省略
`rate_hz/confidence/target_labels` 会保留原值；`lease_sec` 省略时使用默认 3 秒。

关闭：

```json
{"enabled": false, "session_id": "find-object-001"}
```

关闭成功后发布 `source=control,status=stopped,stop_reason=requested`。动作系统崩溃
或停止续租时，租约到期自动发布 `stop_reason=lease_expired`，避免物体模型永久占用
算力。

## 9. 频率、缓存和降级行为

| 项目 | 正式默认值 | 实际语义 |
|---|---|---|
| 相机输入 | 30 Hz（外部配置） | 相机回调只保留最新候选帧，不阻塞等待推理 |
| 完整视觉推理 | `inference_frame_stride=2` | 对合格帧 1、3、5…推理，30 Hz 输入时上限约 15 Hz |
| Hand 空闲探测 | `hand_idle_inference_stride=2` | 在完整视觉推理候选中再隔次探测；发现手后连续探测 8 次 |
| 视觉 Topic | 10 Hz | 重复发布最近一次仍新鲜的视觉快照 |
| 人体候选 | 最多 5 人 | `max_num_poses=5`；每人独立 `target_id`，业务选择在行为树 |
| 相机过期 | 0.5 秒 | 清空当前场景事实，不刷新旧目标年龄 |
| 对齐深度过期 | 0.5 秒 | 立即回退 `range_valid=false`，不允许框大小估距 |
| 正式物体流 | 0 Hz | 由 Action session 按需开启，默认 2 Hz、最大 5 Hz |
| 物体镜像缓存 | 2 秒 | 只影响 `visual_event.tracked_objects[]`，不替代物体 v2 Topic |

Pose、Hand、YuNet 或 SFace 加载失败时节点可能继续运行，但对应数组/字段会为空或
未知。真实 Provider 不会自动伪造固定人体或物体；只有配置显式使用 `type: mock`
才允许合成数据。

## 10. 消费者检查清单

- Behavior Tree：只消费正式 `/perception/visual_event`，不要依赖
  `/perception/vision/gesture_debug`；对重复 `events[]` 做候选/in-flight 去重。
- Behavior Tree：消费通用 `EVT_VISION_STRANGER`，并独立读取 `/emotion/state`；
  陌生人事实与情绪状态的组合、优先级和去重全部由行为树负责。
- Action：视觉跟踪必须校验 `tracking_state`、新鲜度和有限坐标；失效立即停车。
- 唤醒靠近 Action：只接受同一 `vision_epoch + target_id`、`range_valid=true` 的
  人体目标；任何 epoch/ID/sequence/时间/frame/深度失效必须立即停车。
- 寻物 Action：只接受匹配当前 `session_id` 的 `source=stream,status=ok` v2 数据，
  同时检查 `header.stamp`、本地接收时间和 `sequence`。
- 管理端：人脸注册一次只允许一个活动会话；以 `done=true` 为终态。
- 所有 JSON 消费者：忽略未知字段，但应拒绝不支持的更高 `schema_version`。
- 修改契约时同步更新 `messages/visual_event.py`、`messages/visual_event_types.py`、
  本文、`docs/integration/interface_manifest.yaml`、Viewer 和相关测试。

跨项目完整链路见 [ROS2_INTERFACES.md](integration/ROS2_INTERFACES.md)，部署与
启动见 [HANDOFF.md](HANDOFF.md)。
