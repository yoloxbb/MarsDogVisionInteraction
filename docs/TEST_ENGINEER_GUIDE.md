# MarsDog 视觉测试工程师验收说明

> 适用基线：2026-09-02。本文以当前 `vision.yaml`、
> `vision_debug.launch.py` 和 ROS2 契约为准，面向设备端功能验收、问题复现和
> 测试证据交付。

## 1. 测试范围与重要边界

本项目负责相机输入、人体/人脸/姿态/手势推理、对齐深度融合、人脸库、按需物体
识别、视觉事件发布和调试页面。视觉节点不控制底盘、不调用 Nav2，也不能证明
行为树已选择事件或 Action 已经执行。

测试期间请遵守以下边界：

- 跌倒用例必须在软垫上由辅助人员保护，采用受控的快速躺倒动作，禁止无保护真摔。
- 仅启动视觉与 Viewer 时不会控制底盘；做全链路测试时应另行确认底盘安全措施。
- Debug Web 和人脸 HTTP API 当前都不使用 Token。绑定 `0.0.0.0` 后，仅允许接入
  可信隔离局域网；禁止把 8765、8092 端口暴露到互联网。
- 人脸图片和 `face_registry.json` 是本地生物识别数据。测试结束时是否删除由测试
  负责人确认，不得上传到 Git、工单附件或公共网盘。
- `/perception/visual_event` 是默认 10 Hz 的状态流。同一事件在状态保持期间会重复，
  不等于每个包都是一次新的物理事件。

## 2. 测试前记录表

每次测试开始前先填写以下信息，避免把设备、模型或源码版本差异误判为算法问题。

| 项目 | 必填内容 |
|---|---|
| 测试编号 | 日期-设备-轮次，例如 `20260902-dog01-r1` |
| 设备与系统 | 机器狗编号、RK3588 型号、OS/ROS2 Humble 版本 |
| 源码版本 | `git rev-parse HEAD`；若工作区有修改，附 `git status --short` |
| 实际安装包 | `ros2 pkg prefix --share marsdog_vision_interaction` |
| 配置 | `config_path`、`pose_model_variant`、额外 launch 参数 |
| 模型 | 页面“关键点模型 A/B”中的模型文件；物体模型目录 |
| 相机 | 序列号、彩色/深度分辨率和帧率、安装位置 |
| 环境 | 距离、光照、背景、人数、是否戴口罩/眼镜 |
| 时间范围 | 开始/结束时间，精确到秒 |
| 证据目录 | 终端日志、Topic 原文、事件导出 JSON、截图/录像的位置 |

建议每轮新建一个证据目录，并把启动日志完整保留：

```bash
export TEST_RUN_ID="$(date +%Y%m%d-%H%M%S)-vision"
mkdir -p "test-evidence/$TEST_RUN_ID"

ros2 launch marsdog_vision_interaction vision_debug.launch.py \
  web_host:=0.0.0.0 2>&1 | tee "test-evidence/$TEST_RUN_ID/vision-launch.log"
```

## 3. 启动与基础健康检查

### 3.1 启动顺序

1. 先启动 RealSense，至少提供彩色图、对齐深度和 CameraInfo。
2. 再用统一入口同时启动正式视觉节点与 Viewer。
3. 浏览器打开 `http://<机器狗IP>:8765`。

```bash
ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true enable_depth:=true align_depth.enable:=true

ros2 launch marsdog_vision_interaction vision_debug.launch.py \
  web_host:=0.0.0.0
```

`vision_debug.launch.py` 默认 `start_vision_node:=true`。不要同时再启动
`vision.launch.py`。设备已经存在正式视觉节点时，只附加 Viewer：

```bash
ros2 launch marsdog_vision_interaction vision_debug.launch.py \
  web_host:=0.0.0.0 start_vision_node:=false
```

### 3.2 五分钟冒烟表

| 检查项 | 方法 | 通过标准 | 失败时先看 |
|---|---|---|---|
| 节点 | `ros2 node list` | 有 `/vision_interaction`、`/vision_debug_viewer`，且各只有一个 | 是否重复启动、是否 source 了旧工作区 |
| 相机 | `ros2 topic hz /camera/camera/color/image_raw` | 持续有帧；页面相机 Badge 为绿色并显示毫秒年龄 | RealSense 进程、Topic 名、USB |
| 深度 | `ros2 topic hz /camera/camera/aligned_depth_to_color/image_raw` | 持续有帧 | `enable_depth`、`align_depth.enable` |
| 内参 | `ros2 topic echo /camera/camera/color/camera_info --once` | `width/height/k/header.frame_id` 非空且与彩色流匹配 | CameraInfo Topic/相机 profile |
| 视觉状态流 | `ros2 topic hz /perception/visual_event` | 约 10 Hz | `Vision node ready`、模型/相机错误 |
| Service | `ros2 service type /perception/vision/task` | `marsdog_vision_interaction/srv/VisionTask` | 安装包前缀和接口是否为旧版 |
| Viewer | 浏览器访问 8765 | 实时画面可见；相机、视觉、动作 Badge 持续刷新 | `Web dashboard`/端口占用/防火墙 |
| 人脸 API | `curl http://127.0.0.1:8092/health` | `{"ok":true,...}` | `Face FastAPI ready` 或端口占用 |

启动成功时应能看到这些关键日志：

```text
YuNet loaded: ...
MediaPipe PoseLandmarker loaded: variant=... mode=... model=...
MediaPipe HandLandmarker loaded: mode=... model=...
ByteTrack face tracker initialized
SFace recognition throttle initialized
VisionObservationProvider started — <已加载>/<总数> models loaded; ...
Vision node ready: camera=... visual=... objects=... startup=0.00Hz depth=... service=...
Face FastAPI ready: http://127.0.0.1:8092/docs
Web dashboard: http://127.0.0.1:8765
Viewer ready; render<=8.0 FPS scale=0.75 publish_debug_image=False; ...
```

出现 `YuNet failed`、`MediaPipe ... failed`、`SFace recognizer init failed`、
`VisionObservationProvider — no models, unavailable` 或 `Configured vision provider is
unavailable` 时，本轮真实模型测试不得判为通过。

## 4. 模块验收总表

| ID | 模块 | 核心测试 | 页面观察点 | ROS/日志证据 | 主要通过标准 |
|---|---|---|---|---|---|
| CAM-01 | 彩色相机 | 有人、空场、遮挡、拔插 | 画面、相机 Badge、FPS | 相机 `hz`；`Camera frame rejected` | 有流时持续更新；断流后不继续发布旧场景事实 |
| DEP-01 | 深度融合 | 0.5～3 m 多距离；停深度流 | 当前目标/原始 JSON | `range_valid/distance_m/range_source/depth_sync_delta_ms` | 有效时为对齐深度；异常后 0.5 s 内 fail closed |
| HUM-01 | 人体/多人 | 单人、两人、进出画面、短暂遮挡 | 人体框、骨架、当前目标、人体表 | `human_candidates[]/active_target` | 最多 5 人；Track 稳定；过期状态正确 |
| FACE-01 | 人脸在线录入 | 5 个固定身份分别录入 | 录入进度、名单 | `enrollment_event`、HTTP/Service 返回 | 默认连续 3 张；每人最多 5 张；不要求动作 |
| FACE-02 | 人脸样本 CRUD | 增、查、图像、替换、删除 | 名单刷新、现场识别 | HTTP 状态码与 JSON | `sample_id` 稳定；删除不重排；空位复用 |
| FACE-03 | 人脸识别 | 已登记人、陌生人、遮脸、多人 | 人脸表、当前目标 | `Face identity changed`、身份字段 | 已知身份需确认；陌生人不误报固定身份 |
| GATE-01 | 姿态事件身份门控 | 同一动作由已登记人/陌生人执行 | 姿态门控、事件历史 | `Visual state changed` | 仅固定身份 + `confirmed_known` + `tracking` 上报姿态事件 |
| POS-01 | 姿态与动作 | 按动作清单逐项表演 | GesturePose 分数、人体表 | `gesture_debug` | 原始候选可解释；仅稳定命中写兼容动作/事件 |
| STOP-01 | 停止手势 | 正例、挥手、指点、抱臂反例 | 手部特征、黄色事件记录 | `hands[]`、`EVT_VISION_STOP_GESTURE` | 3/5 帧稳定；反例不触发；身份门控有效 |
| FALL-01 | 跌倒 | 直立布防→受控快速躺倒；静态躺卧 | 跌倒状态、红色事件记录 | `Fall event confirmed`、`fall_detector` | 真转换触发一次边沿；静态躺卧不触发；30 s 冷却 |
| STRANGER-01 | 陌生人事件边界 | 陌生人、已知人；情绪状态变化 | 视觉事件和历史 | 节点订阅、`events[]` | 陌生人始终只发 Stranger；视觉不订阅情绪；组合判断属于行为树 |
| EVT-01 | 视觉事件流 | 人/动作进入、保持、离开 | ENTER/ACTIVE/EXIT | `vision_epoch/sequence/events[]` | Topic 约 10 Hz；生命周期正确压缩；序号递增 |
| OBJ-01 | 单次物体识别 | 页面点击一次，目标有/无 | 物体 Badge、紫框、耗时 | object Topic `source=service` | 单次不创建 session；空数组也是正常结果 |
| OBJ-02 | 持续物体识别 | 启动、续租、停止、关页超时 | session、频率、停止原因 | `source/status/stream/stop_reason` | 固定 web session；不抢占 Action；租约终止清缓存 |
| SRV-01 | VisionTask | 查询人物、目标、物体状态 | 页面并非唯一证据 | Service response 与 `latency_ms` | 回传 task ID/type；结果 JSON 可解析；失败原因明确 |
| UI-01 | 调试页面 | 桌面滚动、窄屏、导出/清空 | 左侧画面固定、右侧独立滚动 | Viewer 日志、导出 JSON | 桌面看日志时画面仍可见；窄屏正常上下排列 |
| PERF-01 | 性能/A-B | 相同场景各跑 ≥150 推理帧 | 关键点模型 A/B | `landmarker` | 数据量足够；记录 FPS、平均/P95 和检测率，不凭主观感受 |
| REC-01 | 故障恢复 | 相机/深度/模型/Service 异常 | Badge、错误区 | error/warning 和状态字段 | 不伪造识别；旧目标/物体及时失效；恢复后继续更新 |

## 5. 分模块详细测试方法

### 5.1 相机、人体跟踪与多人候选

| 用例 | 操作步骤 | 关键字段 | 通过标准 |
|---|---|---|---|
| 单人 | 一人从画面外进入，站立 10 s，再离开 | `active_target.track_id/tracking_state/last_seen_age_ms/bbox/body_center/confidence` | 进入后为 `tracking`；框和骨架跟随；离开后不无限保留旧目标 |
| 遮挡 | 当前人被遮挡约 0.5 s 后重新出现 | `track_id/target_id/vision_epoch/tracking_state` | 短时可进入 `temporarily_lost`；恢复后尽量延续稳定 ID，不应跳到无关人 |
| 两人 | 两人同时进入、交叉、分别离开 | `human_candidates[]`、每项 `track_id/target_id` | 同时输出两个候选且 ID 不同；`active_target` 只是一名兼容主目标 |
| 五人上限 | 逐步增加人数 | `human_candidates[]` 长度 | 不超过 5；字段结构完整；页面不阻塞 |
| 相机断流 | 停止 RealSense 彩色流 >0.5 s | 页面 Badge；`visual_event` 内容 | 不继续拿旧画面生成新场景事实，目标自然丢失；恢复相机后继续更新 |

`active_target` 和 `human_candidates[]` 的合格目标至少检查：

```text
vision_epoch 非空
target_id 非空
track_id > 0
tracking_state == "tracking"
last_seen_age_ms 为合理非负值
bbox/body_center 是有限的 [0,1] 归一化值
```

注意：`active_target` 是兼容单目标，并不表示行为树最终选中了这个人。多人选择策略
应在行为树测试中另行验证。

### 5.2 对齐深度与距离

1. 让已跟踪人体分别站在约 0.5 m、1 m、2 m、3 m 位置，每个位置稳定 5 s。
2. 从原始 JSON 保存 `distance_m`、`pose_3d` 和同步字段，同时记录测量真值。
3. 停止对齐深度 Topic，但保留彩色图；等待超过 0.5 s。
4. 再恢复深度流，观察距离恢复。

| 字段 | 含义与判断 |
|---|---|
| `range_valid` | 只有对齐深度、内参、frame、时间同步和 ROI 都有效时才为 `true` |
| `distance_m` | 米制距离；仅当 `range_valid=true` 才可使用 |
| `range_source` | 有效时应为 `aligned_depth`，失败时为 `none` |
| `depth_sync_delta_ms` | 彩色推理帧与所选深度帧时间差；当前上限 100 ms |
| `pose_3d.valid/x/y/z/frame_id` | 有效时给出相机坐标三维点 |

通过标准：异常后 0.5 s 内所有目标变成
`range_valid=false,distance_m=null`；不得用人体框大小伪造距离。物体候选也遵循同一
fail-closed 原则。

### 5.3 在线人脸录入

页面选择 `owner` 或 `family_member_1`～`family_member_4`，点击“开始连续录入”，
自然正对相机并保持稳定。默认录 3 张；若只剩 1～2 个空位，只补满剩余槽位。

| 阶段/检查 | 页面或 Topic 字段 | 通过标准 |
|---|---|---|
| 搜索 | `status=searching, step,total_steps,prompt,done=false` | 未发现合格人脸时不保存图片 |
| 稳定 | `status=tracking, confidence,progress_pct` | 进度连续增加；移动/质量下降后重新等待 |
| 单张完成 | `status=captured,shots` | 只增加一张并继续下一张 |
| 全部完成 | `status=done,shots,done=true` | 只发布一次完成状态并结束会话 |
| 容量 | 页面名单或 `GET /api/v1/faces` | 固定 5 个身份，每个身份最多 5 张 |

当前质量拒绝提示及对应操作：

| 提示 | 原因/处理 |
|---|---|
| `人脸检测置信度不足，请调整光线` | 检测置信度低于 0.85；调整光线和正脸角度 |
| `人脸太小，请靠近摄像头` | 人脸短边小于 80 px；靠近相机 |
| `人脸区域无效，请重新站位` | 裁剪区域无效；回到画面中央 |
| `画面过暗，请增加正面光线` | 灰度均值低于 35 |
| `画面过曝，请避开强光` | 灰度均值高于 225 |
| `画面模糊，请保持头部稳定` | 清晰度不足；停止移动并保证对焦 |

`/perception/vision/enrollment_event` 当前没有 `schema_version/header/task_id`，只能
关联当前唯一活动会话，不能按它判断并发注册。

### 5.4 人脸样本 CRUD

Swagger：`http://127.0.0.1:8092/docs`。当前不需要 Authorization、Token 或 URL
参数。建议用专门测试身份和可删除的测试图片执行，避免误删正式主人数据。

```bash
# 健康检查和人员汇总
curl -sS http://127.0.0.1:8092/health
curl -sS http://127.0.0.1:8092/api/v1/faces

# 新增一张
curl -sS -X POST \
  http://127.0.0.1:8092/api/v1/faces/family_member_4/samples \
  -F "image=@./face-a.jpg"

# 查询列表、单张元数据和图片
curl -sS http://127.0.0.1:8092/api/v1/faces/family_member_4/samples
curl -sS http://127.0.0.1:8092/api/v1/faces/family_member_4/samples/1
curl -sS -o family_member_4-001.jpg \
  http://127.0.0.1:8092/api/v1/faces/family_member_4/samples/1/image

# 原位替换和删除单张
curl -sS -X PUT \
  http://127.0.0.1:8092/api/v1/faces/family_member_4/samples/1 \
  -F "image=@./face-b.png"
curl -sS -X DELETE \
  http://127.0.0.1:8092/api/v1/faces/family_member_4/samples/1
```

| 检查 | 预期 |
|---|---|
| 新增 | HTTP 201；返回 `request_id,name,shots,sample_id,sample_key,image_path` |
| 查询 | 返回 `role,shots,sample_ids,samples[]`，图片响应为 `image/jpeg` |
| 替换 | `replaced=true` 且 `sample_id` 不变 |
| 删除中间编号 | 其他编号不重排；`remaining_sample_ids` 保持稳定 |
| 再次新增 | 复用最小空闲 `sample_id` |
| 删除最后一张 | `face_removed=true`，该身份释放 |
| 第 6 张 | HTTP 409，错误码 `face_sample_limit_reached` |
| 非固定身份 | HTTP 422；不得创建自由姓名 |
| 非 JPG/JPEG/PNG | HTTP 415；空文件 400；超过 10 MiB 为 413 |

### 5.5 人脸识别与姿态事件门控

人脸状态变化时重点看：

```text
Face identity changed: track=<face_track_id> identity=<name> state=<state> confidence=<score> attempts=<n>
```

当前确认规则为连续两次已知匹配进入 `confirmed_known`，连续四次未知证据确认
unknown。第一次匹配可能只是 `candidate_known`，测试时不得把它当作已确认身份。

门控矩阵必须完整执行：

| 当前目标 | 页面姿态/原始分 | 正式 `events[]` | `Visual state changed` |
|---|---|---|---|
| 固定身份 + `confirmed_known` + `tracking` | 可见 | 应发布对应姿态/手势事件 | `pose_event_gate=open` |
| 固定身份 + `candidate_known` | 可见 | 不发布姿态/手势事件 | `pose_event_gate=blocked` |
| 陌生人/unknown | 可见 | 不发布普通姿态、跌倒、Stop 事件 | `pose_event_gate=blocked` |
| 未检测到人脸 | 可见或保留诊断 | 不发布姿态/手势事件 | `blocked` 或 `idle` |
| 已知身份但目标非 `tracking` | 可保留 | 不发布姿态/手势事件 | `pose_event_gate=blocked` |

对应状态变化日志：

```text
Visual state changed: track=<id> tracking=<state> identity=<name> identity_state=<state> pose=<compat_action> hands=<actions> pose_event_gate=<open|blocked|idle> events=<events>
```

该日志按状态签名变化去重，不会跟随 10 Hz Topic 刷屏。缺少日志不代表 Topic 停止，
需要同时检查 `/perception/visual_event`。

### 5.6 姿态与动作识别

测试时先看 `raw_scores[]` 是否出现候选，再看 `recognized_actions[]` 是否经过时序
平滑稳定命中，最后看兼容动作和正式事件。禁止用单帧 `raw_scores` 直接判机器人
应该执行动作。

| 精确动作组 | 代表动作 | 兼容字段/事件方向 |
|---|---|---|
| P0 安全 | `fall`、`stop_gesture` | `fallen_down`/`stop_gesture`；独立安全事件 |
| P1 明显负向 | `hands_on_hips`、`large_arm_swing`、`pointing`、`stomping`、`arms_crossed` | `EVT_VISION_MASTER_SAD` |
| P2 消沉/防御 | `head_down`、`shoulders_slumped`、`face_covering`、`hands_on_head`、`curled_up`、`hunched` | `EVT_VISION_MASTER_SAD` |
| P3 积极互动 | `arms_raised`、`waving`、`victory`、`jumping`、`arms_open`、`fast_nod`、`clapping`、`thumbs_up` | `EVT_VISION_MASTER_HAPPY` |
| P4 普通状态 | `standing`、`sitting`、`lying`、`low_motion` | 兼容普通状态；稳定站/坐/低运动映射 Neutral，静态 lying 不是跌倒 |

每个动作至少保存：站位录像、`primary_action/primary_priority`、对应
`raw_scores[]` 条目、`recognized_actions[]` 条目、`support_ratio/duration_s`、兼容
动作、最终 `events[]`。正例和最相似反例各做至少 3 次。

### 5.7 Stop 手势

正例：正对相机，单手四根非拇指伸直，掌心朝镜头，手部保持稳定。高位俯视安装下，
掌面明显朝前或手臂接近伸直可提供几何证据；Pose/Hand 手腕关联仍是硬条件。

| 观察字段 | 预期 |
|---|---|
| `hand_features.left/right.four_fingers_extended` | 正例应达到 `4` |
| `palm_facing_score` | 正例明显升高，结合页面原始分判断 |
| `motion_energy` | 保持动作时应下降，避免把挥手当 Stop |
| `raw_scores[name=stop_gesture].score` | 达到规则阈值后成为候选；当前阈值 0.65 |
| `recognized_actions[name=stop_gesture]` | 5 帧窗口中至少 3 帧支持后出现 |
| `hands[].hand_action` | 稳定后为 `stop_gesture` |
| `events[]` | 仅已确认固定身份出现 `EVT_VISION_STOP_GESTURE` |

反例必须包括：快速挥手、只伸食指、握拳、侧掌、抱臂。抱臂分数明显成立时 Stop
候选应被冲突规则清除。事件历史中 Stop 用黄色边线显示。

### 5.8 跌倒

建议步骤：

1. 已登记人员正对相机并确认页面门控为 `open`。
2. 保持稳定直立至少 1 s，确认 `fall_detector.armed=true`。
3. 在保护下于约 1.5 s 内快速变为躺卧，并持续躺卧至少 0.8 s。
4. 恢复站立，等待状态进入 `recovering/monitoring`。
5. 30 s 冷却内重复一次，再在冷却结束后重复一次。
6. 另做“进入画面时已经躺着”和“缓慢主动躺下”两个反例。

| 字段/日志 | 含义 |
|---|---|
| `fall_detector.phase` | `unknown/monitoring/falling/lying/recovering` |
| `armed` | 已建立直立基线且不在冷却 |
| `lying_score` | 当前躺卧证据，阈值 0.55 |
| `transition_score` | 快速下落/转倒证据，阈值 0.55 |
| `event_triggered` | 跌倒确认帧的一次边沿 |
| `alert_active` | 确认后约 2 s 的保持状态 |
| `cooldown_remaining_s` | 同一次跌倒的 30 s 重新触发冷却 |
| `Fall event confirmed (transition=..., lying=...)` | 状态机确认边沿；一次受控跌倒应只计这条确认日志一次 |

`EVT_VISION_FALL` 可能因 10 Hz 状态保持出现在多个包中，不能据包数统计跌倒次数。
测试事件次数应数 `Fall event confirmed` 或页面 `ENTER`。陌生人即使状态机确认跌倒，
也只保留调试诊断，不得发布正式跌倒事件。

### 5.9 陌生人事件与下游融合边界

视觉只负责判断人脸是否属于固定人脸库。陌生人出现时，无论机器人当前情绪如何，
视觉端都只发布 `EVT_VISION_STRANGER`；`vision_interaction` 不订阅
`/emotion/state`。情绪与陌生人事实的组合由下游行为树完成。

先检查运行时订阅，输出中不得出现 `/emotion/state`：

```bash
ros2 node info /vision_interaction
```

| 情况 | Vision `events[]` 预期 | 验收重点 |
|---|---|---|
| 未录入人员持续在画面中 | 只含 `EVT_VISION_STRANGER` 人脸事件 | 10 Hz 状态流可重复，但不得改名 |
| 外部 Anxiety/Fear 状态变化 | 仍为 `EVT_VISION_STRANGER` | 视觉不读取情绪，也不产生 Alert 细分 |
| 外部 Joy/Excite/Calm 状态变化 | 仍为 `EVT_VISION_STRANGER` | 视觉不产生 Friend 细分 |
| `/emotion/state` 停止或 JSON 非法 | 仍为 `EVT_VISION_STRANGER` | 视觉结果不依赖情绪节点可用性 |
| 已确认固定身份 | `EVT_VISION_MASTER` | 不得误报 Stranger |

本项目的通过标准是：视觉事件中永远不出现
`EVT_VISION_STRANGER_ALERT/EVT_VISION_STRANGER_FRIEND`，并且视觉节点没有情绪
Topic 订阅。若测试产品最终的 Alert/Friend 行为，必须在行为树项目中关联同一时间段
的 `EVT_VISION_STRANGER`、`/emotion/state`、候选仲裁日志和 Action Result；该结果
不能记为 Vision 单模块通过证据。

### 5.10 视觉事件历史与发布频率

```bash
ros2 topic hz /perception/visual_event
ros2 topic echo /perception/visual_event --once
```

| 检查点 | 预期 |
|---|---|
| 频率 | 当前配置约 10 Hz；这是状态发布频率 |
| `schema_version` | `1` |
| `vision_epoch` | 同一视觉进程内稳定；重启后改变 |
| `sequence` | 同一 epoch 内严格递增 |
| `header.stamp/frame_id` | 对应视觉观察来源时间与相机 frame |
| `events[]` | 当前成立的状态事件，可跨多个包重复 |
| 页面 `ENTER` | 事件首次出现 |
| 页面 `ACTIVE` | 同一事件持续，含 `repeat_count/duration_ms` |
| 页面 `EXIT` | 事件消失；`reason=event_cleared` 或 `vision_epoch_changed` |

点击“导出 JSON”会生成 `marsdog-vision-events-<时间>.json`，内容含
`active_events` 和 `history`。历史记录重点字段为：
`phase,event,received_at,source_stamp,vision_epoch,sequence,evidence,repeat_count,
duration_ms,reason`。默认仅保留 Viewer 进程内最近 200 条，重启 Viewer 或点击清空
都会丢失，因此问题发生后应立即导出。

该历史只能证明 Vision 已发布：

```text
Vision events[] -> Viewer ENTER/ACTIVE/EXIT
```

若测试目标是完整执行链，必须再把行为树候选/仲裁日志和 Action Result 按时间关联，
不能用 Viewer 记录代替。

### 5.11 单次与持续物体识别

正式启动时物体推理频率为 0 Hz，只有手动按钮或任务会话才加载/运行模型。首次调用
包含 RKNN 懒加载，允许明显慢于后续调用，必须单独记录首次和稳态耗时。

单次测试：设置置信度，点击“单次物体识别”。连续测试：设置置信度和 0.1～5 Hz
频率，点击“启动持续识别”，保持 40 s 后停止；另做关闭浏览器不点停止的测试。

| 字段 | 单次预期 | 持续预期 |
|---|---|---|
| `schema_version` | 2 | 2 |
| `source` | `service` | 推理包为 `stream`；停止包为 `control` |
| `status` | `ok` 或 `error` | `ok/error/stopped` |
| `stream.active` | 不因单次创建 session | 启动后 `true` |
| `stream.session_id` | 空或现有会话状态 | 页面固定 `vision-debug-web` |
| `stream.rate_hz/confidence/target_labels` | 本次请求见 `request` | 应与页面/会话参数一致 |
| `stream.lease_remaining_sec` | 不适用 | 页面每 10 s 续租 30 s；关页后最多 30 s 到期 |
| `inference_latency_ms` | 单次模型耗时 | 每次持续推理耗时 |
| `objects[]` | 有目标时含稳定 ID/框/置信度；空数组表示有效未检出 | 同左 |
| `stop_reason` | 空 | 手动为 `requested`，超时为 `lease_expired` |

对应日志：

```text
Object detection stream active: session=<id> rate=<hz>Hz targets=<labels>
Object detection lease expired: session=<id>
```

再验证会话互斥：先用另一个 `session_id` 模拟 Action 占用，再从页面启动。页面应显示
占用错误，不得抢占或停止别人的 session。停止/错误包必须清空旧物体缓存。

### 5.12 VisionTask Service

通用传输字段：

| 请求/响应字段 | 含义 |
|---|---|
| `task_id` | 调用方生成的关联 ID；响应必须原样回传 |
| `task_type` | 任务名；响应必须原样回传 |
| `params_json` | JSON object 字符串 |
| `success` | 来自业务结果中的 `ok` |
| `result_json` | 可解析 JSON object；业务结果和详细字段 |
| `error_message` | 失败原因，与结果 `error` 对齐 |
| `latency_ms` | 整个 Service 回调耗时；同步推理包含模型时间 |

基础查询示例：

```bash
ros2 service call /perception/vision/task \
  marsdog_vision_interaction/srv/VisionTask \
  "{task_id: 'qa-query-001', task_type: 'query_targets', \
params_json: '{\"target_types\":[\"human\"],\"min_confidence\":0.3,\"max_age_ms\":500}'}"

ros2 service call /perception/vision/task \
  marsdog_vision_interaction/srv/VisionTask \
  "{task_id: 'qa-object-state-001', task_type: 'get_object_detection_state', \
params_json: '{}'}"
```

当前任务清单：`check_person`、`query_targets`、`detect_objects`、
`set_object_detection`、`get_object_detection_state`、`recognize_face`、
`start_face_enrollment`、`cancel_face_enrollment`、`upload_face`、`list_faces`、
`list_face_records`、`list_face_samples`、`get_face_sample`、
`replace_face_sample`、`delete_face_sample`、`delete_face`。

至少验证：合法请求、非法 JSON、不支持任务名、Service 不可用、模型不可用、相机
过期。所有失败必须 `success=false` 且有可定位原因，不能返回伪造成功数据。

### 5.13 页面布局与性能面板

桌面宽度大于 1050 px 时，滚动右侧事件/诊断区，左侧实时画面应固定可见；窗口变窄
后应恢复上下布局，不遮挡内容。验证复制、导出、清空事件历史均不影响 ROS Topic。

Lite/Full A/B 必须在相同相机、站位、光照、人物和动作下，各保持至少 150 个推理
帧，记录：

| 字段 | 含义 |
|---|---|
| `received_frames/inference_candidates/inferred_frames` | 收帧、进入候选和实际完成数量 |
| `replaced_pending_frames` | 来不及处理而被新帧替换的旧候选数；这是低延迟策略，不是队列积压 |
| `effective_inference_fps` | 有效完整推理频率 |
| `pipeline_avg_ms/pipeline_p95_ms` | 整条流水线平均/P95 |
| `landmarker.pose.avg_ms/p95_ms` | Pose 耗时 |
| `detection_rate` | 有人的固定场景下才有比较意义 |
| `keypoint_valid_ratio/critical_keypoint_valid_ratio` | 33 点与动作关键点有效率 |
| `landmarker.hand.*` | Hand 耗时、有效 FPS、空闲步长、检测率 |
| `feature_ms/recognition_ms` | 特征提取和规则判断耗时 |

验收报告同时列 Lite 和 Full 原始数值。是否切换模型应依据产品要求的最低动作召回
与有效 FPS，不应只凭单次观感。

## 6. 关键日志与字段速查

### 6.1 控制台日志

| 日志前缀 | 关键字段 | 用途 |
|---|---|---|
| `Vision node ready` | camera、visual、objects、startup、depth、service | 证明实际接线和物体启动频率 |
| `VisionObservationProvider started` | 模型加载数、stride、最大 Pose 数、Hand idle stride | 证明真实推理 Provider 已启动 |
| `Face identity changed` | track、identity、state、confidence、attempts | 身份状态机变化证据 |
| `Visual state changed` | track、tracking、identity、identity_state、pose、hands、pose_event_gate、events | 身份门控和最终事件证据 |
| `Fall event confirmed` | transition、lying | 一次跌倒确认边沿 |
| `Object detection stream active` | session、rate、targets | 持续物体会话启动/续租证据 |
| `Object detection lease expired` | session | 未续租自动终止证据 |
| `Face FastAPI ready/unavailable` | docs 地址或异常 | HTTP 管理面可用性 |
| `Web dashboard` / `Viewer ready` | 地址、渲染 FPS、缩放、Debug Topic | Viewer 启动参数 |
| `Visual observation failed` | 异常 | 视觉观察失败 |
| `Camera frame rejected` / `Depth frame rejected` / `CameraInfo rejected` | 编码或解析原因 | 输入数据不合格定位 |
| `Ultralytics ... failed` / `object model unavailable` | 模型异常 | 物体加载或推理失败 |

### 6.2 `/perception/visual_event` 重点字段

| 区域 | 必查字段 |
|---|---|
| 包级关联 | `schema_version,header,vision_epoch,sequence,snapshot_id` |
| 主目标 | `active_target.target_id,track_id,tracking_state,last_seen_age_ms,confidence` |
| 身份 | `identity,identity_state,identity_confidence,is_registered,face_track_id` |
| 几何 | `bbox,body_center,face_center,bearing_deg,range_valid,distance_m,pose_3d` |
| 姿态 | `pose_state,pose_action,pose_action_label,keypoints[]` |
| 多人 | `human_candidates[]`，每项字段与主目标保持同一语义 |
| 人脸/手/物体 | `faces[],hands[],tracked_objects[]` |
| 正式事件 | `events[]` |

### 6.3 `/perception/vision/gesture_debug` 重点字段

| 区域 | 必查字段 |
|---|---|
| 关联 | `schema_version,stamp,track_id,tracking_state,pose_state` |
| 兼容输出 | `legacy_pose_action,legacy_hand_actions[]` |
| 规则结果 | `primary_action,primary_priority,state_hint,recognized_actions[],raw_scores[]` |
| 跌倒 | `fall_phase,fall_event_triggered,fall_alert_active,fall_detector` |
| 手势 | `hand_features.left/right` |
| 时序 | `temporal_features`，包括头部运动、反转、位移和双腕开合 |
| 耗时 | `feature_ms,recognition_ms,landmarker` |

## 7. 故障定位表

| 现象 | 优先判断 | 证据/处理 |
|---|---|---|
| 页面打不开 | Viewer 未启动、端口占用或网络不通 | 找 `Web dashboard`；本机 `curl http://127.0.0.1:8765/api/state` |
| 页面有画面但无识别 | 模型没加载或视觉 Topic 过期 | 看三类模型日志、视觉 Badge、`gesture_debug` |
| 页面完全无画面 | 彩色 Topic 名/相机异常 | `ros2 topic hz`、`ros2 topic info -v`、相机日志 |
| 姿态可见但没有事件 | 身份门控未打开或时序未稳定 | `identity_state/tracking_state/pose_event_gate/raw_scores/recognized_actions` |
| 已知人偶尔变陌生人 | 人脸质量、尺寸、阈值或 Track 变化 | `Face identity changed` 的 confidence/attempts；保存对应帧条件 |
| lying 不触发跌倒 | 这是预期，静态 lying 缺少转换 | 看 `armed/transition_score/event_triggered`，不要直接降阈值 |
| Stop 被识别成挥手 | 手在移动、掌向/伸指不足或未稳定 | `four_fingers_extended/palm_facing_score/motion_energy/support_ratio` |
| 物体按钮长时间等待 | 首次 RKNN 懒加载或模型失败 | 页面耗时、`Ultralytics YOLOE loaded/failed`；首次与稳态分开记录 |
| 持续物体被拒绝 | 另一 session 已占用 | 查 `get_object_detection_state`；不得从页面抢占 |
| 有框但 `range_valid=false` | 深度/内参/frame/同步/ROI 任一不合格 | 查三个相机 Topic 与 `depth_sync_delta_ms/range_source` |
| 日志没有每帧事件 | `Visual state changed` 本来就按状态去重 | 用 `ros2 topic hz/echo` 和页面事件历史证明状态流 |
| Service 类型不存在/字段不对 | source 了旧安装包 | `ros2 pkg prefix --share`、`ros2 interface show .../srv/VisionTask` 后重建/source |

## 8. 单用例结果表与最终报告模板

每个用例至少留一行，不允许只写“页面正常”。

| 字段 | 填写要求 |
|---|---|
| Case ID/名称 | 对应本文 ID，加动作或场景后缀 |
| 前置条件 | 身份、人数、距离、光照、物体 session 等 |
| 操作时间 | 开始与结束时间 |
| 输入动作 | 测试人员实际操作，必要时附视频时间码 |
| 页面结果 | 面板字段、Badge、颜色、截图编号 |
| ROS 结果 | Topic、epoch/sequence、event、task_id、status |
| 日志结果 | 精确日志前缀和关键字段，不只写“有日志” |
| 耗时 | Service `latency_ms`、物体 `inference_latency_ms`、事件 ENTER 时间差 |
| 期望/实际 | 分栏填写 |
| 结论 | PASS / FAIL / BLOCKED；BLOCKED 必须写阻塞原因 |
| 证据 | 日志文件、事件 JSON、截图、录像的相对路径 |

最终报告建议汇总：

```text
测试基线：commit / installed prefix / config / model
测试环境：设备 / 相机 / 光照 / 距离 / 网络
执行统计：计划、通过、失败、阻塞、未执行
关键性能：相机 Hz、visual_event Hz、Lite/Full FPS 与 P95、物体首次/稳态耗时
安全用例：Stop 正反例、跌倒正反例、身份门控、深度 fail-closed
数据用例：5 人 x 5 张限制、样本 CRUD、删除后空位复用
缺陷清单：缺陷号、Case ID、首次异常时间、epoch/sequence/task_id、证据路径
遗留风险：硬件未覆盖、模型未加载、下游 Tree/Action 尚未联调等
```

测试工程师交付的最小证据包应包含：完整 launch 日志、相机和视觉 Topic 频率、至少
一个原始 visual JSON、GesturePose JSON、物体 schema v2 包、人脸 CRUD 响应、页面
事件导出 JSON，以及所有失败用例前后 10 秒的录像或截图。
