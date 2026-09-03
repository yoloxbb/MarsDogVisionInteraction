# MarsDog 视觉测试工程师验收说明

> 适用基线：2026-09-03。本文以当前 `vision.yaml`、
> `vision_debug.launch.py` 和 [ROS2_CONTRACT.md](ROS2_CONTRACT.md) 为准，面向设备端
> 功能验收、问题复现和测试证据交付。

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
  web_host:=0.0.0.0 \
  test_run_id:="$TEST_RUN_ID" \
  test_case_id:="SMOKE-01" \
  log_dir:="test-evidence/$TEST_RUN_ID" 2>&1 | \
  tee "test-evidence/$TEST_RUN_ID/vision-launch.log"
```

`logging.event_trace=true` 时，节点同时生成
`vision_trace_<时间>_<pid>.jsonl`。其中每行均以 `VISION_TRACE ` 开头，后接单行
JSON；`run_id/case_id` 来自上述 launch 参数，也可分别使用环境变量
`MARSDOG_TEST_RUN_ID/MARSDOG_TEST_CASE_ID` 注入。

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
| JUMP-01 | 跳跃 | 全身/缺脚踝原地跳；快速起立、踮脚、走路反例 | 识别通道、起跳/回落证据、肩/髋/脚速度 | `jump_detector`、`recognized_actions` | 全身两帧共同上升；半身要求肩髋上升后回落；身份门控有效 |
| STOMP-01 | 跺脚 | 全身/缺脚踝跺脚；抬腿保持、跳跃、走路反例 | 脚踝/膝部通道、速度、幅度、换向 | `stomp_detector`、`recognized_actions` | 完成局部抬落周期；身体中心稳定；3/5帧确认；身份门控有效 |
| HOLD-01 | 手持玩具/狗粮 | 手持正例；地面、桌面、远离手腕反例 | 手持状态和关联证据 | `active_target.held_object`、TOY/FOOD事件 | 两个物体结果确认；只关联当前人；身份门控有效 |
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

#### 5.6.1 一次动作必须按五层结果对齐

视觉不像语音指令那样“一条输入对应一个独立事件”。测试人员必须从精确识别结果
逐层检查到正式事件，不能只看到 Happy/Sad 就判定具体动作正确：

| 层级 | 字段 | 含义 | 能证明什么 |
|---|---|---|---|
| 1. 原始候选 | `raw_scores[name=<精确名>].score` | 当前帧规则分数，尚未经过时序确认 | 只能定位规则是否接近成立，不能判成功 |
| 2. 稳定识别 | `recognized_actions[name=<精确名>]` | 经过动作自己的窗口、支持率和滞回后成立 | 证明 GesturePose 确实识别了该精确动作 |
| 3. 主动作 | `primary_action/primary_priority` | 同帧多个稳定动作中按 P0～P4 选出的最高优先动作 | 证明本帧展示主结果；不能替代完整 `recognized_actions[]` |
| 4. 兼容输出 | `legacy_pose_action/legacy_hand_actions[]`；正式包中的 `active_target.pose_action/hands[].hand_action` | 折叠给既有下游的公开动作名称 | 证明精确动作已正确折叠；多个精确动作可能共用同一个值 |
| 5. 正式事件 | `/perception/visual_event.events[]` 和 `VISION_TRACE event_publish` | 身份门禁后的正式 Vision 事件 | 证明 Vision 已发布；不能证明 Tree 已选择或 Action 已执行 |

一条正例只有第2～5层都符合下表预期，才能判“识别与 Vision 路由 PASS”。若第2层
成功，但人物不是固定人脸库中的 `confirmed_known + tracking`，应看到
`event_suppressed`，该用例只能判“动作识别 PASS、身份门禁 PASS”，不能要求正式姿态
事件。`raw_scores` 单帧升高但始终没有进入 `recognized_actions[]`，动作识别仍是 FAIL。

#### 5.6.2 25 个精确动作逐项测试对齐表

“表演要点”用于统一测试人员动作，不代替算法完整公式。动态动作应完整做完一个周期；
静态动作应保持到 `recognized_actions[]` 稳定出现。所有正式姿态事件都默认要求
`identity ∈ {owner,family_member_1..4}`、`identity_state=confirmed_known`、
`tracking_state=tracking`。

| 用例 ID | 中文动作与表演要点 | 精确名称（优先级/组） | 预期兼容字段 | 身份门禁打开时的正式事件 |
|---|---|---|---|---|
| GP-001 | 受控跌倒：先直立完成布防，再快速转倒并持续躺卧 | `fall`（P0/event） | `pose_action=fallen_down` | `EVT_VISION_FALL` |
| GP-002 | 停止：举起单掌，四根非拇指伸直、掌心朝镜头并保持稳定 | `stop_gesture`（P0/event） | `hand_action=stop_gesture` | `EVT_VISION_STOP_GESTURE` |
| GP-003 | 双手叉腰：双手落在腰/髋部附近，双肘向两侧展开 | `hands_on_hips`（P1/gesture） | `pose_action=hands_on_hips` | `EVT_VISION_MASTER_SAD` |
| GP-004 | 大幅挥臂/拍打：手臂做明显、连续的大范围摆动 | `large_arm_swing`（P1/dynamic） | `pose_action=rapid_wave_slap` | `EVT_VISION_MASTER_SAD` |
| GP-005 | 指点：一只手伸出食指形成明确指向，其余手指收拢 | `pointing`（P1/gesture） | `hand_action=finger_pointing` | `EVT_VISION_MASTER_SAD` |
| GP-006 | 急促跺脚：一侧脚或膝完成可见的抬起—落下周期，身体中心基本稳定 | `stomping`（P1/dynamic） | `pose_action=stomping` | `EVT_VISION_MASTER_SAD` |
| GP-007 | 抱臂：两条前臂交叉于胸前并保持 | `arms_crossed`（P1/gesture） | `pose_action=arms_crossed` | `EVT_VISION_MASTER_SAD` |
| GP-008 | 低头：头部相对肩线明显下垂并保持 | `head_down`（P2/posture） | `pose_action=head_down_slumped` | `EVT_VISION_MASTER_SAD` |
| GP-009 | 垂肩：低头并让肩部呈明显塌陷/消沉姿态 | `shoulders_slumped`（P2/posture） | `pose_action=head_down_slumped` | `EVT_VISION_MASTER_SAD` |
| GP-010 | 掩面：双手同时靠近并遮住脸部 | `face_covering`（P2/gesture） | `hand_action=hands_covering_face` | `EVT_VISION_MASTER_SAD` |
| GP-011 | 抱头：双手同时放到头顶或头部两侧 | `hands_on_head`（P2/gesture） | `hand_action=hands_covering_face` | `EVT_VISION_MASTER_SAD` |
| GP-012 | 蜷缩：下蹲/收拢身体，使头、躯干和四肢呈明显蜷缩 | `curled_up`（P2/posture） | `pose_action=body_curled_up` | `EVT_VISION_MASTER_SAD` |
| GP-013 | 驼背：躯干明显向前弓曲并保持 | `hunched`（P2/posture） | `pose_action=hunched_back` | `EVT_VISION_MASTER_SAD` |
| GP-014 | 举手：至少一侧手腕高于肩部并保持 | `arms_raised`（P3/gesture） | `pose_action=arm_raise_wave` | `EVT_VISION_MASTER_HAPPY` |
| GP-015 | 挥手：手臂举起后做连续往返挥动 | `waving`（P3/dynamic） | `pose_action=arm_raise_wave` | `EVT_VISION_MASTER_HAPPY` |
| GP-016 | V 字手势：同一只手伸出食指和中指形成 V 字 | `victory`（P3/gesture） | `hand_action=victory` | `EVT_VISION_MASTER_HAPPY` |
| GP-017 | 跳跃：全身起跳；脚踝裁切时做肩髋整体抬升并正常回落 | `jumping`（P3/dynamic） | `pose_action=jump` | `EVT_VISION_MASTER_HAPPY` |
| GP-018 | 张开双臂：双臂向两侧明显展开，可伴轻微前倾并保持 | `arms_open`（P3/gesture） | `pose_action=lean_forward_arms_open` | `EVT_VISION_MASTER_HAPPY` |
| GP-019 | 快速点头：脸部可见，头部完成连续快速上下往返 | `fast_nod`（P3/dynamic） | `pose_action=nodding` | `EVT_VISION_MASTER_HAPPY` |
| GP-020 | 鼓掌：双掌在胸前重复靠近—接触—分开 | `clapping`（P3/dynamic） | `hand_action=clapping` | `EVT_VISION_MASTER_HAPPY` |
| GP-021 | 点赞：一只手拇指伸出，其余手指收拢并保持 | `thumbs_up`（P3/gesture） | `hand_action=thumbs_up` | `EVT_VISION_MASTER_HAPPY` |
| GP-022 | 站立：躯干直立，腿部呈站姿并稳定保持 | `standing`（P4/posture） | `pose_action=neutral_stand_sit` | `EVT_VISION_MASTER_NEUTRAL` |
| GP-023 | 坐姿：躯干直立、髋膝弯曲形成明确坐姿并保持 | `sitting`（P4/posture） | `pose_action=neutral_stand_sit` | `EVT_VISION_MASTER_NEUTRAL` |
| GP-024 | 静态躺卧：测试人员已经躺好后进入画面或保持静止 | `lying`（P4/posture）；同时 `pose_state=lying` | 无精确动作兼容输出 | 无；静态躺卧不得产生 `EVT_VISION_FALL` |
| GP-025 | 低运动：自然站立或坐下并长时间基本不动 | `low_motion`（P4/activity） | `pose_action=neutral_stand_sit` | `EVT_VISION_MASTER_NEUTRAL` |

其中 P0 优先级最高，P4 最低；这是识别器在同一帧选择主动作的优先级，不是 ROS QoS，
也不是行为树候选优先级。一个帧中可以同时存在一个 `pose_action` 和一个
`hand_action`，所以例如站立时做 Stop，正式消息可以同时保留粗姿态和 Stop 手势。

#### 5.6.3 多模态手持姿态对齐表

下面两项不属于25个 GesturePose 原始标签，而是 Pose/Hand 与物体结果关联后产生：

| 用例 ID | 测试输入 | 诊断名称与确认条件 | 预期兼容字段 | 身份门禁打开时的正式事件 |
|---|---|---|---|---|
| MM-001 | 手持支持的玩具并让物体靠近有效手腕 | `candidate_action=holding_toy`；两个阳性物体结果后 `action=holding_toy` | `pose_action=holding_toy`、标签“手持玩具” | `EVT_VISION_TOY` |
| MM-002 | 手持狗碗、狗粮罐或狗粮袋并靠近有效手腕 | `candidate_action=holding_dog_food`；两个阳性物体结果后 `action=holding_dog_food` | `pose_action=holding_dog_food`、标签“手持狗粮” | `EVT_VISION_FOOD` |

只在画面、地面或桌面上检测到对应物体不算手持姿态，也不能下发 TOY/FOOD。详细的
同步时间、腕部距离、两次证据和反例要求见5.10节。

#### 5.6.4 名称折叠与事件共用关系

以下名称会在正式接口中合并。测试精确动作时必须以 `recognized_actions[]` 为准，
不能倒推：

| 精确识别名称 | 合并后的正式兼容值 | 共用事件 | 测试注意事项 |
|---|---|---|---|
| `arms_raised`、`waving` | `pose_action=arm_raise_wave` | `EVT_VISION_MASTER_HAPPY` | 事件和兼容值都不能区分“静态举手”还是“动态挥手” |
| `head_down`、`shoulders_slumped` | `pose_action=head_down_slumped` | `EVT_VISION_MASTER_SAD` | 必须回看精确名称判断识别的是低头还是垂肩 |
| `face_covering`、`hands_on_head` | `hand_action=hands_covering_face` | `EVT_VISION_MASTER_SAD` | 正式字段不能区分掩面与抱头 |
| `standing`、`sitting`、`low_motion` | `pose_action=neutral_stand_sit` | `EVT_VISION_MASTER_NEUTRAL` | 站/坐看 `pose_state` 和精确名称；低运动看精确名称 |
| P1/P2 的多个负向动作 | 各自兼容值 | `EVT_VISION_MASTER_SAD` | Sad 只证明负向类别，不证明具体动作 |
| P3 的多个积极动作 | 各自兼容值 | `EVT_VISION_MASTER_HAPPY` | Happy 只证明积极类别，不证明具体动作 |

静态 `lying` 是明确的“仅调试/状态输出”能力：识别到它但没有兼容动作和动作事件是
当前正确行为，不得登记为“事件漏发”。`victory` 已正式折叠为 `hand_action=victory`，
并路由到 `EVT_VISION_MASTER_HAPPY`。`lying != fall`；只有完成跌倒状态机
的直立布防、快速转换和持续躺卧，才会得到 `fallen_down/EVT_VISION_FALL`。

#### 5.6.5 同一动作在不同身份下的期望事件组

事件生成顺序固定为“人脸事件 → 主目标姿态事件 → 手势事件”，数组内同名去重。
以跳跃为例：

| 场景 | GesturePose/兼容结果 | 正式 `events[]` 典型结果 | 判定 |
|---|---|---|---|
| 主人已确认并在跟踪 | `recognized_actions=jumping`、`pose_action=jump` | `EVT_VISION_MASTER`，随后 `EVT_VISION_MASTER_HAPPY` | 动作识别、折叠和事件路由均可 PASS |
| 第一次匹配，仅 `candidate_known` | 同上 | 可有人脸事件，但不得有 `EVT_VISION_MASTER_HAPPY` | 动作识别 PASS；门禁阻止姿态事件 PASS |
| 陌生人/unknown | 同上 | `EVT_VISION_STRANGER`；不得有 Happy | 动作识别 PASS；门禁 PASS |
| 没有检测到人脸 | 可保留 GesturePose 诊断 | 通常没有 Master/Stranger，也不得有 Happy | 只判识别层和门禁层 |
| 已知身份但 `temporarily_lost` | 可保留旧诊断 | 不得发布新的姿态事件 | 门禁 PASS |

其他动作把 Happy 换成上表对应的 Sad/Neutral/Fall/Stop/TOY/FOOD 即可。注意
`EVT_VISION_MASTER` 只表示当前主目标身份非 `unknown` 且存在人脸观察，不等于身份已
达到 `confirmed_known`；姿态事件仍必须单独检查 `pose_event_gate=open`。

#### 5.6.6 动作专项判定与统计口径

建议每个动作正例执行10次，并执行最相似反例至少10次。项目若另有正式准确率门槛，
以测试计划为准；没有冻结门槛时必须报告原始次数，不能用“偶尔成功”代替结论：

```text
精确动作识别召回率 = recognized_actions 精确名称正确次数 / 正例执行次数
兼容字段正确率 = 兼容字段和值正确次数 / 精确动作识别成功次数
Vision 事件路由正确率 = 门禁打开且期望事件正确次数 / 门禁打开的识别成功次数
反例误触发率 = 反例中出现该精确名称的次数 / 反例执行次数
```

每一次动作至少保存：动作起止时间和录像时间码、`primary_action/primary_priority`、
对应 `raw_scores[]` 条目、`recognized_actions[]` 条目、`support_ratio/duration_s`、兼容
动作、`pose_event_gate`、最终 `events[]` 和同一时段的 `VISION_TRACE`。测试表建议直接
使用以下列：

| Case ID | 身份/门禁 | 表演动作 | 期望精确名 | 实际精确名 | 期望兼容字段 | 实际兼容字段 | 期望事件 | 实际事件 | 识别结论 | 路由结论 | 证据路径 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `GP-017-r01` | `owner/confirmed_known/tracking/open` | 原地小跳 | `jumping` |  | `pose_action=jump` |  | `EVT_VISION_MASTER_HAPPY` |  |  |  |  |
| `GP-024-r01` | `owner/confirmed_known/tracking/open` | 已躺好后保持 | `lying` |  | `pose_state=lying`，无动作兼容值 |  | 无 Fall |  |  |  |  |

#### 5.6.7 当前正式事件库存

当前代码实际可能写入 `/perception/visual_event.events[]` 的事件只有下表9种：

| 正式事件 | 来源 | 是否需要姿态身份门禁 | 同包主要证据 |
|---|---|---:|---|
| `EVT_VISION_MASTER` | 当前有 `faces[]`，且主目标 `identity` 非空、非 `unknown` | 否；它本身也不证明 `confirmed_known` | `faces[]`、`active_target.identity/identity_state` |
| `EVT_VISION_STRANGER` | 当前有 `faces[]`，且主目标身份未知 | 否 | `faces[]`、`active_target.identity_state` |
| `EVT_VISION_MASTER_HAPPY` | GP-014～GP-021 的兼容动作 | 是 | `recognized_actions[]`、`pose_action` 或 `hands[].hand_action` |
| `EVT_VISION_MASTER_SAD` | GP-003～GP-013 的兼容动作 | 是 | 同上 |
| `EVT_VISION_MASTER_NEUTRAL` | GP-022、GP-023、GP-025 折叠出的 `neutral_stand_sit` | 是 | `pose_state`、`recognized_actions[]`、`pose_action` |
| `EVT_VISION_FALL` | GP-001 跌倒状态机确认 | 是 | `fall_detector`、`pose_action=fallen_down` |
| `EVT_VISION_STOP_GESTURE` | GP-002 Stop 时序确认 | 是 | 手部特征、`hands[].hand_action=stop_gesture` |
| `EVT_VISION_TOY` | MM-001 手持玩具确认 | 是 | `active_target.held_object`、`pose_action=holding_toy` |
| `EVT_VISION_FOOD` | MM-002 手持狗粮确认 | 是 | `active_target.held_object`、`pose_action=holding_dog_food` |

`EVT_VISION_ANIMAL_CALM/GREET/PLAY/BOUNDARY` 虽然已声明常量，但当前没有生成路径，
属于预留事件。测试中没有看到它们是正确现状；若意外出现反而应登记缺陷。

上述“下发”只表示 Vision 已把字符串写入正式 Topic。状态保持期间，同一个事件可能随
约10 Hz状态流重复出现；Viewer 的 `ENTER/ACTIVE/EXIT` 是对状态流的页面压缩记录。
需要验证行为树和动作执行时，必须继续关联 Tree 的候选注入/选择日志以及 Action 的
goal/result，不能用 Vision `event_publish` 代替端到端成功。

#### 5.6.8 单次动作的最小证据链示例

已确认主人完成一次跳跃时，至少应保存同一人物、相邻时间段的以下证据：

```text
/perception/vision/gesture_debug
  track_id=46
  recognized_actions[].name=jumping
  primary_action=jumping
  legacy_pose_action=jump

/perception/visual_event
  active_target.track_id=46
  active_target.identity=owner
  active_target.identity_state=confirmed_known
  active_target.tracking_state=tracking
  active_target.pose_action=jump
  events[]=EVT_VISION_MASTER_HAPPY

VISION_TRACE
  record=event_publish
  event_type=EVT_VISION_MASTER_HAPPY
  track_id=46
  identity_state=confirmed_known
  pose_action=jump
  pose_event_gate=open
```

陌生人完成同样跳跃时，前两行 GesturePose 识别证据仍可成立，但正式证据应改为：

```text
/perception/visual_event events[]=EVT_VISION_STRANGER
VISION_TRACE record=event_suppressed reason_code=identity_not_confirmed
  pose_action=jump pose_event_gate=blocked
```

如果只保存到 `raw_scores[name=jumping]`，缺少 `recognized_actions`，不能证明识别完成；
如果只保存到 `EVT_VISION_MASTER_HAPPY`，则无法区分它来自跳跃、举手、挥手、V字、
鼓掌、点赞或其他积极动作，也不能证明测试的精确动作正确。

推荐直接用 rosbag 同时保存精确动作和正式事件。开始录制后执行一次动作，完成后按
`Ctrl-C` 停止：

```bash
ros2 bag record \
  -o "test-evidence/$TEST_RUN_ID/GP-017-r01" \
  /perception/vision/gesture_debug \
  /perception/visual_event
```

不录 bag 时，至少在两个终端分别保存一段连续 Topic 原文，不能只取动作前或动作后的
单帧：

```bash
timeout 15s ros2 topic echo /perception/vision/gesture_debug \
  > "test-evidence/$TEST_RUN_ID/GP-017-r01-gesture.yaml"

timeout 15s ros2 topic echo /perception/visual_event \
  > "test-evidence/$TEST_RUN_ID/GP-017-r01-visual.yaml"
```

同一用例的结构化日志使用 `case_id` 提取：

```bash
rg '"case_id":"GP-017-r01"' \
  "test-evidence/$TEST_RUN_ID"/vision_trace_*.jsonl \
  > "test-evidence/$TEST_RUN_ID/GP-017-r01-trace.jsonl"
```

上述三个文件分别证明精确识别、正式发布和门禁/事件边沿，缺一项时应在报告中明确
标为“未取证”，不能从另外一项推断。

### 5.7 Stop 手势

正例：正对相机，单手四根非拇指伸直，掌心朝镜头，手部保持稳定。高位俯视安装下，
掌面明显朝前或手臂接近伸直可提供几何证据；Pose/Hand 手腕关联仍是硬条件。

| 观察字段 | 预期 |
|---|---|
| `hand_features.left/right.four_fingers_extended` | 正例应达到 `4` |
| `palm_facing_score` | 正例明显升高，结合页面原始分判断 |
| `motion_energy` | 保持动作时应下降，避免把挥手当 Stop |
| `stop_wrist_height_ratio` | 以髋线为0、肩线为1；自然下垂通常不大于0，举掌应进入正值指令区 |
| `stop_command_zone_score` | Stop 高度门控分；低于0.35时直接拒绝 |
| `raw_scores[name=stop_gesture].score` | 达到规则阈值后成为候选；当前阈值 0.65 |
| `recognized_actions[name=stop_gesture]` | 5 帧窗口中至少 3 帧支持后出现 |
| `hands[].hand_action` | 稳定后为 `stop_gesture` |
| `events[]` | 仅已确认固定身份出现 `EVT_VISION_STOP_GESTURE` |

反例必须包括：双手自然下垂、手贴大腿、快速挥手、只伸食指、握拳、侧掌、抱臂。
自然下垂时即使手臂笔直、四指伸开，也必须因 `stop_command_zone_score<0.35` 被拒绝；
抱臂分数明显成立时 Stop 候选应被冲突规则清除。事件历史中 Stop 用黄色边线显示。

### 5.8 跳跃

跳跃不直接依赖普通动作的单帧分数，而是使用两条互斥通道：

1. `full_body`：髋部和双脚踝共同向上，0.35秒内两个证据帧直接确认。
2. `upper_body_fallback`：仅在脚踝无法测量、但肩部和髋部有效时启用。先要求
   约0.2秒稳定基线，再要求肩髋同步快速上升、躯干尺度变化不大，并累计相对基线的
   整体抬升量。为适配约5 FPS的现场推理频率，一个强起跳帧或0.35秒内两个普通
   起跳帧即可进入回落等待，最后必须在1.1秒内观察到肩髋同步回落或回到基线才确认。

脚踝可见但保持原位时，不会改走半身通道，因此普通蹲起/起身仍被拒绝。两条通道确认后
都把结果保持约0.65秒，供3/5帧稳定器输出 `jumping`。

| 字段 | 预期 |
|---|---|
| `jump_detector.mode` | `full_body`、`upper_body_fallback` 或 `unavailable` |
| `jump_detector.phase` | `monitoring/takeoff_candidate/awaiting_return/active/cooldown` |
| `jump_detector.missing_landmarks` | 列出当前缺失的肩、髋或脚踝关键点 |
| `temporal_features.shoulder_vertical_velocity` | 半身起跳向上时为明显负值 |
| `temporal_features.hip_vertical_velocity` | 起跳向上时为明显负值 |
| `temporal_features.ankle_vertical_velocity` | 双脚离地向上时同样为明显负值 |
| `temporal_features.torso_scale_change_ratio` | 半身通道要求躯干形状基本稳定，避免弯腰/起身误触发 |
| `jump_detector.full_body_score` | 髋部与脚踝共同上升分，达到0.55才累计 |
| `jump_detector.upper_body_score` | 缺脚踝时的肩髋同步上升分；0.35累计普通帧，0.55可作为单个强起跳帧 |
| `jump_detector.return_score` | 半身通道回落分，当前阈值0.25 |
| `jump_detector.baseline_ready` | 半身稳定基线是否就绪；未就绪时先静止约0.2秒 |
| `jump_detector.upward_displacement_ratio` | 肩髋中心相对稳定基线的上移量，以躯干尺度归一化 |
| `jump_detector.component_scores` | 肩、髋、脚踝上升、肩髋同步、躯干稳定、基线抬升及位置回落分；最小项可定位当前瓶颈 |
| `jump_detector.evidence_score` | 当前可用通道的起跳证据分 |
| `jump_detector.evidence_frames` | 起跳阳性帧数；半身一个强帧或两个普通帧后进入 `awaiting_return` |
| `jump_detector.rejection_reason` | 基线不足、运动不足、等待回落或未观察到回落等原因 |
| `jump_detector.event_triggered` | 每次确认只在一帧为 `true` |
| `jump_detector.active/hold_remaining_s` | 确认后约0.65秒保持，保证页面和状态流可观察 |
| `recognized_actions[name=jumping]` | 保持期间经过稳定器后出现 |
| `active_target.pose_action` | 已稳定后为 `jump` |
| `events[]` | 仅已确认固定身份映射为 `EVT_VISION_MASTER_HAPPY` |

正例至少做全身原地小跳、全身正常原地跳、脚踝被画面裁掉时的原地跳各3次；反例至少
做快速起立、踮脚、走路、上下蹲各3次。半身正例应看到 `mode=upper_body_fallback`，
`phase` 依次进入 `takeoff_candidate`、`awaiting_return`和`active`。若证据确认但
没有正式事件，再检查 `recognized_actions` 和身份门控。
节点日志中的 `GesturePose jump confirmed` 应包含对应 `track_id`、`mode`、确认时的
全身/半身/回落证据及缺失关键点；它证明识别器完成确认，不代表身份门控后的正式
事件一定已经发布。

### 5.9 跺脚

跺脚使用腿部关键点相对髋部的局部运动，不使用整个人在画面中的绝对位移：

1. `ankle`：脚踝有效时，要求脚踝相对同侧髋部出现足够的垂直速度、幅度以及至少
   一次方向反转，证明完成“抬起—落下”周期。
2. `knee_fallback`：脚踝因近距离构图被裁掉时，改用膝部相对同侧髋部的相同周期；
   膝部也不可用时不识别。

两条通道都要求肩部/髋部整体垂直速度较低、人物保持直立，并排除过大的全身运动。
这会阻止跳跃的整个人共同上下移动直接形成跺脚证据。完成周期后采用3/5帧确认，
适配约5 FPS的现场推理频率。

| 字段 | 预期 |
|---|---|
| `stomp_detector.source` | 脚踝有效时为 `ankle`；脚踝裁切且膝部有效时为 `knee_fallback` |
| `stomp_detector.score` | 原始跺脚分，达到0.55后进入3/5帧确认 |
| `stomp_detector.recognized` | 时序确认后为 `true` |
| `ankle/knee_vertical_speed` | 对应关键点相对髋部的近期最大垂直速度 |
| `ankle/knee_vertical_range_ratio` | 抬落幅度，以躯干尺度归一化 |
| `ankle/knee_direction_changes` | 完整抬落至少为1；只抬腿不落下应为0 |
| `recognized_actions[name=stomping]` | 完成抬落并通过3/5帧后出现 |
| `active_target.pose_action` | 已稳定后为 `stomping` |
| `events[]` | 仅已确认固定身份映射为 `EVT_VISION_MASTER_SAD` |

正例分别做脚踝完整可见和脚踝被裁切的原地单脚跺脚各10次。反例至少包括抬腿保持、
原地小跳、快速蹲起和正常走路；反例不应出现 `recognized_actions=stomping`。

### 5.10 手持玩具/狗粮特定姿态

支持的玩具标签为 `dog toy ball/dog frisbee toy/dog tug ring toy`，狗粮标签为
`dog bowl/dog food can/dog treat bag`。检测到人物后，内部
`vision-human-holding` 流自动以2 Hz运行；显式 Action 或页面持续识别会话优先，
不会和内部流并发抢占RKNN。

判定必须同时满足：物体置信度至少0.35；物体框接近当前人物框内的有效
Pose/Hand 手腕；Pose与物体源帧时间差不超过0.75秒；1.5秒内两个不同物体推理
结果均成立。允许两个阳性结果之间出现一次短暂漏检，但漏检不计证据，超过确认窗口
仍会失效。确认后短暂保持1.25秒，跌倒姿态不被覆盖。物体可随伸出的手部分超出
人物框，单独位于人物附近但远离手腕的物体仍不得触发。

| 字段/日志 | 含义与通过标准 |
|---|---|
| `active_target.pose_action` | 确认后为 `holding_toy` 或 `holding_dog_food` |
| `active_target.pose_action_label` | `手持玩具` 或 `手持狗粮` |
| `held_object.state` | `inactive/candidate/confirmed` |
| `held_object.object_label/object_track_id` | 实际关联的模型标签和稳定物体ID |
| `held_object.hand_source` | 关联的 Pose 手腕或 HandLandmarker 手腕 |
| `held_object.association_score` | 手腕接近度与物体置信度组成的诊断分 |
| `held_object.wrist_distance_ratio` | 手腕到物体框距离除以人体框对角线 |
| `held_object.evidence_hits/required_hits` | 第一帧为 `1/2`，第二个不同物体结果后确认 |
| `held_object.object_result_sequence` | 实际参与本次手持判断的物体推理结果序号 |
| `held_object.rejection_reason` | 最近一次判断未成立的明确原因；成功关联时为空 |
| `pose_object_sync_delta_ms` | 两条异步流水线的源帧时间差；当前上限750 ms |
| `evaluated_wrist_distance_ratio/wrist_distance_threshold_ratio` | 最近物体到最近有效手腕的归一化距离及阈值 |
| `Visual state changed ... held=...` | 应依次记录 candidate、confirmed及释放状态 |

板端复测时可直接提取每个实际物体推理结果的手持判断：

```bash
rg '"record":"held_object_evaluation"' /tmp/marsdog_vision_qa/HOLD-01/*.jsonl
```

`reason_code=""` 表示本次关联成立，并应看到 `evidence_hits` 从1到2。
`timestamp_mismatch` 重点看 `pose_object_sync_delta_ms`；`wrist_too_far` 比较
`wrist_distance_ratio` 与 `wrist_distance_threshold_ratio`；`no_valid_wrist` 则检查人物是否
完整入镜及手腕关键点是否有效。`stale_object_sequence` 表示展示的是
以前物体结果中保持的轨迹，不能作为当前证据；若每个新结果都出现该原因，
应检查 `tracked_objects[].source_sequence` 是否与 `object_result_sequence` 一致。

测试用例：

1. 已登记人员分别手持三种玩具、狗粮罐、狗粮袋和狗碗，每种保持至少2秒。
2. 将相同物体放在地上、桌面或人物框内但远离双手，均不得出现手持姿态。
3. 只让物体靠近手腕不到一次检测周期，然后移开，不得达到 `confirmed`。
4. 两个人同时入镜，把物体放在非当前目标手中，不得关联到当前目标。
5. 陌生人手持时可以看到结构化姿态，但 `events[]` 不得出现 TOY/FOOD。
6. 已确认主人/家人手持时，分别出现 `EVT_VISION_TOY/EVT_VISION_FOOD`；物体单独
   出现不得再触发这两个事件。
7. 人物离开后内部物体流最多保持2秒；通过页面启动显式流时，应立即显示显式
   session，停止后人物仍在则自动恢复手持判断流。

### 5.11 跌倒

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

### 5.12 陌生人事件与下游融合边界

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

### 5.13 视觉事件历史与发布频率

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

### 5.14 单次与持续物体识别

正式启动时固定物体流为0 Hz；人物进入后，手持姿态内部流自动以2 Hz运行，人物
离开2秒后停止。手动按钮、Action任务或内部流第一次调用都可能触发RKNN懒加载，
必须分别记录首次和稳态耗时。

单次测试：设置置信度，点击“单次物体识别”。连续测试：设置置信度和 0.1～5 Hz
频率，点击“启动持续识别”，保持 40 s 后停止；另做关闭浏览器不点停止的测试。

| 字段 | 单次预期 | 持续预期 |
|---|---|---|
| `schema_version` | 2 | 2 |
| `source` | `service` | 推理包为 `stream`；停止包为 `control` |
| `status` | `ok` 或 `error` | `ok/error/stopped` |
| `stream.active` | 不因单次创建 session | 启动后 `true` |
| `stream.session_id` | 空或当前有效流 | 页面固定 `vision-debug-web`；自动手持流为 `vision-human-holding` |
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

### 5.15 VisionTask Service

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

### 5.16 页面布局与性能面板

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

### 6.0 `VISION_TRACE` 测试追踪

测试判定优先使用机器可解析的 `VISION_TRACE`，普通控制台日志用于人工排障，ROS
Topic/Service 原文用于证明实际接口结果。当前追踪记录包括：

| `record` | 模块/阶段 | 关键字段 |
|---|---|---|
| `runtime_start` | 节点启动 | `result,node,vision_epoch,camera_topic,visual_topic,service,timing_trace_interval_sec` |
| `stage_start/stage_complete` | VisionTask | `task_id,task_type,result,latency_ms,error` |
| `stage_complete` | 物体推理 | `source,session_id,sequence,latency_ms,object_count` |
| `stage_complete` | `continuous_vision/pipeline` | `inference_sequence,latency_ms,face_count,human_count,hand_count` |
| `stage_complete` | `face_detection/yunet_inference` | `inference_sequence,latency_ms,detection_count` |
| `stage_complete` | `face_tracking/bytetrack_update` | `inference_sequence,latency_ms,detection_count,track_count` |
| `stage_complete` | `face_recognition/sface_inference` | `inference_sequence,track_id,latency_ms,identity,confidence,reason_code` |
| `stage_complete` | `face_recognition/sface_task_recognize` | `latency_ms,identity,confidence,reason_code,template_identity_count` |
| `stage_complete` | `pose_landmarker/inference` | `inference_sequence,latency_ms,detection_count,model_variant` |
| `stage_complete` | `hand_landmarker/inference` | `inference_sequence,latency_ms,detection_count` |
| `stage_complete` | `gesture_pose/feature_extraction` | `inference_sequence,track_id,latency_ms` |
| `stage_complete` | `gesture_pose/action_recognition` | `inference_sequence,track_id,latency_ms,primary_action` |
| `stage_complete` | `depth_fusion/aligned_depth_range` | `observation_stamp,latency_ms,candidate_count,fused_count` |
| `stage_complete` | `visual_event/event_publish` | `sequence,latency_ms,event_count,events[]` |
| `event_publish` | 正式视觉事件首次出现 | `event_type,vision_epoch,sequence,track_id,identity_state` |
| `event_cleared` | 正式事件退出 | 同上 |
| `event_suppressed` | 姿态候选被身份门控 | `reason_code,pose_event_gate,identity_state,tracking_state` |

提取某一轮或某一用例：

```bash
rg 'VISION_TRACE' "test-evidence/$TEST_RUN_ID" | \
  rg '"case_id":"STOP-01-r1"'
```

`event_publish` 是相对上一视觉状态首次出现的边沿证据；不要用约 10 Hz
`/perception/visual_event` 包数量代替物理事件次数。`latency_ms` 的语义由
`module/stage` 限定：Service 是完整回调耗时，物体阶段是推理耗时，两者不可混算。

连续视觉阶段默认按 `logging.timing_trace_interval_sec=5.0` 各自采样，记录中的
`sampled=true` 表示采样耗时；失败记录不受限频影响。SFace 只在身份节流器实际请求
识别时运行，因此没有人脸、未录入人脸或未到复核周期时不会产生该阶段记录。需要逐次
性能分析时可把该配置设为 `0`，但会明显增加日志量，不建议长期运行。物体推理和
VisionTask 是按需任务，仍然逐次记录。

测试报告不得把以下数值混为同一种耗时：

- `continuous_vision/pipeline`：单次连续视觉模型流水线，包含人脸、Pose 和当帧实际运行的 Hand。
- `pose_landmarker`、`hand_landmarker`、`face_*`：各模型或子阶段自身耗时。
- `gesture_pose/*`：关键点结果之后的特征提取和规则判断，不包含模型推理。
- `depth_fusion`：深度匹配、取样和反投影；`depth_sync_delta_ms` 是时间戳差，不是计算耗时。
- `visual_event/event_publish`：JSON 序列化和 ROS Publisher 调用耗时，不是相机输入到动作执行的端到端耗时。
- `vision_task/service`：完整 Service 回调耗时，可能包含其中调用的物体推理。

### 6.1 控制台日志

| 日志前缀 | 关键字段 | 用途 |
|---|---|---|
| `Vision node ready` | camera、visual、objects、startup、depth、service | 证明实际接线和物体启动频率 |
| `VisionObservationProvider started` | 模型加载数、stride、最大 Pose 数、Hand idle stride | 证明真实推理 Provider 已启动 |
| `Face identity changed` | track、identity、state、confidence、attempts | 身份状态机变化证据 |
| `Visual state changed` | track、tracking、identity、identity_state、pose、hands、pose_event_gate、events | 身份门控和最终事件证据 |
| `Fall event confirmed` | transition、lying | 一次跌倒确认边沿 |
| `GesturePose jump confirmed` | track、mode、全身/半身/回落证据、缺失关键点 | 一次跳跃确认边沿及所用识别通道 |
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
| 手势 | `hand_features.left/right`，包括 Stop 手腕高度与指令区域分 |
| 跳跃 | `jump_detector` 的识别模式、阶段、缺失关键点、全身/半身/回落证据、确认边沿、保持、冷却和拒绝原因 |
| 跺脚 | `stomp_detector` 的脚踝/膝部通道、原始分、识别状态、速度、幅度和换向次数 |
| 手持姿态 | 正式视觉包 `active_target.held_object`，不属于 GesturePose 原始分数 |
| 时序 | `temporal_features`，包括头部运动、肩/髋/脚踝垂直速度、躯干尺度变化和双腕开合 |
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
| 物体有框但没有手持姿态 | 未接近手腕、源帧不同步、置信度不足或只有一次证据 | `held_object.state/hand_source/wrist_distance_ratio/evidence_hits`及物体`source_sequence/header.stamp` |
| 地面物体误报手持 | 人物框/手腕关联过宽或关键点错误 | 保存人物框、手腕和物体框；不得先降低两帧确认规则 |
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
| 动作识别结论 | `raw_scores → recognized_actions` 是否符合预期；PASS / FAIL / BLOCKED |
| Vision 路由结论 | 兼容字段、身份门禁、`events[]` 是否符合预期；PASS / FAIL / BLOCKED |
| 下游结论 | 未联调 Tree/Action 时填“未验证”；不得用 Vision 发布代替执行成功 |
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
