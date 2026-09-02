# MarsDog Vision Interaction

> 项目交接请先阅读 [docs/HANDOFF.md](docs/HANDOFF.md)。四项目总览、统一接口清单
> 和联调手册位于 [docs/integration/README.md](docs/integration/README.md)。测试工程师请
> 使用 [视觉测试工程师验收说明](docs/TEST_ENGINEER_GUIDE.md)，其中包含逐模块测试表、
> 日志字段、通过标准和证据模板。

独立的 MarsDog 视觉交互 Python/ROS2 项目，负责相机、视觉推理、人脸注册和
`/perception/visual_event`。本项目不导入语音项目，也不维护 VAD、声纹或语音会话状态。

## 环境

- Python 3.10
- uv
- ROS2 Humble

`rknn-toolkit-lite2` 只在 Linux AArch64（RK3588）环境安装；其他平台仍可安装、
运行单元测试和 Mock 联调，但不能执行 RKNN 模型。

模型目录按以下顺序解析：`MARSDOG_VISION_MODEL_DIR` 环境变量、仓库内
`models/vision`、仓库同级的 `models/vision`。因此既可以让单项目自包含，也可以
让多个项目共享一份大模型而不写死开发机目录。运行数据默认写入仓库内 `data`，
可通过 `MARSDOG_VISION_DATA_DIR` 覆盖。
人脸注册表和人脸样本是运行时生成的生物识别数据，仅保存在本机 `data/`
目录，不进入 Git，也不打进源码发布包；新设备需单独完成人脸注册或安全迁移数据。

以下命令均假定当前目录为仓库根目录，并且已经按本机 ROS2 安装方式加载环境：

```bash
uv sync --extra models --extra dev
uv run pytest
```

直接运行源码节点：

```bash
uv run marsdog-vision-interaction \
  --ros-args -p config_path:=config/vision.yaml
```

ROS2 构建：

```bash
colcon build --base-paths . --packages-select marsdog_vision_interaction
source install/setup.bash
ros2 launch marsdog_vision_interaction vision.launch.py
```

该 launch 只启动视觉节点，并订阅已经存在的 RealSense 彩色图、对齐深度与
CameraInfo；相机驱动需单独启动且启用 `enable_depth` 与 `align_depth.enable`。

将 `providers.vision.type` 设置为 `mock` 可以在没有模型和相机的环境中做下游联调。
真实配置不会在模型、相机或 RKNN 不可用时自动伪造人体/物体；对应 Topic
字段为空或 Service 明确返回失败。

姿态与手势使用基于 MediaPipe 关键点的时序规则引擎。每个稳定目标独立保存
动作历史；跌倒必须经过“稳定直立、快速转变、持续躺卧”才产生事件，静态躺卧
只属于姿态，不触发跌倒告警。生产配置使用 `inference_frame_stride: 2`，即
相机第 1、3、5…帧执行完整人脸/姿态/手部推理；30 Hz 输入时推理上限约 15 Hz，
但相机新鲜度仍按每个原始帧更新。相机回调只替换一个最新待处理帧，完整模型
流水线由独立单线程工作器执行；算力不足时丢弃旧候选帧而不积压延迟，10 Hz
事件发布和 VisionTask Service 不再等待关键点推理完成。Pose 和 Hand 默认使用
MediaPipe `VIDEO` 模式，通过严格递增的单调时间戳复用帧间跟踪。
正式配置每帧最多请求 5 个 Pose 并输出独立 `human_candidates[]`；兼容
`active_target` 仍只选一个人。Hand 在未发现手部时每两个关键点推理帧探测一次，
发现手部后连续逐帧推理至少 8 次，在降低空场景负载的同时保留手势响应。

### Pose Lite/Full A/B

Lite 和 Full 模型统一放在共享模型目录。Full 模型可用以下命令安装：

```bash
export MARSDOG_VISION_MODEL_DIR="${MARSDOG_VISION_MODEL_DIR:-$PWD/models/vision}"
mkdir -p "$MARSDOG_VISION_MODEL_DIR"
curl --fail --location \
  --output "$MARSDOG_VISION_MODEL_DIR/pose_landmarker_full.task" \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

不修改 YAML 即可分别启动两组实验：

```bash
ros2 launch marsdog_vision_interaction vision.launch.py \
  pose_model_variant:=lite

ros2 launch marsdog_vision_interaction vision.launch.py \
  pose_model_variant:=full
```

每组在相同站位、光照和动作下保持至少150个推理帧。Web Viewer 的“关键点模型
A/B”面板显示有效推理FPS、平均/P95耗时、人体检测率、33点有效率及动作关键点
有效率，并显示接收帧、推理候选、实际完成和被新帧替换的数量。画面无人时的
检测率没有比较意义；选择 Full 前应确认有效推理频率满足实际动作时序要求。

## ROS2 接口

- 订阅：`/camera/camera/color/image_raw`
- 订阅：`/camera/camera/aligned_depth_to_color/image_raw`
- 订阅：`/camera/camera/color/camera_info`
- 发布：`/perception/visual_event`
- 发布：`/perception/vision/object_detections`（按需物体检测数据流）
- Service：`/perception/vision/task`（单次查询及物体流控制）

外部相机流超过 0.5 秒未更新时，视觉节点停止发布缓存场景事实，并让目标自然
进入丢失状态。人体米制距离只有在对齐深度、内参、frame、时间同步和躯干 ROI
都有效时才设置 `range_valid=true`；否则 fail closed，不使用框高度估距。

## 按需物体检测

正式配置启动时不运行物体模型。动作系统开始寻物任务后，通过
`VisionTask.set_object_detection` 开启数据流，订阅
`/perception/vision/object_detections` 取得二维检测结果，并在完成、取消或异常
路径中关闭。视觉节点不控制底盘、不调用 Nav2、不判断是否已经靠近物体。

开启一个 2 Hz、3 秒租约的数据流：

```bash
ros2 service call /perception/vision/task \
  marsdog_vision_interaction/srv/VisionTask \
  "{task_id: 'find-object-001-start', task_type: 'set_object_detection', \
    params_json: '{\"enabled\":true,\"session_id\":\"find-object-001\",\
\"rate_hz\":2.0,\"confidence\":0.25,\"target_labels\":[\"dog toy ball\"],\
\"lease_sec\":3.0}'}"
```

动作系统应在 3 秒到期前使用相同 `session_id` 重复调用完成续租。结束时关闭：

```bash
ros2 service call /perception/vision/task \
  marsdog_vision_interaction/srv/VisionTask \
  "{task_id: 'find-object-001-stop', task_type: 'set_object_detection', \
    params_json: '{\"enabled\":false,\"session_id\":\"find-object-001\"}'}"
```

`detect_objects` 仍是单帧同步查询，不会开启数据流。Topic 使用 schema v2，包含
`stream.session_id`、租约剩余时间、请求标签、时间戳、推理耗时和停止原因；完整
格式见 [docs/ROS2_CONTRACT.md](docs/ROS2_CONTRACT.md)。

## 陌生人事件职责边界

视觉节点只判断“当前存在未登记人脸”，并统一发布
`EVT_VISION_STRANGER`。本项目不订阅 `/emotion/state`，也不再产生
`EVT_VISION_STRANGER_ALERT` 或 `EVT_VISION_STRANGER_FRIEND`。

需要根据 Anxiety/Fear/Joy/Excite/Calm 等情绪决定陌生人行为时，由下游行为树同时
消费 `/perception/visual_event` 和 `/emotion/state`，在自己的候选、优先级、去重和
冷却生命周期内完成组合判断。这样视觉事件只表达可观测事实，不把机器人内部情绪
写回视觉分类。

## 统一视觉调试页面

人脸、人体/Pose、手势、视觉事件和物体识别统一使用
`vision_debug.launch.py`，不再要求测试人员切换到单独的物体调试命令。该入口默认
同时启动正式 `vision_interaction` 节点和 Web Viewer；RealSense 驱动仍需提前发布
彩色图、对齐深度和 CameraInfo。

物体 Provider 使用 `ultralytics.YOLOE` 完成图像预处理、模型调用、NMS 和
`Results.boxes` 解码。当前 `object_model` 指向 RKNN 导出目录，因此最终仍由
Ultralytics 的 RKNNBackend 通过 `rknn-toolkit-lite2` 在 RK3588 NPU 上执行；
Provider 只对 RKNN 模型已有类别做不区分大小写的精确标签过滤。

重新构建并 source 工作区后，只需执行一个调试命令：

```bash
export VISION_REPO="$PWD"
export ROS2_WS="${ROS2_WS:-$HOME/ros2_ws}"
cd "$ROS2_WS"
colcon build --symlink-install \
  --packages-select marsdog_vision_interaction
source install/setup.bash

ros2 launch marsdog_vision_interaction vision_debug.launch.py \
  web_host:=0.0.0.0
```

打开 `http://<机器狗IP>:8765`。默认 `start_vision_node:=true`，因此不要同时运行
普通 `vision.launch.py`，否则会产生同名节点和重复 Topic。如果设备上已经有正式
视觉节点，只附加 Viewer：

```bash
ros2 launch marsdog_vision_interaction vision_debug.launch.py \
  web_host:=0.0.0.0 start_vision_node:=false
```

“手势与物体”面板支持“单次物体识别”“启动持续识别”“停止持续识别”。持续识别
使用固定 `vision-debug-web` session，并每10秒续租30秒；关闭页面后最多30秒自动
停止。若 Action 已持有其他 session，页面会显示占用错误且不会抢占。物体模型保持
懒加载，首次点击可能明显较慢。

也可以绕过 ROS2，用单张图片做 RKNN 硬件冒烟测试；模型路径默认复用
`vision.yaml` 的解析结果：

```bash
uv run python tests/test_rknn.py path/to/input.jpg
```

## 调试页面功能

在同一局域网的电脑或手机打开
`http://<机器狗局域网IP>:8765`。例如机器狗 IP 为
`192.168.1.50` 时，访问
`http://192.168.1.50:8765`。页面显示人体框与姿态骨架（绿）、人脸框与
身份（蓝）、物体框（紫）、手部骨架与手势（橙）、当前目标（红），并列出
相机/视觉消息年龄、FPS、动作、置信度、事件以及原始 JSON。GesturePose 面板
显示精确的内部标签（例如 `waving`、`victory`）、P0–P4 优先级、原始候选分数
和经过时序平滑后真正命中的动作，便于区分粗粒度 `standing/lying` 与动作规则。
快速点头区域还显示头部垂直运动能量、方向反转次数和归一化位移幅度；
`fast_nod` 必须同时满足三项条件，避免鼻尖关键点的小幅抖动被误判为点头。

“Vision 已发布事件记录”面板直接在 Viewer 的 ROS Topic 回调中记录
`/perception/visual_event.events[]`，不会因网页 350 ms 轮询漏掉短事件。相同事件的
10 Hz 重复状态流会压缩成 `ENTER`（开始）、`ACTIVE`（持续）和 `EXIT`（结束），
同时保留接收时间、`vision_epoch/sequence`、目标身份、姿态和手势证据。测试人员可
复制、导出或清空本次 Viewer 进程内最多200条记录；可通过
`event_history_limit:=500` 调整为20～1000条。该面板只证明 Vision 已发布事件，
不表示行为树已选中候选，也不表示 Action 已执行；后两段链路必须查看各自日志。
桌面端会把事件记录排在右栏首位，并在滚动右侧诊断信息时固定左侧实时画面；窗口
宽度小于1050像素时自动恢复上下排列，避免手机或窄屏被固定画面遮挡。

页面的“在线人脸录入”可直接管理设备本地人脸库。身份固定为 `owner`（主人）和
`family_member_1`～`family_member_4`（家人1～4），不接受自由姓名。选择身份后，只需自然面对
摄像头并保持稳定；系统通过单人脸、检测置信度、人脸尺寸、清晰度和光照检查后，
默认自动连续保存三张，不需要转头、抬头或做其他动作。每个身份最多保存5张，
剩余槽位不足3张时页面只补满剩余槽位。完成后可在
同一页面现场验证识别、刷新名单或删除录入。统一的 `vision_debug.launch.py` 默认
已经启动正式视觉节点；只有传入 `start_vision_node:=false` 时，才要求设备上已有
可用的 `/perception/vision/task` Service。

每张样本会分别生成 SFace 模板，识别时对同一固定身份的所有模板取最高相似度。
所有姿态/手势事件只在当前主目标属于固定人脸库（`owner` 或
`family_member_1`～`family_member_4`），目标仍为 `tracking`，且身份状态达到 `confirmed_known` 后
发布。第一次匹配形成的 `candidate_known`、陌生人或未检测到人脸时，页面仍可
看到结构化姿态调试结果，但不会上报普通姿态、跌倒或 Stop 手势事件。

正式节点会在视觉状态发生变化时输出一条 `Visual state changed` 日志，包含
`track`、`identity`、`identity_state`、`pose`、`hands`、
`pose_event_gate` 和最终发布的 `events`。人脸识别状态发生变化时还会输出
`Face identity changed`。这些日志按状态变化去重，不会随 10 Hz Topic 重复刷屏。
人脸图片及 `face_registry.json` 是设备本地生物数据，位于配置的数据目录，
不得提交 Git。页面默认只监听 `127.0.0.1`；当前 Debug Web 和人脸管理接口暂不
鉴权，设置 `web_host:=0.0.0.0` 后局域网内任何客户端都可操作，因此只允许在
可信隔离网络中使用，并优先考虑下方 SSH 端口转发。

## 人脸样本 CRUD API

正式视觉节点默认在 `http://127.0.0.1:8092` 提供 FastAPI，OpenAPI/Swagger 页面为
`http://127.0.0.1:8092/docs`。接口固定提供5个身份槽位，每个身份最多5张图片：

| 身份 | 显示名称 | 角色 |
|---|---|---|
| `owner` | 主人 | `owner` |
| `family_member_1`～`family_member_4` | 家人1～家人4 | `family` |

| 方法和路径 | 用途 |
|---|---|
| `POST /api/v1/faces/{name}/samples` | 上传 JPG/JPEG/PNG，检测并新增最大人脸 |
| `GET /api/v1/faces` | 列出人员、样本数、固定上限和可用身份 |
| `GET /api/v1/faces/{name}/samples` | 列出该人员每张图片的稳定 `sample_id` 和状态 |
| `GET /api/v1/faces/{name}/samples/{sample_id}` | 查询单张样本元数据 |
| `GET /api/v1/faces/{name}/samples/{sample_id}/image` | 查看或下载裁剪后的人脸 JPG |
| `PUT /api/v1/faces/{name}/samples/{sample_id}` | 校验新图片并原位替换，编号不变 |
| `DELETE /api/v1/faces/{name}/samples/{sample_id}` | 只删除指定样本；最后一张删除后释放身份槽位 |

`sample_id` 范围固定为1～5。删除 `002.jpg` 后不会重编号 `003.jpg`；下一次新增会
复用最小空闲编号。POST/PUT 请求使用 `multipart/form-data` 的 `image` 文件字段，
默认单文件上限10 MiB。上传成功后会同步当前进程的人脸识别模板。

本机示例：

```bash
curl -X POST http://127.0.0.1:8092/api/v1/faces/owner/samples \
  -F "image=@./owner.jpg"
```

需要局域网访问时：

```bash
ros2 launch marsdog_vision_interaction vision.launch.py \
  face_api_host:=0.0.0.0
```

当前人脸 FastAPI 暂不鉴权，不需要请求头或 URL token。默认仅监听回环地址；绑定
`0.0.0.0` 时只应接入可信隔离局域网，仍推荐通过 SSH 转发 `8092`，不要把设备
本地生物数据管理接口暴露到不可信网络。

跌倒区域显示是否已完成直立基线布防、躺卧分数、转换分数和报警保持状态；
Stop 调试显示左右手四指伸直数量、掌面朝向分数及手部运动能量，可直接判断
是姿态条件未满足，还是仍在等待时序稳定。
针对安装在狗头高位并向下俯视的相机，Stop 允许“手掌明显朝前/靠近镜头”或
“手臂接近伸直”任一几何证据成立，不再要求两者同时满足；四根非拇指伸直、
掌心朝镜头、手部稳定、Pose/Hand 手腕关联以及 3/5 帧时序确认仍是硬条件。
正式视觉节点默认不运行物体检测。页面“手势与物体”面板可设置置信度和持续频率，
并通过按钮执行单次识别、启动持续识别或停止持续识别。单次调用不会创建 session；
持续调试固定使用 `vision-debug-web` session，且不会抢占 Action 已持有的 session。
首次识别需要懒加载模型，等待时间可能较长。

为避免调试页面反过来拖慢识别，Web Viewer 默认限制为 8 FPS、按 0.75 比例
渲染、JPEG 质量 75，并关闭完整 BGR 调试 Topic 的重复发布。需要原分辨率或
`/perception/vision/debug_image` 时可显式打开：

```bash
ros2 launch marsdog_vision_interaction vision_debug.launch.py \
  max_render_fps:=10 render_scale:=1.0 publish_debug_image:=true
```

页面默认只监听本机回环地址。远程查看推荐使用 SSH 端口转发：

```bash
ssh -L 8765:127.0.0.1:8765 <user>@<robot-ip>
```

启用 `publish_debug_image:=true` 后会发布 `/perception/vision/debug_image`，可以
使用 `rqt_image_view` 查看。需要保留原 OpenCV 窗口时传入 `show_window:=true`。

视觉与调试节点默认订阅 RealSense 彩色流
`/camera/camera/color/image_raw`，并按普通单目画面处理。
- 发布：`/perception/vision/enrollment_event`
- 调试：`/perception/vision/gesture_debug`
- Service：`/perception/vision/task`

视觉任务：`check_person`、`query_targets`、`detect_objects`、`set_object_detection`、
`get_object_detection_state`、`recognize_face`、
`start_face_enrollment`、`cancel_face_enrollment`、`upload_face`、
`list_faces`、`list_face_records`、`list_face_samples`、`get_face_sample`、
`replace_face_sample`、`delete_face_sample`、`delete_face`。

完整字段约定见 [docs/ROS2_CONTRACT.md](docs/ROS2_CONTRACT.md)。
