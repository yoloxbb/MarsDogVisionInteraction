# 对接归档变更记录

## 1.3.6 — 2026-09-03

- `victory` V字手势由仅调试标签升级为正式 `hand_action=victory`，已确认固定身份且
  处于 `tracking` 时下发 `EVT_VISION_MASTER_HAPPY`。
- 跺脚从全窗口平均人体运动改为腿部相对髋部的抬落周期：降低原脚踝速度门槛，要求
  垂直速度、幅度和方向反转；脚踝裁切时增加膝部兜底，并改为3/5帧确认。Viewer
  新增通道、得分及脚踝/膝部轨迹诊断。
- 跳跃增加裁切画面兜底：全身路径仍要求髋部与双脚共同上升；仅在脚踝关键点不可用
  时，半身路径才使用稳定基线、肩髋同步上升、相对基线的整体抬升量、躯干尺度稳定
  及1.1秒内同步回落或回到基线确认。
  针对约5 FPS现场采样，半身路径接受一个强起跳帧或两个普通起跳帧；脚踝可见但
  仍着地的普通起身不会绕过全身条件。
- GesturePose 与 Viewer 新增跳跃识别模式、检测阶段、缺失关键点、全身/半身/回落
  证据、稳定基线、归一化抬升量、分项瓶颈分、肩部速度、躯干尺度变化及结构化拒绝
  原因；正式事件与身份门禁不变。
- 修复 `visual_event.tracked_objects[]` 归一化时丢失 `header/source_sequence/
  source_snapshot_id` 的问题；之前手持关联器因此会把所有实际检出的玩具/
  狗粮误判为 `stale_object_sequence`，无法进入候选和确认。
- 新增多模态特定姿态 `holding_toy/holding_dog_food`：玩具/狗粮物体框必须接近
  当前人物有效手腕、源帧时间差不超过0.75秒，并由1.5秒内两个阳性物体推理结果
  确认；允许中间一次短暂漏检，物体也可随伸出的手部分超出人物框。
- `EVT_VISION_TOY/FOOD` 从“物体在画面中出现”收紧为已确认固定身份的手持姿态；
  地面、桌面或其他人附近的物体不再直接触发事件。跌倒仍优先。
- 人物处于跟踪状态时启用可让路的 `vision-human-holding` 2 Hz内部流，无人2秒后
  停止；Action/Debug显式session始终拥有调度优先级。
- visual_event 主目标及匹配人体增加 `held_object` 诊断，Viewer展示候选、关联手腕、
  物体标签、关联分、证据帧及拒绝原因；`VISION_TRACE` 同步记录时间差和
  实际/阈值手腕距离比。

## 1.3.5 — 2026-09-02

- Stop 手势增加相对肩髋的手腕高度门控：自然下垂的张开手掌不再触发，举到躯干
  指令区域的正对镜头手掌及高位相机下的弯肘手势仍受支持。
- 跳跃改为髋部与双脚共同上升、两帧确认的专用时序检测，并保持约0.65秒供
  3/5 帧稳定器和10 Hz状态流观察；脚踝着地的快速起身不再视为跳跃。
- GesturePose 调试新增 Stop 高度/区域分、髋/脚踝速度、跳跃证据/保持/冷却字段，
  Viewer 同步展示；正式 `/perception/visual_event` schema 和身份门控不变。

## 1.3.4 — 2026-09-02

- 撤销 Vision 内部的陌生人情绪细分：Vision 不再订阅 `/emotion/state`，陌生人脸
  统一只发布 `EVT_VISION_STRANGER`。
- `EVT_VISION_STRANGER_ALERT/FRIEND` 不再属于 Vision 的可发布事件；Behavior Tree
  分别消费通用陌生人事实和情绪状态，在下游完成组合判断、优先级和去重。
- 同步更新运行配置、接口清单、联调手册和测试工程师验收标准。该版本取代 1.2.0
  中“Vision 细分陌生人事件”的旧设计。

## 1.3.3 — 2026-09-02

- 人脸、姿态、事件和物体调试合并到 `vision_debug.launch.py` 与同一 Web 页面；
  该 launch 默认同时启动正式视觉节点和 Viewer，并提供 `start_vision_node:=false`
  以附加到已经运行的视觉节点。
- 页面新增持续物体识别启动/停止按钮，使用固定 `vision-debug-web` session 和30秒
  租约；不会抢占 Action 的 session。删除独立 `object_debug.launch.py` 启动入口。

## 1.3.2 — 2026-09-01

- Vision Debug 页面新增带置信度参数的“单次物体识别”按钮，通过 `detect_objects`
  检测最新相机帧；该操作不调用 `set_object_detection`，不会创建、续租或抢占
  Action 持有的持续物体识别 session。

## 1.3.1 — 2026-09-01

- 普通姿态、跌倒和 Stop 手势统一增加固定人脸库身份门控：当前主目标必须是
  `owner/family_member_1～4`、仍为 `tracking`，且达到 `confirmed_known`，才写入正式
  `/perception/visual_event.events[]`。
- 身份未确认时仍保留结构化姿态字段和 GesturePose 调试 Topic，便于区分“模型已
  识别但事件被身份门控”与“模型没有识别到动作”。人脸、陌生人和物体事件规则不变。

## 1.3.0 — 2026-09-01

- Vision 新增与声纹一致的样本级 Face FastAPI：固定 `owner/family_member_1～4`
  五个身份、每人最多5张，提供 POST/GET/PUT/DELETE、单张 JPG 查看和 Swagger。
- 人脸样本采用稳定 `sample_id=1～5`；删除不重编号，新增复用最小空闲编号，
  新增/替换/删除后同步运行时多模板索引。
- 当前远程人脸管理暂不鉴权，只允许部署在可信隔离局域网；旧自由姓名数据保留在
  设备本地但不再加载。

## 1.2.0 — 2026-08-12

- Vision 新增 RELIABLE depth 10 的 `/emotion/state` 订阅，仅用于陌生人脸事件
  细分；不订阅当前规则未使用的 `/internal_need/state`。
- 新增 `EVT_VISION_STRANGER_ALERT`：陌生人脸存在且新鲜状态中 Anxiety 或 Fear
  已达到情绪系统阈值。
- 新增 `EVT_VISION_STRANGER_FRIEND`：没有 Alert 情绪，且 Joy、Excite 或 Calm
  已触发；Alert 在正负情绪同时触发时优先。
- 情绪状态尚未收到、超过 2.5 秒、schema/字段不合法或只有 Curious 触发时，
  继续发布 `EVT_VISION_STRANGER`。细分事件替换通用事件，避免双行为候选。
- 明确细分事件不得再次配置为情绪增量输入，避免 10 Hz 视觉状态流形成反馈环。

## 1.1.0 — 2026-08-12

- 扩充视觉权威契约：完整列出 visual_event 字段、25 个 GesturePose 标签到兼容
  动作和正式事件的映射、9 个当前可发事件、4 个仅预留事件，以及重复/去重语义。
- 正式物体检测从固定 1 Hz 改为默认关闭；`object_debug.launch.py` 仍保留固定
  1 Hz，避免影响独立模型调试。
- `VisionTask` 新增 `set_object_detection` 和 `get_object_detection_state`；
  Action 使用唯一 `session_id` 开启、续租、更新和关闭数据流，不同 session
  不能抢占或停止当前所有者。
- `/perception/vision/object_detections` 升级为 schema v2，增加 `stream`、
  `request`、`stop_reason`；显式停止或租约到期会发布终态并清除旧缓存。
- `target_labels` 对 RKNN 已有类别执行不区分大小写的精确过滤；不宣称运行时
  动态扩展 RKNN 开放词汇。
- 固化职责边界：视觉只发布事实数据；寻物搜索、靠近、丢失处理、Nav2 和
  `/cmd_vel` 全部由 Action 系统负责。

## 1.0.3 — 2026-08-12

- `face_covering` 收紧为当前帧的双手掩面：必须同时存在可靠鼻尖/双肩人体锚点，
  且左右两只手都由 HandLandmarker 实际检出并靠近面部；单个手掌或 Pose 猜测
  的手腕不再单独触发。
- 人体或第二只手离开画面时立即清除 `face_covering` 和 `hands_on_head` 的历史
  平滑票，避免把上一时刻的动作标签附着到当前画面中唯一的手上。
- `fast_nod` 增加当前头肩结构和 YuNet 人脸双重门控；明确没有人脸的帧不再
  进入鼻尖速度、反转和位移历史，并立即清除旧点头票，避免移动手掌被 Pose
  错拟合后触发快速点头。Viewer 增加“当前帧人脸确认”诊断。
- 物体 Provider 改由 `ultralytics.YOLOE` 负责预处理、RKNNBackend、NMS 和
  `Results.boxes` 解码；ROS2 Topic/Service 字段保持兼容。

## 1.0.2 — 2026-08-11

- 增加 `/perception/vision/object_detections`：物体检测默认按 1 Hz 定时推理，
  Service 查询结果也发布到同一 Topic；RKNN 定时任务与 Service 串行执行。
- Viewer 改为默认订阅物体 Topic，不再额外以 1 Hz 轮询 VisionTask。
- 相机回调改为 latest-frame 投递，Pose/Hand/YuNet 在独立单工作线程串行执行；
  忙碌时覆盖旧候选帧，不再阻塞相机接收和 10 Hz 事件发布。
- ROS 相机、事件定时器与 VisionTask Service 使用独立回调组和多线程执行器；
  物体检测 Service 不再暂停相机回调与视觉事件定时器。
- Web Viewer 默认限制 8 FPS、0.75 渲染比例、JPEG 质量 75，并默认关闭完整
  `/perception/vision/debug_image` 复制发布；页面分别显示输入与渲染 FPS。
- Gesture 调试诊断增加接收帧、候选帧、完成帧和 pending-frame 替换计数。
- 单目标模式将 Pose 最大人数设为 1；Hand 空闲时隔次探测，检测到手后恢复逐次
  推理并保持 8 次，减少没有手势时约一半的 Hand Landmarker 调用。

## 1.0.1 — 2026-08-11

- 当前视觉输入统一为外部 RealSense 640×480 单目彩色流，修正文档中残留的
  640×240 双目说明。
- 相机断流 0.5 秒后不再发布缓存场景，且发布定时器不再刷新目标年龄。
- 姿态/手势切换为确定性 GesturePose 时序规则；静态躺卧不产生跌倒事件。
- 真实 Provider 失败时不再自动返回固定人体或固定物体，Mock 必须显式配置。
- `detect_objects` 支持 JSON object 的 `confidence`，结果短暂镜像到
  `tracked_objects[]`；VisionTask 开始填充 `latency_ms`。
- 人脸注册完成后结束会话，删除后同步清理内存识别库，并拒绝路径型名称。
- 增加本地 Web 调试仪表盘，叠加人体/姿态骨架、人脸、手势、物体和
  ActiveTarget；同时显示 Topic 新鲜度、原始 JSON，并可定时调用物体识别。
- 真实人脸/姿态/手部流水线默认设置 `inference_frame_stride=2`，隔帧推理，
  30 Hz 相机下推理上限约 15 Hz，同时保留逐帧相机断流检测和 10 Hz 事件发布。
- 增加 `/perception/vision/gesture_debug`，Viewer 显示精确动作标签、优先级、
  原始25项候选分数和时序平滑命中结果，不改变正式视觉事件字段。
- 修复鼻尖关键点抖动反复触发 `fast_nod`：在速度和方向反转之外增加相对肩部的
  归一化垂直位移幅度门槛，并在 Viewer 中展示三项点头诊断量。
- 修复双臂交叉过程误判 `clapping` 以及 `hands_on_hips` 漏检：鼓掌要求重复
  开合，交叉姿态增加腕部换侧关系；腰部遮挡时允许较低置信度的腕髋关键点，
  并使用2D/3D肘角中更可靠的弯曲证据。Viewer 同步显示双手间距时序量。
- Pose/Hand Landmarker 从逐帧 `IMAGE` 推理切换为带单调时间戳的 `VIDEO`
  模式；增加 Lite/Full 启动参数和滚动A/B指标，包括有效FPS、平均/P95耗时、
  人体/手部检测率及关键点有效率。正式视觉事件契约不变。

## 1.0.0 — 2026-08-04

首个正式归档基线：

- 固化语音、视觉、行为树、动作系统的职责边界和 ROS2 主链路。
- 记录 Topic、Service、Action 类型、QoS、JSON schema 和启动顺序。
- 明确跟随是语音会话生命周期内的视觉闭环，不是固定 Twist 动作。
- 明确行为树使用延迟优先级队列，充电途中低优先级情绪行为只排队。
- 明确 `/behavior/result_event` 由行为树根据 Action Result 发布。
- 补齐 `ExecuteBehavior.Result.metadata_json` 和充电 `energyValue` 契约。
- 修正双目选单眼后的坐标说明：坐标相对单眼推理画面归一化到 `[0,1]`。
- 记录当前 ROS2 运行时的唤醒处理：开启 `face_body_centering`，不下发
  `respond_owner_call` Action；旧映射只作为 standalone/兼容路径保留。

后续每次跨项目接口变更必须在此增加版本、日期、影响项目、兼容性和迁移方法。

### 行为树运行时补充

- CandidatePool 增加按 `behavior_name + candidate_id` 的 in-flight reservation。
- 同名 queued/in-flight 候选禁止重复注入，`allow_repeat` 只在终态后生效。
- 成功、失败、超时、取消、抢占和未 dispatch 路径都会释放 reservation。
