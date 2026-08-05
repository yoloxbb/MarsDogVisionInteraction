# 迁移说明

来源为 `20260622_MarsDogPro/marsdog_perception` 当前工作区状态。

- `camera_driver_node.py` → 视觉项目
- `vision_observation.py`、`object_detector.py`、人脸与姿态 Provider → 视觉项目
- `TargetManager` → 改写为不接收音频的 `VisualTargetManager`
- `EnrollmentManager` → 仅保留 `FaceEnrollmentManager`
- `PerceptionBridgeNode` → 仅提取相机、视觉发布和视觉任务处理
- 原 `registry.json` → 独立 `face_registry.json`

刻意未迁移：唤醒、VAD、ASR、声纹、意图、语音状态机、跨模态
`target_tracker`、通用 HTTP 网关和 Web 面板。
