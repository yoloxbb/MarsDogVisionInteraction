# MarsDog Vision Interaction

> 项目交接请先阅读 [docs/HANDOFF.md](docs/HANDOFF.md)。四项目总览、统一接口清单
> 和联调手册位于 [docs/integration/README.md](docs/integration/README.md)。

独立的 MarsDog 视觉交互 Python/ROS2 项目，负责相机、视觉推理、人脸注册和
`/perception/visual_event`。本项目不导入语音项目，也不维护 VAD、声纹或语音会话状态。

## 环境

- Python 3.10
- uv
- ROS2 Humble

推理模型继续统一放在 `/home/cat/xbb/models/vision`，避免在多个项目中复制大文件；
项目自己的配置、人脸注册表和人脸样本已随视觉项目迁移。

```bash
cd /home/cat/xbb/MarsDogVisionInteraction
uv sync --extra models --extra dev
source /opt/ros/humble/setup.bash
uv run pytest
```

直接运行源码节点：

```bash
source /opt/ros/humble/setup.bash
uv run marsdog-vision-interaction \
  --ros-args -p config_path:=config/vision.yaml
```

ROS2 构建：

```bash
source /opt/ros/humble/setup.bash
cd /home/cat/xbb/MarsDogVisionInteraction
colcon build --base-paths . --packages-select marsdog_vision_interaction
source install/setup.bash
ros2 launch marsdog_vision_interaction vision.launch.py
```

该 launch 只启动视觉节点，并订阅已经存在的 RealSense 彩色流；相机驱动需
单独启动。

将 `providers.vision.type` 设置为 `mock` 可以在没有模型和相机的环境中做下游联调。

## ROS2 接口

- 订阅：`/camera/camera/color/image_raw`
- 发布：`/perception/visual_event`

## 跟随调试画面

另开终端启动可视化节点：

```bash
cd ~/xbb/MarsDogVisionInteraction
uv run marsdog-vision-viewer
```

窗口会显示实际输入分辨率、人体框（绿）、人脸框（蓝）、当前目标框与躯干
控制点（红）、中心滞回区、跟随模式及实时 `/cmd_vel`。按 `q` 或 `Esc` 关闭
窗口。节点同时发布 `/perception/vision/debug_image`，无桌面环境时可使用
`rqt_image_view /perception/vision/debug_image` 在远程桌面查看；也可传入
`-p show_window:=false` 只发布调试图像。

视觉与调试节点默认订阅 RealSense 彩色流
`/camera/camera/color/image_raw`，并按普通单目画面处理。
- 发布：`/perception/vision/enrollment_event`
- Service：`/perception/vision/task`

视觉任务：`check_person`、`detect_objects`、`recognize_face`、
`start_face_enrollment`、`cancel_face_enrollment`、`upload_face`、
`list_faces`、`delete_face`。

完整字段约定见 [docs/ROS2_CONTRACT.md](docs/ROS2_CONTRACT.md)。
