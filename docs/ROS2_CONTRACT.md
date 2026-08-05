# 视觉 ROS2 契约

## `/perception/visual_event`

- 类型：`std_msgs/String`
- 编码：UTF-8 JSON
- QoS：BEST_EFFORT、KEEP_LAST、depth 5
- 默认频率：10 Hz
- `schema_version`：1

顶层字段：

```text
schema_version
header
active_target
faces[]
humans[]
hands[]
tracked_objects[]
events[]
```

本 Topic 只表达视觉事实。`speaker_id`、`is_speaking` 和
`speaker_confidence` 仅为旧消费者兼容占位，始终使用默认值。跨模态关联由
`/perception/target_event` 的消费者完成。

坐标均相对实际参与推理的画面归一化到 `[0, 1]`。当前 640×240 双目横拼
输入只选择左侧 320×240 画面执行二维推理，因此左目画面中心仍为 `x=0.5`；
消费者不能再按原始 640 像素宽度解释这些坐标。

完整字段、目标有效性和跨项目消费规则见
[多项目 ROS2 接口契约](integration/ROS2_INTERFACES.md)。

## `/perception/vision/task`

类型：`marsdog_vision_interaction/srv/VisionTask`。`params_json` 接受 JSON
对象；结果通过 `result_json` 返回。视觉项目不处理声纹或 ASR 任务。
