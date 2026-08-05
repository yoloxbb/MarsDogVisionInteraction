# MarsDog 多项目对接归档

> 归档版本：`1.0.0`
>
> 基线日期：`2026-08-04`
>
> 适用项目：语音、视觉、行为树、动作执行器

本目录是四个独立项目之间的集成入口。各负责人可以独立开发和测试自己的项目，跨项目协作只依赖这里列出的 ROS2 接口，不应直接导入其他项目的 Python 内部模块。

## 1. 系统边界

```text
麦克风/唤醒板 ──> Voice ── /perception/audio_event ───────┐
                                                          │
摄像头 ──> Vision ── /perception/visual_event ───────┐    │
                    /perception/vision/task <─────┐   │    │
                                                  │   ▼    ▼
Emotion/InternalNeed ── state + signal_event ──> Behavior Tree
                                                     │
                     /behavior/attention_tracking ───┼────────┐
                     /execute_behavior Action ───────┘        ▼
                                                       Action Executor
                                                             │
                                   /cmd_vel、Nav2、仿真/动作控制器

Action Result ──> Behavior Tree ── /behavior/result_event ──> InternalNeed
```

关键边界：

- 语音项目只负责听见什么、识别出什么意图以及会话何时结束，不直接控制底盘。
- 视觉项目只发布视觉事实和稳定目标，不决定执行哪个行为。
- 行为树项目负责事件映射、优先级、延迟队列、抢占、超时和会话跟踪模式。
- 动作系统只接受语义 `behavior_name`，解析为 `ACT_*`，并控制 AGV/Nav2/仿真。
- `/behavior/result_event` 由行为树发布，不由动作系统直接发布；行为树会把 Action Result 转成内部需求系统需要的数据。

## 2. 仓库与责任人交接入口

| 子系统 | 负责人 | 仓库 | 交接文档 | 对外主接口 |
|---|---|---|---|---|
| 语音 | 待填写 | `/home/cat/xbb/MarsDogVoiceInteraction` | `docs/HANDOFF.md` | `/perception/audio_event`、`/perception/voice/task` |
| 视觉 | 待填写 | `/home/cat/xbb/MarsDogVisionInteraction` | `docs/HANDOFF.md` | `/perception/visual_event`、`/perception/vision/task` |
| 行为树 | 待填写 | `/home/cat/xbb/20260702_MarsDogTree` | `docs/HANDOFF.md` | `/execute_behavior` Client、`/behavior/result_event` |
| 动作 | 待填写 | `/home/cat/xbb/20260707_MarsDogAction` | `docs/HANDOFF.md` | `/execute_behavior` Server、`/cmd_vel` |
| 公共 Action 接口 | 待填写 | `/home/cat/xbb/marsdog_interfaces` | `action/ExecuteBehavior.action` | `marsdog_interfaces/action/ExecuteBehavior` |

确定人员后应立即补齐负责人和联系方式；跨项目接口变更由生产者与所有直接消费者共同评审。

## 3. 本归档内容

- [ROS2_INTERFACES.md](ROS2_INTERFACES.md)：跨项目 Topic、Service、Action、JSON 字段和 QoS。
- [RUNBOOK.md](RUNBOOK.md)：构建、启动顺序、联调命令、验收与排障。
- [interface_manifest.yaml](interface_manifest.yaml)：供脚本或 CI 使用的机器可读接口清单。
- [CHANGELOG.md](CHANGELOG.md)：归档版本变化和兼容性说明。
- 各仓库 `docs/HANDOFF.md`：面向该项目负责人的输入、输出、配置和测试说明。

## 4. 契约的唯一来源

| 内容 | 权威来源 |
|---|---|
| `ExecuteBehavior` 字段 | `marsdog_interfaces/action/ExecuteBehavior.action` |
| 语音 Service 字段 | `MarsDogVoiceInteraction/srv/VoiceTask.srv` |
| 视觉 Service 字段 | `MarsDogVisionInteraction/srv/VisionTask.srv` |
| 语音 JSON | `MarsDogVoiceInteraction/messages/audio_event.py` |
| 视觉 JSON | `MarsDogVisionInteraction/messages/visual_event.py` |
| 事件到 Behavior 的映射 | 行为树 `config/*.yaml` |
| Behavior 到 `ACT_*` 的映射 | 动作系统 `config/behavior_tree_actions.yaml` |
| AGV 动作参数 | 动作系统 `config/agv_motion_groups.yaml` 及 launch 参数 |

当本文档与权威源码冲突时，以权威源码为准，同时必须在同一次合并中更新本文档和 `interface_manifest.yaml`。

## 5. 版本与变更规则

接口变更按语义化版本管理：

- PATCH：只补充说明或增加可选字段，旧消费者仍能运行。
- MINOR：增加 Topic、任务类型、事件类型或 Behavior，旧接口仍保留。
- MAJOR：改名、删除字段、修改字段含义、QoS 不兼容或状态机语义变化。

任何跨项目变更至少需要：

1. 修改权威接口/配置和本归档。
2. 给出一条可复制的发布或调用示例。
3. 验证生产者与消费者的 QoS 匹配。
4. 在变更说明中标明兼容窗口和回滚方法。
5. 通知所有受影响仓库负责人，不允许仅在群聊中口头变更契约。

## 6. 当前必须共同遵守的约束

- 所有 JSON Topic 使用 UTF-8，顶层必须是 JSON object。
- 时间戳统一使用 Unix epoch 秒；角度字段必须明确度或弧度。
- 图像框坐标为 `[x, y, w, h]`，均相对视觉节点选中的单目画面归一化到 `[0,1]`。
- 行为优先级数值越小越高：Lv0 最高，Lv6 最低。
- 同名 Behavior 在 queued 或 in-flight 期间只能有一个实例；重复事件被抑制，
  不进入延迟重放。
- `interaction_id` 从一次唤醒持续到会话结束；同一句语音的事件共享 `utterance_id`。
- 跟随是会话级闭环控制，不是预录制的前进/摆动动作。
- 视觉数据超时或目标丢失时，动作系统必须发布零速度。
- 同一时刻只能有一个底盘速度控制链路生效；Action 执行期间会暂停后台视觉跟踪。
