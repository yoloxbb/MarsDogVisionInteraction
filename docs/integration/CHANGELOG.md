# 对接归档变更记录

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
