# 活跃持续学习方案

只维护原 EWC 和 GPM；保留 PPO/DQN、单任务、普通顺序训练和联合训练作为教学及比较基线。

- **EWC**：`algorithms/ewc.py` 原实现未改动，继续使用 `scripts/run_experiments.py train continual --ewc-mode on`（完整选项见 `--help`）。它是参数稳定性基线。
- **GPM**：`algorithms/subspace_projection.py` 与 `scripts/study_gpm.py`。共享层的实际 Adam 位移投影到旧输入子空间的正交补，偏置纳入常数输入坐标；旧任务 head 保持不变。现已补上任务边界的子空间累积，并扩展到 Pong → Breakout → SpaceInvaders；原两任务入口保留。不声称完整复现原论文。

GPM 已解除对策略回放类和退役脚本的导入依赖。旧状态只用于构建子空间，两任务诊断中的 teacher logits 只用于评估策略漂移，不参与回放训练；三任务实验直接评估各任务回报。共享 PPO 运行时保持冻结实验版本，便于核对历史行为。

## 两任务入口

```bash
direnv exec . python scripts/run_experiments.py train gpm \
  --study-config experiments/gpm_v1.json --seed 61001 --dry-run
```

去掉 `--dry-run` 才会启动训练。此命令对应两任务协议，三任务运行命令见下方。配置继续读取 `outputs/policy_replay_v1/seed-61001/anchor.pt` 及其 manifest/源码快照，输出到新的 `outputs/gpm_v1/`，不会覆盖已有 `outputs/subspace_v1/`。这依赖已存在的冻结 anchor；从零创建该历史 anchor 的代码保存在归档中。当前仅使用 seed 61001。

## 两任务历史结果

同一 Pong 边界（均分 11），新任务 524,288 transitions，固定 seed 61001，确定性完整回合、原始奖励：

| 方案 | Pong 最终 | Breakout 最终 | Breakout AUC / budget |
|---|---:|---:|---:|
| 原 EWC | -6.2 | 8.7 | 7.3125 |
| GPM | 9.8 | 9.5 | 5.2625 |

GPM 在这次运行中更好地保留旧任务，但新任务 AUC 低于 EWC，不能称为全面解决任务冲突。原始数据见 `outputs/policy_replay_v1/seed-61001/ewc/` 和 `outputs/subspace_v1/seed-61001/gpm/`。上述数值是既有实验结果，不是整理后重跑的结果。

其余探索的代码、协议、结果说明和证据路径见[归档索引](../archive/continual_learning_2026-09-07/README.md)。

## 三任务扩展

[固定协议](three_tasks_v1.md)与 `three_tasks_v1.json` 使用同一个 Pong 起点，比较普通 PPO、原 EWC 与 GPM，重新训练 Breakout 后继续训练 SpaceInvaders。每个后续任务 524,288 transitions，只用 seed 61001。

```bash
direnv exec . python scripts/run_experiments.py train three-tasks \
  --study-config experiments/three_tasks_v1.json --seed 61001 \
  --log-dir logs/three_tasks_v1
```

GPM 在旧基的正交补中提取必要的新方向；EWC 原实现继续累积每个任务的 Fisher。结果写入独立的 `outputs/three_tasks_v1/`，不覆盖两任务历史。

三任务实验已完成，完整阶段矩阵、AUC、子空间维数和审计见[三任务结果](three_tasks_v1_results.md)。本轮 GPM 的三个终点最好，但两个新任务 AUC 均较低；结论仅限该 seed、任务顺序与预算。
