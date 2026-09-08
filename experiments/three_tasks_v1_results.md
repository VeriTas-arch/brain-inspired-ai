# 三任务实验结果（seed 61001）

复用已学好的 Pong 起点，重新训练 Breakout 与 SpaceInvaders，各 524,288 transitions。每项分数为固定 10 个完整回合的确定性原始奖励均值。只作此单 seed、顺序和预算下的描述性比较。

本轮 GPM 的三个最终分数均最高：Pong 9.2、Breakout 9.0、SpaceInvaders 307。Pong 相对最初学成时下降 1.8 分，Breakout 相对学成时下降 1.1 分；增加第三任务后仍保留了前两个任务的大部分表现。

但 GPM 的 Breakout AUC 为 5.3125，SpaceInvaders AUC 为 230.375，均低于本轮普通 PPO 和 EWC。结论是本协议下终点兼顾更好，不能说全面改善了学习速度或消除了稳定性与可塑性的取舍。

EWC 的 Pong 表现在阶段间和第三任务内部出现过明显恢复与回落；遗忘不是单调的。应同时阅读阶段矩阵、原始 history.json 和新任务曲线，不能把单个检查点解释为知识永久丢失。

| 方法 | 完成阶段 | Pong | Breakout | SpaceInvaders |
|---|---|---:|---:|---:|
| finetune | Pong-v5 | 11.00 | 0.00 | 100.00 |
| finetune | Breakout-v5 | -20.60 | 8.40 | 145.00 |
| finetune | SpaceInvaders-v5 | -21.00 | 0.40 | 265.00 |
| ewc | Pong-v5 | 11.00 | 0.00 | 100.00 |
| ewc | Breakout-v5 | -20.80 | 11.70 | 134.50 |
| ewc | SpaceInvaders-v5 | -1.50 | 3.50 | 232.50 |
| gpm | Pong-v5 | 11.00 | 0.00 | 100.00 |
| gpm | Breakout-v5 | 9.50 | 10.10 | 145.00 |
| gpm | SpaceInvaders-v5 | 9.20 | 9.00 | 307.00 |

未学任务的分数仅用于观察前向变化。不同游戏原始分数不合并平均。本轮统一比较重新运行的三任务结果，不混用旧两任务终点；现有 seed_everything 固定随机种子但不强制确定性 GPU 内核，首批 rollout 一致不等于整个训练轨迹逐位相同。

| 方法 | Breakout AUC / budget | SpaceInvaders AUC / budget | Pong 最终减学习后 | Breakout 最终减学习后 |
|---|---:|---:|---:|---:|
| finetune | 7.2750 | 247.5000 | -32.00 | -8.00 |
| ewc | 7.2625 | 237.8750 | -12.50 | -8.20 |
| gpm | 5.3125 | 230.3750 | -1.80 | -1.10 |

## GPM 累计保护维数

| 层 | 输入维数 | Pong 后 | Breakout 后 | SpaceInvaders 后 |
|---|---:|---:|---:|---:|
| 0 | 257 | 3 | 20 | 97 |
| 2 | 513 | 42 | 133 | 371 |
| 4 | 577 | 266 | 415 | 461 |
| 7 | 3137 | 939 | 1682 | 2024 |

第三任务训练使用的是 Breakout 后的累计基；SpaceInvaders 后的基在最后评估结束后才构建，供后续任务续接，不参与本轮 SpaceInvaders 训练。各层尚有剩余维数，但维数本身不能证明任意后续任务都可兼容。

## 运行开销

| 方法 | 任务 | 训练秒数（采集、更新与编译） | 训练与评估墙钟秒数 | GPM 边界采集与构基秒数 |
|---|---|---:|---:|---:|
| finetune | Breakout-v5 | 178.8 | 709.2 | 0.0 |
| finetune | SpaceInvaders-v5 | 177.6 | 628.3 | 0.0 |
| ewc | Breakout-v5 | 206.4 | 795.3 | 0.0 |
| ewc | SpaceInvaders-v5 | 190.1 | 801.2 | 0.0 |
| gpm | Breakout-v5 | 235.4 | 784.5 | 8.1 |
| gpm | SpaceInvaders-v5 | 233.6 | 762.8 | 8.2 |

GPM 每个新任务边界额外采集 32,768 transitions，只用于构基，不计入 PPO 更新预算；墙钟列不含此边界开销，也不含 checkpoint 写入和 EWC Fisher 估计。

![comparison](/home/veritas/Documents/Atari_Playground/outputs/three_tasks_v1/comparison.png)

审计通过：源码与 anchor 哈希、精确训练检查点、回合数、AUC、旧 head 不变、EWC 多边界状态、GPM 基的嵌套与正交性、实际位移投影误差、Breakout 首批 rollout 一致。见输出目录 audit.json。

[冻结协议](three_tasks_v1.md)。结果不能证明一般容量冲突，也不能替代其他任务顺序或 seed 的验证。
