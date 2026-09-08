# 三任务固定协议 v1

状态：三分支、双边界 GPU smoke 与 seed 61001 正式实验完成，结果审计通过；未扫描权重或阈值。结果见 [三任务报告](three_tasks_v1_results.md)。

- 顺序：Pong → Breakout → SpaceInvaders；共享 backbone，任务独立 actor/critic head，已知 task ID。
- 方法：普通 PPO 微调、原 EWC（λ=0.4，100 个样本的策略经验 Fisher）、硬投影 GPM（阈值 0.995，每个卷积观察 16 个 patch）。不加入其他机制。
- 复用 `outputs/policy_replay_v1/seed-61001/anchor.pt` 的 Pong 起点（1,048,576 transitions，10 回合均分 11）。重新训练两项后续任务，各 524,288 transitions。
- 8 个异步环境，rollout 128，4 epochs，minibatch 32，编译 PPO；不改变原 PPO 和 EWC 代码。先创建相同的第三任务 head，使用隔离 RNG；各方法恢复相同 Breakout 初始 RNG，第三任务统一 seed+2000，但其 backbone/Adam 状态是各自训练结果。
- 每 131,072 transitions 用固定种子、确定性动作、10 个完整回合、原始奖励评估已见任务；阶段边界评估全部三个任务，形成完整 3×3 分数矩阵。未学任务只作前向参考，不与已学任务合并汇总。
- 分别报告各游戏保留程度、Breakout 与 SpaceInvaders AUC、GPM 每层累计保护维数及剩余维数、实际 Adam 位移投影误差。不跨游戏平均原始奖励。
- GPM 起始用原 Pong 的 2,048 个状态。后续任务结束后，用冻结当前策略额外采集 32,768 transitions，从中固定随机抽取 2,048 个状态；仅用于边界子空间估计，不用于 RL 更新。边界采集开销与训练预算分开记录。最终阶段也保存更新后的基，供检查容量和后续续接。
- 累积遵循 [GPM 原文 §5，式 8–9](https://arxiv.org/html/2103.09762)：先去掉已有基解释的方向，再选择最少残余方向，使旧基与新增基合计解释当前任务至少 99.5% 输入能量。不是对残余本身再保留 99.5%。旧保护空间不覆盖、不遗忘，允许新增 0 维。
- EWC 在每次任务结束后按原方法用最后一个 PPO rollout 累积 Fisher。保存完整 EWC checkpoint、最后 rollout 与 GPM 基；学习第三任务时同时保护前两个任务。
- 所有输出写入 `outputs/three_tasks_v1/seed-61001`；原两任务证据与归档不改动。保存配置、源码 SHA-256、源码快照、历史回合分数与最终 checkpoint。单 seed 仅作当前协议下的描述性检验，不做显著性或普适性结论。

运行：

```bash
direnv exec . python scripts/run_experiments.py train three-tasks \
  --study-config experiments/three_tasks_v1.json --seed 61001 \
  --log-dir logs/three_tasks_v1
```
