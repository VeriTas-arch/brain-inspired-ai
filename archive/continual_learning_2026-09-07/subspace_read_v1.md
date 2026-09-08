# GPM + 单层对角子空间缩放：单 seed 协议

候选在最后共享全连接层、ReLU 前增加 W_Pong U diag(a) U^T；U 和 W_Pong U 固定，a 为 939 维、零初始化、仅 Breakout 使用。共享参数仍按 GPM 投影 Adam 实际位移，偏置同样纳入投影。Pong 不经过新增残差。新增参数使用原 Adam 默认超参数和 PPO 全局梯度裁剪，不修改已有参数的 Adam 状态。不叠加蒸馏、EWC、软投影或额外损失。

这是 TRGP（https://arxiv.org/abs/2202.02931）的单层对角简化候选，不是完整论文复现；不预设在 PPO 上有效。

仅 seed 61001。从 outputs/policy_replay_v1/seed-61001/anchor.pt 分叉，与已完成 GPM 共享 Pong 主干、新任务 heads、已有 Adam 状态、随机数起点和子空间构造数据。主干和 heads 初始输出必须逐位相同，首批 rollout 哈希必须一致，子空间必须与旧 GPM 逐位相同。

正式预算沿用原 manifest：Breakout 524288 transitions，8 async envs，rollout 128，4 epochs，minibatch 32，compile PPO。每 131072 transitions 评估一次，每个任务 10 个完整确定性回合，raw reward，最多 30000 agent steps。只新增 gpm_read 分支，不重跑历史分支、不增加 seeds。

报告相对 GPM 的 Pong 终点、Breakout 终点和 AUC / budget 的独立差值，以及旧策略 KL、额外参数、存储和实际耗时。三个成绩指标均不下降且至少一个 Breakout 指标提高，才记为本实例的完整改善；否则明确报告折中或无收益，不调整本候选后重试。这是已暴露单 seed 的描述性判断，不是统计优越性检验。参照值为 Pong 9.8、Breakout 9.5、AUC 5.2625。

先做单 seed 的 1024-transition smoke（每任务 1 回合），只验证运行、任务切换、投影和序列化，不用于效果判断。正式实验使用独立目录；保存配置、源码快照、起点哈希、逐回合评分、投影诊断和最终 checkpoint。恢复 checkpoint 后验证旧任务缩放隔离、固定字典不变、缩放确实训练、共享更新约束和指标重算。
