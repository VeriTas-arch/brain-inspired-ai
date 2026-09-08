# 边界策略期望 Fisher：诊断及最多一个训练对照

仅 seed 61001；使用原 Pong anchor 和训练用 memory 的全部 2048 条旧状态。独立 probe 不用于 Fisher、尺度校准或诊断准入。保持既有实验与源码快照。

## 固定估计与尺度规则

比较三种共享 backbone 的对角 Fisher：(1) 原 EWC 在最后 rollout 上的 100 个状态-动作样本；(2) memory 上每个状态从边界策略抽取一个动作的 MC 估计；(3) 同一 memory 上对全部 6 个动作按边界策略概率加权的期望估计。通过逐样本 log-policy Jacobian 求平方期望，不使用批平均梯度的平方，不使用 advantage、return 或 critic 梯度。状态期望和对角近似仍然存在。

固定私有 RNG 为 seed+50000，将 memory 随机排列并分为两个 1024-state 半集；记录期望与 MC 的分半余弦一致性、每层 trace、非零比例、两估计间余弦与相对误差。原估计保持原实现的采样规则。

如进入训练，使用 F_expected * trace(F_original)/trace(F_expected)，只匹配真正受保护的共享 backbone 总 trace，EWC lambda 保持 0.4。此规则控制各向同性扰动的平均二次惩罚尺度，不声称匹配实际 PPO 轨迹上的全部约束强度。不扫描 lambda、不改变架构或 PPO 超参数。

## 小扰动诊断与准入

仅在 memory 中固定抽取 256 条状态。私有 RNG 为 seed+51000。对共享 backbone 构造 8 个全局高斯方向及 4 个单层高斯方向（每个 Conv/Linear 一组、含偏置）；每个方向归一化到该组原参数 L2 范数，扰动半径固定为 0.002、0.005、0.01。比较真实 KL(pi_anchor || pi_perturbed) 和 0.5 * sum(F * delta^2)。使用 float64 策略网络进行这一局部诊断，防止很小 KL 的数值抵消。记录所有扰动点，不选择有利方向或半径。

原始预测误差为中位绝对 log(predicted KL / actual KL)。去除整体尺度后的形状误差为上述 log 比值减去其中位数后的中位绝对值；它仅用于诊断，不反向调整训练权重。

准入条件事先固定：期望 Fisher 有限且非负、总量为正；其分半余弦不低于同状态 MC；其形状误差相比原 100-sample Fisher 至少降低 10%。任何一点不满足，就报告诊断结果并结束本候选，不进行长训练。不根据独立 probe 或游戏评估改变准入条件。

## 若准入：单 seed 训练

只新增 ewc_expected 分支。起点 backbone、所有 heads、已有 Adam 状态、随机数起点均来自原 anchor；新任务首批 rollout 哈希必须与原实验一致。使用原共享 PPO runtime、8 async envs、compile PPO、rollout 128、4 epochs、minibatch 32、Breakout 524288 transitions；每 131072 transitions 对每任务评估 10 个完整确定性 raw-reward 回合，max_steps 30000。先做独立目录的 1024-transition smoke，每任务 1 回合，只验证运行，不用于效果选择。

报告相对原 EWC 与 GPM 的 Pong 终点、Breakout 终点、AUC/budget、策略 KL、成本。相对 GPM 三个成绩指标均不降低、且至少一项 Breakout 指标提高，才记为本实例完整改善；否则报告折中或无收益。这是已暴露单 seed 的描述性判断，不是统计优越性证明。失败后不更换 Fisher 公式或权重重试。

依据：[On the Computation of the Fisher Information in Continual Learning](https://iclr-blogposts.github.io/2025/blog/fisher/)。本实验只检查估计质量及其有限预算效果，不预设 EWC 能解决真实任务冲突。
