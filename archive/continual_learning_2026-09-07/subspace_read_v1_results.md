# GPM + 单层对角子空间缩放实验结果

未达到预先规定的本实例完整改善条件；保留 GPM 为主候选，不扩大或调参重试这个变体。

单 seed 61001，与已有 GPM 使用同一 Pong 起点、新任务 heads、已有 Adam 状态、随机数起点、子空间和训练评估预算。仅新增最后共享全连接层 ReLU 前的 939 维任务专属缩放，零初始化；使用原 PPO 损失。缩放只影响 Breakout，Pong 仍使用共享 GPM 路径。

| 方法 | Pong 最终 | Breakout 最终 | Breakout AUC / budget | 旧策略 KL | 更新秒数 |
|---|---:|---:|---:|---:|---:|
| gpm | 9.80 | 9.50 | 5.2625 | 0.058788 | 242.1 |
| gpm_read | 6.20 | 7.60 | 5.0500 | 0.046677 | 244.1 |

相对 GPM 的 Pong、Breakout 终点及 AUC 差值依次为：-3.6000、-1.9000、-0.2125。

![comparison](/home/veritas/Documents/Atari_Playground/outputs/subspace_read_v1/comparison.png)

这是已暴露单 seed 的描述性比较，不代表统计优越性。完整改善条件要求三个成绩指标均不下降，且至少一个 Breakout 指标提高。未把旧策略 KL、缩放非零或训练 loss 改善替代任务成绩。这里只检验两任务边界，不宣称长任务序列有效。

## 实现和资源

新增可训练参数 939 个；当前实现额外缓存输入子空间和固定读取字典，共 13.071 MiB，另有缩放参数与其 Adam 状态。因而参数增量不等于全部存储增量。共享投影原有存储另计。

缩放向量最终范数 2.156506，最大绝对值 0.253246；完成 65536 次 Adam 更新。

候选 CUDA peak allocated/reserved：155.89/246.00 MiB；含评估总墙钟 964.9 秒。更新时间包含编译预热与诊断；未作为独立吞吐基准。

## 审计

audit.json 核验配置、源码快照与起点哈希、首批 rollout 相同、子空间逐位一致、准确 transition 和 optimizer-step 数、每任务评估回合数、原始分数重算、旧 heads 不变、固定读取字典、共享投影误差，以及最终 checkpoint 恢复后的任务隔离和策略 KL。

方法依据是 [TRGP](https://arxiv.org/abs/2202.02931) 的 scaled weight projection；本候选只取单层对角缩放，不是完整 TRGP 或其 PPO 效果复现。[冻结协议](subspace_read_v1.md)。
