# 教学代码清理与当前默认配置复测

2026-09-09。本轮移除重复入口和无调用接口，缩短 PPO 更新链路，修复视频编码，并复测当前 8 环境教学配置。测速和 smoke 不代表正式训练收敛，也不替代新配置的学习得分。

## 保留与移除

- 删除 `scripts/demo.py`、`scripts/full_demo.py`、`scripts/visualize_gameplay.py` 和根目录 `test_framework.py`；Notebook、训练脚本、统一评估和 pytest 已覆盖这些入口。
- PPO 更新集中在 `PPOLearner → optimize_ppo`，删除两套只转发参数的 `Agent.update`、两套 `compute_gae` 及未调用的权重、策略 logits、缓冲就绪接口。GAE 与截断 bootstrap 的有效实现和测试保留。
- EWC 删除只有测试调用的独立 `compile_regularizer` 分支，惩罚项仍由完整损失一起编译。EWC 汇总统计与 GPM 的 Adam 位移投影保持原有语义。
- GPM 任务边界仅收集所选观察并生成动作，不再构造训练 rollout 或计算价值/log-prob。私有观察选择 RNG、外层训练 RNG 和任务头初始化保持隔离。
- 保留同步/eager、尾批、原生 ALE、完整 CUDA 更新图和 DQN 延迟指标；这些路径有实际调用或正确性/性能用途。双缓冲采样、回放预取和 EnvPool 候选此前已离开维护代码，不删除历史结果和冻结源码。
- `VideoRecorder` 改为逐帧写入 FFmpeg，检查退出码并传播编码错误；统一视频和分数的完整回合规则。训练录像按向量步取帧，避免环境数为偶数时每步都录像。移除不再直接使用的 `imageio` 依赖，保留 `imageio-ffmpeg`。

## 当前 8 环境流水线

下表为 seed 0 的三个独立进程重复的中位数，单位 transitions/s。PPO、DQN 的 minibatch 都为 32；不同算法的更新工作量不同，不能据此比较算法优劣。

| 算法 | 游戏 | async wrapper | 原生 ALE |
| --- | --- | ---: | ---: |
| DQN | Pong-v5 | 3,452 | 3,766 |
| DQN | Breakout-v5 | 2,983 | 2,758 |
| DQN | SpaceInvaders-v5 | 3,336 | 3,397 |
| PPO | Pong-v5 | 5,065 | 5,659 |
| PPO | Breakout-v5 | 4,236 | 3,908 |
| PPO | SpaceInvaders-v5 | 4,707 | 5,052 |

ALE 在 Pong 上更快，但没有在三个游戏上统一胜出，尤其 Breakout 更慢。它还使用不同的观测预处理协议，因此继续保留为显式选项，教学默认保持 8 个 async wrapper 环境。

| Pong 默认配置复测 | transitions/s | 预热秒 | allocated / reserved MiB |
| --- | ---: | ---: | ---: |
| 当前 DQN，compile + 完整更新图 | 3,417 | 1.56 | 35.6 / 172 |
| 当前 PPO，compile + 完整更新图 | 5,028 | 1.94 | 245.0 / 322 |
| DQN eager，8 async | 1,623 | 1.00 | 111.5 / 130 |
| 清理前 PPO，compile + 完整更新图 | 5,083 | 1.95 | 245.0 / 322 |

PPO 本轮主要是职责和重复代码清理，不把小幅吞吐差异宣称为性能提升。完整更新图的独立收益见此前的[学习器比较](learner_performance.md)。

## GPM 边界采样

每次收集 32,768 条 transition、选择 2,048 张堆帧观察；固定 8 环境、任务策略初始化与采样种子。预热对照先收集 1,024 条 transition，首次边界对照仅提前运行一次策略前向。三个游戏各路径均得到相同的观察和子空间 SHA256，且外层 CPU/CUDA RNG 恢复。表中 Pong 路径重复三次，其余游戏各一次。

| Pong 路径 | 收集秒，中位数 |
| --- | ---: |
| 原完整 rollout | 6.47 |
| 仅观察，仍计算价值/log-prob | 6.04 |
| 隔离编译候选，预热后 | 5.31 |
| 当前仅动作，首次边界调用 | 5.85 |
| 仅动作 + compile，首次边界调用 | 5.88 |

首次边界计时包含本进程首次调用/编译，允许已有磁盘编译缓存，不能解释为全新缓存机器的冷启动。编译候选的首次边界总耗时没有稳定优势，因此不加入维护代码；当前仅动作路径保持 eager。

## 视频与输出阶段

使用 3,000 帧 210×160 RGB 合成序列、30 FPS，三次重复并验证全部帧可解码。新编码使用 libx264 veryfast、2 线程、2 Mbps；旧实现失败后走备用编码器，两者压缩参数不同，不能解释为相同编码质量的比较。

| 录像实现 | 总时间秒 | 结束时等待秒 | 主进程峰值 RSS MiB |
| --- | ---: | ---: | ---: |
| 旧：保留全部帧 | 1.087 | 0.972 | 1089.0 |
| 新：流式编码 | 0.633 | 0.009 | 799.5 |

流式写入将编码工作移入采集期间，总耗时和内存均需计入，不能只比较最后的 close 时间。RSS 是 Python 主进程峰值，不含 FFmpeg 子进程。

使用最终 smoke 的 Pong checkpoint，独立评估 1 个完整回合，另录 1 个示例回合；下表为三次重复的中位数。分数阶段包含该调用需要的推理编译，checkpoint 保存计时不含模型构建/加载，也不包含强制落盘 fsync。

| 算法 | 分数回合秒 | 录像回合秒 | 绘图秒 | 评估总秒 | 保存 checkpoint 秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| DQN | 0.440 | 0.637 | 0.054 | 1.203 | 0.012 |
| PPO | 0.436 | 0.633 | 0.054 | 1.234 | 0.009 |

这些是短训练 checkpoint 的输出成本，不能外推到已学会游戏的长回合或完整训练预算；仍应保留教学所需的定期评估。

## 作业并发

对同样两个 DQN Pong 完整测速进程比较总墙钟时间，包括进程启动、100,000 条回放填充、预热和计时段。两种设置总共分配 8 个 CPU 核：串行作业各用 8 核，并行作业各用互不重叠的 4 核。

| 同时作业数 | 两个作业总秒，中位数 |
| --- | ---: |
| 1 | 42.73 |
| 2 | 29.17 |

该结果用于选择两个独立 DQN 作业的调度，不代表 PPO/GPM 混合队列也有同等收益。全局默认仍串行；显存和 CPU 足够时可显式用 `--max-workers 2 --cpus-per-job 4`。

## 配置、验证与来源

- 硬件为 RTX 5090、Core Ultra 9 285K；PyTorch 2.14.0+cu130、CUDA 13.0、ALE 0.12.1、Gymnasium 1.3.0。流水线测量绑定 CPU 0–7，PyTorch/OpenMP/MKL/OpenCV 单线程，原生 ALE 为 4 线程。
- DQN：回放填满 100,000，epsilon=0.1，预热 1,024 transitions，计时 16,384 transitions / 4,096 次更新，uniform without replacement；回放留在 CPU。
- PPO：预热 2,048、计时 40,960 transitions，8×128 rollout，4 epochs，minibatch 32。compile 使用 reduce-overhead、fullgraph，完整 CUDA 更新图只关闭内部嵌套捕获。
- 流水线计时不包括 checkpoint、录像和周期评估；预热时间独立记录。阶段 CPU 墙钟不等同于 CUDA kernel 时间。
- 初轮 PPO 六项 repeat-0 测量与 CPU 回归检查时间重叠，按时间窗口预先排除，随后隔离重测并替换统计；原始结果和排除记录均保留。
- 最终代码通过 Ruff、格式检查和 CUDA 可用环境下的 218 项回归测试；wrapper / ALE 各 24 个训练及独立评估作业完成。逐项核对 checkpoint 有限性、观测协议、预算、DQN 更新数、初始化及新任务头，56 个 MP4 均可解码。最终 smoke 使用两个有独立 8 核配额的并发作业，仅作功能验证。
- 大产物在 `outputs/benchmarks/cleanup_20260909T093344Z`。`baseline/source` 是本轮清理前状态（已含此前完整更新图）；`final/source` 用于主流水线/视频对照；`qualified/source` 再加入 GPM 仅动作采样，最终 smoke 与此运行代码一致。最终路径另做三个独立进程的默认配置复测。
- 初轮 smoke 和中间视频候选保留为历史产物，以上验证数字只引用 `smoke_final_wrapper` / `smoke_final_native`。源码哈希、全部重复值、显存、测试与初始化记录见[机器可读结果](cleanup_performance.json)。
