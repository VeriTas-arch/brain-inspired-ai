# 综合计算流程优化

2026-09-09。本轮覆盖 DQN/PPO 的采样、回放、更新、评估、EWC/GPM 任务边界和作业调度。Pong 的完整采样与更新测速中，原生 ALE 配合批量推理可进一步提高吞吐；收益随游戏和环境数变化。默认教学配置继续使用 wrapper，ALE 作为显式选择。

这些结果衡量执行速度及流程正确性。短测没有验证收敛得分或达到目标分数的时间，原生 ALE 的得分也不能直接沿用 wrapper 的历史结果。

## 已实现的调整

| 流程 | 调整与保留的行为 |
| --- | --- |
| DQN 回放 | 使用独立、可设种子的 NumPy Generator，仍然均匀无放回采样；连续 CPU 数组支持整批写入，并保存观察副本。容量 100,000 的两组图像约占 5.26 GiB，回放仍留在 CPU。 |
| DQN 采样 | 支持多个环境和集中批量推理；按累计 transition 数计算应执行的更新次数，保留原单任务、顺序和联合协议各自的学习起点。截断时从终止观察 bootstrap。 |
| PPO 更新 | 在 `training/ppo_runtime.py` 合并 minibatch 索引、损失和 EWC 惩罚的编译；编译梯度裁剪与 GPM 投影，复用 GPU rollout 观察。不足一个 minibatch 的尾批走 eager。 |
| PPO 联合训练 | 每次选择一个游戏，完整收集该游戏的 rollout 后再更新；所有 PPO 入口复用同一采样器。默认 8 环境、每环境 128 步，共 1,024 transitions。 |
| 评估和任务边界 | 编译确定性贪心推理；仅在模型、种子、回合数和步数限制相同的阶段末尾复用当前任务评估。GPM 只保留预先选中的观察和卷积 patch；EWC 将选定样本整批传入 GPU，仍计算逐样本梯度平方。 |
| Atari | 接入 ALE 原生 C++ 向量环境和预处理；checkpoint 保存观察协议，评估和视频使用匹配路径。训练中的 life-loss 边界与评估中的完整回合分开处理。 |
| 作业调度 | 有界并发、依赖等待、CPU 核分配和失败状态；每次训练使用新目录，保存可独立运行的源码快照与 SHA-256。 |

教学入口默认启用 `reduce-overhead` 编译，可编译的热路径使用 `fullgraph=True`，保留 fused Adam 和 eager 路径。没有增加精度放宽或混合精度。

独立回放 RNG 改变了旧版本的随机序列；DQN 批量到达的 transition 也改变动作与更新的交错顺序。联合 PPO 改为按 rollout 抽取任务。这些配置需要新的学习结果，不能声称与旧版本轨迹完全相同。顺序学习的普通训练、EWC 和 GPM 则直接核对了初始化、任务顺序和预算。

## 测量条件

硬件为 RTX 5090、Intel Core Ultra 9 285K，进程限定在 CPU 0–7；PyTorch 2.14.0+cu130、CUDA 13.0、Gymnasium 1.3.0、ALE 0.12.1。PyTorch、OpenMP、MKL 和 OpenCV 均使用 1 个线程，ALE 线程数见各表。`--torch-threads` 可在 PPO 测速入口显式覆盖线程数。

完整训练流水线的每项配置重复 3 次，均使用 seed 0 和确定性计算，表中为中位数。测速固定 minibatch 为 32，不通过放大 minibatch 提速。

| 项目 | 预热 | 计时范围 |
| --- | --- | --- |
| DQN | 先填满 100,000 条回放，再预热 1,024 transitions | 16,384 transitions、4,096 次更新；固定 epsilon=0.1 |
| PPO | 2,048 transitions | 40,960 transitions；每环境 rollout 128 步、每批更新 4 epochs |
| 环境单独运行 | 2,048 transitions | 40,960 transitions；提前生成动作，不执行神经网络 |

完整流水线的吞吐排除回放填充、预热、绘图、保存和评估；各次预热及填充耗时保存在 [JSON](computation_performance.json)。编译磁盘缓存没有清空，因此预热时间不代表首次安装后的冷启动。计时边界同步 CUDA。显存表示 PyTorch 的峰值 allocated / reserved，不是整张显卡总占用。

## DQN：回放与批量采样

先独立比较回放 RNG，其他配置固定：Pong 从 **998 → 1,227 transitions/s，增加 22.9%**；计时区间内回放耗时从 4.00 降至 0.99 秒，网络更新仍约 4.0 秒。该对照使用 `baseline/` 与 `replay_optimized/` 快照。

随后比较完整流水线，以下各项均使用新的独立 RNG：

| 环境与存储 | 环境数 / ALE 线程数 | transitions/s | 预填充 / 预热（秒） | 显存 MiB |
| --- | --- | ---: | ---: | ---: |
| wrapper、环形列表 | 1 / — | 1,217 | 34.01 / 2.12 | 27.5 / 112 |
| wrapper、连续数组 | 1 / — | 1,227 | 33.22 / 2.11 | 27.5 / 112 |
| ALE、连续数组 | 1 / 1 | 1,032 | 43.32 / 2.28 | 27.5 / 112 |
| ALE、连续数组 | 4 / 4 | 2,000 | 13.49 / 1.80 | 27.5 / 112 |
| ALE、连续数组 | 8 / 4 | 2,263 | 10.15 / 1.71 | 27.5 / 112 |
| ALE、连续数组 | 8 / 8 | 2,396 | 8.27 / 1.71 | 27.5 / 112 |

ALE 8 环境、8 线程比当前 wrapper 单环境快 **95.3%**，其中同时包含原生环境和增大推理批次的收益。仅将单环境切换到 ALE 反而更慢。8/8 配置的计时区间内，推理约 0.48 秒、环境约 1.40 秒、回放约 0.78 秒、更新约 4.16 秒；网络更新成为更大的耗时部分。阶段计时包含主机调度，不能当作 CUDA kernel 时间。

连续数组相对环形列表的完整吞吐仅增加 **0.8%**。保留它主要因为可以直接整批写入并明确数据所有权。较小的存储测试每次写 4 条、采样 32 条并执行一次真实更新，容量同为 100,000，预热 64 次、计时 1,024 次更新：

| 存储候选 | updates/s，3 次中位数 |
| --- | ---: |
| 环形列表 | 801 |
| 环形列表 + pinned staging | 771 |
| 连续数组 | 921 |
| 连续数组 + pinned staging | 801 |

本机 pinned staging 的额外复制抵消了传输收益，未采用。手动 pinning 是否有效需要测量完整路径，[PyTorch 的传输教程](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)也区分了页锁定成本和传输条件。

## PPO：更新融合与原生环境

先在相同 wrapper、8 环境上隔离 PPO 更新改动：**3,662 → 3,955 transitions/s，增加 8.0%**；更新阶段从 4.91 降至 4.03 秒，采样仍约 6.27 秒。峰值 allocated / reserved 从 81.1 / 152 MiB 变为 82.6 / 172 MiB。

最终环境比较使用同一优化后运行时：

| 环境 | 环境数 / ALE 线程数 | transitions/s | 采样 / 更新（秒） | 预热（秒） | 显存 MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| wrapper async | 8 / — | 4,011 | 6.23 / 3.98 | 2.04 | 82.6 / 172 |
| ALE | 8 / 4 | 4,373 | 5.32 / 4.04 | 1.89 | 82.6 / 172 |
| ALE | 8 / 8 | 4,783 | 4.58 / 3.98 | 1.87 | 82.6 / 172 |
| ALE | 16 / 8 | 5,829 | 3.03 / 4.02 | 1.82 | 113.2 / 244 |

8/8 ALE 比同环境数的 wrapper 快 **19.2%**，主要缩短采样时间。16 环境进一步提高吞吐，但每次 rollout 从 1,024 变成 2,048 条，属于另一个训练配置，不能据此认定学习时间更短。

PPO 仍然先完成 rollout，再更新策略。前一轮相同 8 环境的双缓冲测试没有收益，见 [此前的测量](performance.md#环境数线程与双缓冲)；本轮没有增加该运行分支。

## Atari 本身的执行成本

[ALE 原生向量环境](https://ale.farama.org/vector-environment/)将多环境执行与图像处理放入 C++，减少 Python wrapper 调用和跨进程通信；模拟器仍在 CPU 上运行。下面包含推进、预处理、重置和观察交付，不能把它解释为纯模拟器指令吞吐。

每个游戏各测一次，8 环境、ALE 4 线程，均不运行策略网络：

| 游戏 | wrapper async transitions/s | ALE transitions/s |
| --- | ---: | ---: |
| Pong | 9,384 | 11,579 |
| Breakout | 6,502 | 4,677 |
| Space Invaders | 6,978 | 8,593 |

Breakout 本次较慢；不同 reset、life-loss 和预处理行为会影响这类测量，单次结果不能确定具体原因。完整训练流水线的三次重复只覆盖 Pong，因此没有把 ALE 设为所有游戏的默认后端。

原生协议固定四帧跳帧与堆叠、0.25 sticky action、30 个最大随机 no-op；训练使用 reward clipping 和 episodic life，评估关闭二者。灰度、池化、堆帧初始值与 wrapper 不完全相同，分别记为 `ale_native_v1` 和 `gymnasium_wrappers_v1`。独立评估读取 checkpoint 协议；ALE 视频展示策略观察到的灰度帧。

## GPM 和 EWC 边界

GPM 采样对照在相同冻结 PPO 策略、8 个 async 环境下收集 32,768 transitions，最终保留 2,048 张观察；先预热 1,024 transitions，重复 3 次。

| 指标 | 原实现 | 优化后 |
| --- | ---: | ---: |
| 采样耗时中位数 | 7.30 秒 | 6.48 秒 |
| 主进程峰值 RSS | 3,943.8 MiB | 2,156.3 MiB |
| PyTorch allocated / reserved | 41.7 / 56 MiB | 69.7 / 84 MiB |

六次运行选出的观察 SHA-256 完全相同。RSS 是整个主进程的峰值，包含框架及暖机，不含环境子进程。

另用固定的 2,048 张随机 uint8 观察、batch 32、threshold 0.995 测量子空间构建，先用 32 张观察预热。三次中位数从 **0.346 → 0.311 秒**，allocated / reserved 从 **582.7 / 690 → 581.5 / 684 MiB**。四层的 rank 均为 250、341、352、1,134，三组子空间基逐元素相同。该子空间测速输入是合成观察，不代表训练后的保护空间或遗忘程度。

GPM 在旧基已覆盖要求能量时跳过无需执行的残差分解。EWC 仍采用逐样本梯度平方，不将平均梯度的平方替代 Fisher；本轮只减少选定样本的重复传输。二者的数值回归及真实任务切换均已通过，未为 EWC 单独报告边界提速幅度。

## 多作业吞吐

在总计 8 个 CPU 核和同一 GPU 上运行两个相同的 DQN 测速作业，每个作业使用 ALE 8 环境、4 线程和上述满回放预算：

| 调度 | 每作业 CPU 核 | 两作业完成时间 |
| --- | ---: | ---: |
| 串行 | 8 | 44.14 秒 |
| 两作业并发 | 4 | 26.34 秒 |

这是一次作业组比较，包含进程启动、回放填充、预热、计时和退出，总用时减少 **40.3%**。并发时每个作业自身从约 2,280 降至约 1,840 transitions/s。没有据此推断完整教学矩阵也有相同收益；默认仍为一个 worker。

## 运行与验证

先在新目录运行全部短配置，原 wrapper 与原生 ALE 分开保存：

```bash
direnv exec . python scripts/run_experiments.py train teaching --seed 0 --smoke

direnv exec . python scripts/run_experiments.py train teaching --seed 0 --smoke \
  --env-backend ale --num-envs 8 --dqn-num-envs 8 --env-threads 4
```

可添加 `--dry-run` 预览、`--run-dir` 指定新目录，或 `--max-workers 2 --cpus-per-job 4` 限制并发资源。移除 `--smoke` 后使用正式预算。PPO 默认 8 环境、async，DQN 默认单环境、sync；两者默认编译。联合训练记录每个游戏实际分配的 transitions，并不保证与顺序训练逐游戏预算相等。

复测 Pong 的主要配置：

```bash
direnv exec . taskset -c 0-7 python scripts/benchmark_dqn_runtime.py \
  --compile-dqn --env-backend ale --num-envs 8 --env-threads 8 \
  --replay-size 100000 --warmup 1024 --transitions 16384 --batch-size 32 --seed 0 \
  --output outputs/benchmarks/new-dqn-ale-8.json

direnv exec . taskset -c 0-7 python scripts/benchmark_ppo_runtime.py \
  --configuration ale --num-envs 8 --env-threads 8 --torch-threads 1 \
  --warmup-transitions 2048 --transitions 40960 --batch-size 32 --seed 0
```

DQN wrapper 对照使用 `--env-backend sync --num-envs 1`；PPO wrapper 对照使用 `--configuration optimized`。给 PPO 测速命令增加 `--environment-only --device cpu` 可独立测量环境路径。每次输出应写入新的文件。

验证覆盖以下范围：

- `ruff check .`、`ruff format --check .` 通过；CUDA 环境下 `pytest -q` 为 **209 passed**。显式多线程的 async 测速测试产生 1 条 Python `fork()` 弃用警告，未过滤该警告。
- 原 wrapper 与原生 ALE 各 24 个 smoke 作业，合计 24 次训练、24 次独立评估；全部完成，产生 24 个最终 checkpoint 和 56 个视频。
- DQN 每任务 10,032 transitions，PPO 每任务 2,048，联合训练使用三倍总预算；DQN 单任务、顺序、联合分别完成 8、24、5,025 次更新。
- 32 项直接初始化检查通过：普通顺序、EWC、GPM 的初始网络、第一阶段结束权重和后续新任务头匹配；任务顺序和顺序训练预算一致。
- 参数与分数有限，终止和截断观察、确定性动作、编译/eager 更新、回放环绕、GPM 子空间与投影、源码哈希、调度依赖均通过验证。
- 产物审计发现 EWC 原生评估的 JSON 协议标签读取了外层 wrapper。实际评估环境正确；已修复标签读取，在新目录重跑两算法、两环境协议的 4 项评估。原始产物保留，修正结果见 `evaluation_fix/`。

当前 TorchRL 用于已有的数值交叉验证与依赖检查；本轮训练运行时继续使用这里的 PyTorch 实现。没有把依赖导入当作 TorchRL 训练集成。

## 证据位置

小型汇总见 [computation_performance.json](computation_performance.json)。原始日志、测试脚本和源码位于 `outputs/benchmarks/audit_20260909T054333Z/`：

- `baseline/` 保存本轮开始时的工作树，包含前两轮尚未提交的优化；它不是裸 `ae43870`。前两份性能记录保留为历史阶段结果。
- `replay_optimized/` 与 `ppo_optimized/` 用于独立增量对照；`pipeline_measurements_v2/` 的主要配置来自 `qualified_v2/`，其中环形列表对照来自 `replay_optimized/`。
- `validated/` 用于全部 48 项 smoke、环境单独测速与边界/并发测试；`smoke_wrapper/source/`、`smoke_native/source/` 分别保存执行源码。
- `evaluation_fix/source/` 保存四项补充评估的源码。最后只补正了 PPO 测速的显式线程参数及其测试，最终源码见 `qualified_final/source/`；吞吐表的线程数始终为 1，不受该参数修复影响。
- `qualification_verification.json` 记录 checkpoint 哈希及初始化检查；`boundary_measurements/parity.json` 记录观察与子空间的一致性；每个源码快照都有对应 manifest。

历史输出未被覆盖。本轮没有启动新的完整预算教学训练。
