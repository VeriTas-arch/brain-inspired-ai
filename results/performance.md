# 运行性能记录

2026-09-09，Pong-v5，seed 0，确定性训练。优化后的正式路径保留原来的观测像素、采样随机序列和 PPO 更新方式。

## 已采用的优化

- 灰度转换按通道计算，避免构造 RGB 大小的浮点临时数组。系数、加法顺序和 uint8 取整与 Gymnasium 一致；全部 16,777,216 种 RGB 颜色逐值相同。84×84 图像的转换核从约 **101 μs 降至 13 μs**。
- DQN 回放使用环形列表，消除 deque 随机索引的链表遍历。保留逐批不放回采样、全局 NumPy RNG 消耗顺序及 EWC 的独立采样生成器；覆盖容量后仍按原来的逻辑顺序取样。
- 周期评估和顺序训练的阶段记录增加累计 `elapsed_seconds`；联合训练在 `training_summary.json` 记录总耗时。计时从环境和网络初始化前开始，包括训练函数内此前的采样、更新、评估和阶段处理。

## 完整 PPO 与环境吞吐

RTX 5090（32 GB）、Core Ultra 9 285K，固定 CPU 0–7，PyTorch/OMP/MKL 线程数均为 1。PyTorch 2.14.0+cu130、ALE 0.12.1。每个配置用独立进程；三轮交替测量，表内为中位数。

固定 8 个环境、rollout 长度 128、minibatch 32、4 个更新 epoch；每次预热 2,048 transitions，再计时 40,960 transitions。策略和损失使用 `reduce-overhead`、`fullgraph=True`。单位是 agent transitions/s，每个动作重复 4 帧，不包含额外 reset 帧。

| 路径 | 仅环境吞吐 | 收集＋更新吞吐 | PPO 预热耗时 | GPU allocated / reserved 峰值 |
| --- | ---: | ---: | ---: | ---: |
| 优化前 Gymnasium async | 8,702 | 3,425 | 2.05 s | 81.1 / 152 MiB |
| 优化后 Gymnasium async | 9,913 | 3,588 | 2.02 s | 81.1 / 152 MiB |
| ALE 原生向量环境，4 线程 | 11,471 | 3,888 | 1.88 s | 81.1 / 152 MiB |

保持正式训练协议的改动使环境吞吐提高 **13.9%**，完整 PPO 吞吐提高 **4.8%**。更新部分耗时基本不变，收益来自采样。单独测量同步环境时，吞吐从 2,350 提高到 3,034 transitions/s；这两项各测一次。

ALE 一行采用不同的原生预处理与 reset 流程，仅用于性能比较。原生实现包含 C++ 批处理，但其帧栈填充、灰度/最大池化和 reset 路径不能直接视为现有 wrapper 的等价替换。因此 ALE 入口只放在 benchmark 中，尚未接入正式训练和 checkpoint 评估。[ALE 官方说明](https://ale.farama.org/vector-environment/)

## 环境数、线程与双缓冲

以下参数探索各测一次，预热 2,048、计时 10,240 transitions，其余参数同上。增加环境数也增加每次收集的 rollout 大小，尚未验证对学习质量的影响。

| 后端 | 环境数 | 环境线程数 | 环境吞吐 | 收集＋更新吞吐 |
| --- | ---: | ---: | ---: | ---: |
| Gymnasium async | 16 | 每环境一个进程 | 11,665 | 4,312 |
| Gymnasium async | 32 | 每环境一个进程 | 13,596 | 4,645 |
| ALE 原生 | 8 | 8 | 16,091 | 4,247 |
| ALE 原生 | 16 | 8 | 22,580 | 5,149 |
| ALE 原生 | 32 | 8 | 26,867 | 5,767 |

8 个 Gymnasium 环境下，将 PyTorch 线程数增到 2、4 后，完整吞吐分别为 3,270、3,138 transitions/s，因此继续使用 1 个线程。ALE 的环境线程另行分配，不能与 PyTorch 线程数混为一谈。

双缓冲原型采用两组环境，rollout 内固定策略，所有未完成的环境步返回后才允许更新。验证了预步观测拷贝、终止帧、截断 bootstrap、每环境时间轴和精确预算。相同原型代码、8 个环境下，三轮中位数为：整批等待 **3,635**，双缓冲 **3,581** transitions/s（每次计时 10,240 transitions）。拆分推理批次的额外开销抵消了重叠执行收益；原型保存在测试快照中，正式代码未增加双缓冲训练分支。[双缓冲方法说明](https://www.samplefactory.dev/07-advanced-topics/double-buffered/)

## DQN 回放与 TorchRL

CPU 0、1 个线程、容量 100,000、minibatch 32，uint8 的当前和下一观测，seed 0。先填满缓冲；三轮测量，每轮预热 20 次采样，计时 500 次采样及 2,000 次“写入、每四步采样一次”的组合操作。

| 实现 | 单次采样 | 组合操作每条 transition |
| --- | ---: | ---: |
| 原 deque | 1,016 μs | 263 μs |
| 环形列表 | 904 μs | 233 μs |
| TorchRL LazyTensorStorage ＋ index_select 候选 | 848 μs | 259 μs |

环形列表使组合耗时下降 **11.4%**。TorchRL 候选的采样更快，但逐条 TensorDict 写入抵消了大部分收益，整体未优于环形列表；当前训练采用环形列表。这里测量的是回放操作，不代表完整 DQN 训练提速 11.4%。

TorchRL 的普通与向量化 GAE 已加入数值交叉验证，覆盖多环境、终止边界和已加入奖励的截断 bootstrap。训练中的损失、GAE 与 EWC/GPM 更新逻辑保持原实现。

## 运行与验证

复测完整 PPO：

```bash
direnv exec . env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0-7 \
  python -m scripts.benchmark_ppo_runtime --configuration optimized \
  --num-envs 8 --batch-size 32 --torch-threads 1 \
  --warmup-transitions 2048 --transitions 40960
```

加 `--environment-only` 测量环境路径；使用 `--configuration ale --env-threads 8` 比较原生环境。输出包含预热、收集、更新、硬件、版本和 GPU 内存记录。
表中的“优化前”取自 ae43870 冻结源码；CLI 的 `baseline` 表示当前代码的同步/eager 运行方式。

本轮未测量达到目标得分的时间，也未重跑正式学习曲线。后续需按游戏预先确定阈值，再从包含 `elapsed_seconds` 的评估点报告首次达标时间；周期评估只能确定首次观测到达标的时间。联合训练当前只有总训练耗时，尚无周期达标曲线。

Ruff 检查、格式检查及全部 **158 项 GPU/CPU 测试通过**。教学矩阵 **24/24 smoke 通过**：12 个训练入口及 12 个独立评估入口，12 个 checkpoint 均可安全加载且参数有限，生成 28 个非空视频。PPO 每游戏 2,048 transitions，DQN 每游戏 10,032 transitions（跨过 10,000 步预热并实际更新），联合训练用三倍预算；GPM 使用 16 个样本、32 步边界采样。全部 seed=0、确定性训练，独立评估每游戏 1 回合、上限 30,000 步。

另完成 12 项直接参数比较：PPO/DQN 的普通顺序训练、EWC、GPM 在首任务结束后的网络参数及后续新头初始化逐值一致。Smoke 仅证明执行路径可用，不代表学习收敛。

完整数值见 [performance.json](performance.json)。本地原始输出、冻结源码、源文件哈希与测试脚本位于 `outputs/benchmarks/pipeline_20260909T030108Z/`：`baseline/` 是 ae43870，`optimized/` 用于正式路径测速，`double_buffer_candidate/` 保存未采用的原型，`smoke/workspace/` 保存本次教学入口验证。历史教学分数和模型未被覆盖。

DQN 网络计算的后续优化与完整吞吐测量见 [DQN 计算性能](dqn_performance.md)。
