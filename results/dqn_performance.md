# DQN 计算性能

2026-09-09，seed 0，确定性 FP32 训练。基线已包含上一轮灰度转换和环形回放优化；本轮比较推理、TD 更新、EWC 与 GPM 的计算路径。

## 已采用的实现

- 在线网络、目标网络、TD target 和 MSE loss 在同一编译函数中计算；EWC 的已缓存惩罚项纳入该函数。反向传播由编译后的 loss 生成。
- 贪心动作推理、梯度裁剪和 GPM 投影也使用 `mode="reduce-overhead", fullgraph=True`。多头网络按任务缓存计算函数，任务调度和探索随机数在图外处理。
- 观测保持 uint8 传输，在网络中归一化；H2D 复制使用 `non_blocking=True`。诊断值打包后在更新结束统一读取，移除反向传播之前的 `.item()` 同步。
- GPM 复用更新前权重缓冲区，保持 Adam 更新后投影、投影后同步目标网络的顺序。统计累加放在投影图外，让投影计算可以被 CUDA Graph 捕获。
- 保留 fused Adam。其当前包装器含显式 graph break；允许 graph break 的 Adam 编译在真实更新测试中没有表现出稳定额外收益。
- 模型仍保存原始模块的 state dict；编译开关不进入 checkpoint 键名。保留 eager 路径。

`fullgraph=True` 检查所选函数是否完整捕获；`reduce-overhead` 的 CUDA Graph 适用条件仍需运行验证。本轮正式基准日志没有 CUDA Graph 跳过或编译次数超限记录。[PyTorch compile](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html)、[CUDA Graph Trees](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html)、[优化器编译](https://docs.pytorch.org/tutorials/recipes/compiling_optimizer.html)。

## 测量设置

RTX 5090 32 GB、Core Ultra 9 285K，固定 CPU 0–7，Torch/OMP/MKL 各 1 线程。PyTorch 2.14.0+cu130、ALE 0.12.1、Gymnasium 1.3.0。每项使用独立进程，三轮交替测量，下表为中位数。

完整路径使用 Pong-v5、单环境、minibatch 32、每四条 transition 更新一次，探索概率固定为 0.1。先用固定动作填入 10,000 条回放，随后预热 1,024 条，计时 16,384 条（4,096 次更新，覆盖目标网络同步）。计时包含动作选择、环境推进/reset、回放写入/采样和学习更新；不含评估、绘图和 checkpoint 保存。

独立更新测试从 64 条真实回放中取固定 CPU batch，预热 64 次、计时 512 次，包含 H2D、反向传播、裁剪、Adam 和诊断读取。EWC/GPM 先进行一次旧任务更新，用 16 个观察完成真实边界处理，再切换至新任务头。这里评估阶段计算成本；完整联合任务切换和三阶段顺序协议由 smoke 检查。

## 吞吐

| 路径 | 基线 transitions/s | 优化后 transitions/s | 增幅 | 基线 updates/s | 优化后 updates/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 单任务 | 914 | 1,132 | 23.8% | 617 | 1,091 |
| 多头，固定当前任务 | 897 | 1,131 | 26.1% | 599 | 1,105 |
| 多头＋EWC | 783 | 1,104 | 41.1% | 434 | 976 |
| 多头＋GPM | 788 | 1,068 | 35.5% | 432 | 837 |

单任务重构后的 eager 路径为 **928 transitions/s**。探索概率固定为 0.9 时，完整吞吐为 **1,098 → 1,351 transitions/s**。

独立贪心推理微测为 **8,345 → 8,696 次/s**（64 次预热，512 次计时）。推理收益与调用上下文有关，应以完整路径为主要性能依据。

| 优化后路径 | 预热时间 | allocated 峰值 | reserved 峰值 |
| --- | ---: | ---: | ---: |
| 单任务 | 2.15 s | 27.5 MiB | 100.0 MiB |
| 多头，固定当前任务 | 2.15 s | 27.5 MiB | 100.0 MiB |
| 多头＋EWC | 1.99 s | 53.3 MiB | 154.0 MiB |
| 多头＋GPM | 2.14 s | 40.1 MiB | 132.0 MiB |

预热包含当次图捕获和编译缓存载入；没有清除磁盘编译缓存。回放预填充时间单独记录在 JSON 中。GPU 数值为 PyTorch allocator 统计，不含驱动/context。

满容量 100,000 条回放缓冲的单任务复测（三次独立重复，其余设置相同）：**818 → 996 transitions/s（+21.8%）**。每次先在真实环境中填满回放，再开始预热和计时。

## 使用与验证

教学矩阵默认启用 DQN 编译，`--no-compile-dqn` 可用于 eager 对照；单独调用训练或评估脚本时使用 `--compile-dqn`。Notebook 的 DQN 预览命令已同步。

```bash
direnv exec . env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0-7 \
  python -m scripts.benchmark_dqn_runtime --variant single --component pipeline \
  --compile-dqn --warmup 1024 --transitions 16384 --batch-size 32 \
  --replay-size 10000 --epsilon 0.1 --seed 0 --output outputs/benchmarks/dqn-new/run.json
```

`--variant` 可选 `single`、`multi`、`ewc`、`gpm`；`--component update` 或 `inference` 测量对应部分。移除编译开关测量当前 eager 实现。冻结的历史基线由下方源码快照复现。

Ruff 检查与格式检查通过，全部 **164 项测试通过**；教学矩阵 **24/24 smoke 通过**，包含 12 个训练和 12 个独立评估入口。DQN 每游戏 10,032 步、PPO 每游戏 2,048 步；联合训练用三倍预算，GPM 使用 16 个样本/32 步边界采样。验证了安全 checkpoint 加载、有限参数和新任务头初始化一致性。

数值测试对比了 loss、梯度、参数更新、目标网络、EWC 惩罚和 GPM 投影，允许 FP32 编译融合产生的小幅舍入差异。测速不建立学习效果等价性；本轮未运行完整学习曲线或测量达到目标得分的时间。EWC/GPM 的长期训练子空间与惩罚规模也可能改变计算成本。

## 证据

- 小型记录：[dqn_performance.json](dqn_performance.json)。
- 冻结源码、哈希清单、逐次测量和 smoke：`/home/veritas/Documents/Atari_Playground/outputs/benchmarks/dqn_20260909T034802Z`。
- `baseline/` 是本轮开始时的工作区；`optimized/` 是测速源码；`validated/` 是最终源码与 smoke 使用版本。后者补齐替换 backbone 后的参数缓存刷新，计算热路径 AST 相同（`hot_path_parity.json`）。`benchmark_matrix.py` 与 `dqn_full_replay.py` 保存测量矩阵。
- 先前的环境与回放优化：[performance.md](performance.md)。
