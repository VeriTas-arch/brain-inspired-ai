# 完整学习器更新与环境后端比较

2026-09-09。先将此前更改提交为 `5811524`，再在相同训练配置下比较完整 CUDA 更新图、DQN 指标延迟读取、回放预取和 EnvPool。保留完整更新图与指标延迟读取；回放预取没有实质收益，EnvPool 的收益随游戏变化，暂未加入正式入口或依赖。

本轮没有调整 minibatch、环境数、训练预算、更新频率或精度设置，也没有改为异步 actor–learner。张量仍为 float32，沿用既有 `matmul_allow_tf32=False`、`cudnn_allow_tf32=True`，未新增 AMP 或放宽精度。以下为短时吞吐和正确性验证，尚未测量收敛得分或达到目标得分的时间。

## 完整更新图

CUDA 学习器用一张图执行损失、反向传播、梯度裁剪、fused Adam 和可选的 GPM 投影。PPO 将整批 rollout 放入静态缓冲区，每个 minibatch 仅更新索引；DQN 保留逐次更新、epsilon 和目标网络同步的原有时钟。DQN 训练指标每 128 次更新读取一次，在评估或阶段结束前清空，保留每条指标及原有滚动 Q 均值。

推理和常规编译路径继续用 `reduce-overhead`。完整更新图内部只关闭 Inductor 的嵌套 CUDA Graph 捕获，保留 `fullgraph=True` 的算子编译；捕获预热后原位恢复参数、Adam 状态、RNG 和 GPM 计数。任务模块或 optimizer 状态改变时使缓存失效。实现依据 [PyTorch CUDA Graph 文档](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs)。

同一项重复 3 次、均为 seed 0，表中为中位数；计时包含采样、回放及更新，不含视频编码、checkpoint 写入和周期性评估。

| 配置 | 修改前 transitions/s | 修改后 transitions/s | 提升 | 修改后预热秒 | 修改后 allocated / reserved MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| DQN wrapper，1 环境 | 1,231 | 1,540 | 25.1% | 1.96 | 35.6 / 170 |
| DQN ALE，8 环境 | 2,222 | 3,768 | 69.6% | 1.56 | 35.6 / 172 |
| PPO wrapper async，8 环境 | 4,025 | 5,046 | 25.4% | 1.93 | 245.0 / 322 |
| PPO ALE，8 环境 | 4,342 | 5,615 | 29.3% | 1.79 | 245.0 / 322 |

移除未采用的预取分支后，当前 DQN ALE 8 环境入口单次复测为 3,690 transitions/s，更新数仍为 4,096；此值单独记录，不并入上述三次中位数。

计时区间内 PPO 峰值 allocated 从约 82.6 MiB 增至 245.0 MiB，主要增加静态 rollout 和图缓冲区；DQN 从约 27.5 MiB 增至 35.6 MiB。完整回放仍留在 CPU。

## 回放预取的对照

同一环境采样后，候选实现仅预取当时已到期的更新；后台读取不会跨越下一次回放写入。验证了环形覆盖、槽位复用和独立 RNG 序列。固定 8 环境时每组只有 2 次更新，预取余量有限。

| DQN ALE 8 环境候选 | transitions/s，3 次中位数 | 相对完整更新图 |
| --- | ---: | ---: |
| 此前编译路径 | 2,222 | -41.0% |
| 仅延迟指标读取 | 2,289 | -39.2% |
| 完整更新图 + 延迟指标 | 3,768 | +0.0% |
| 再加 CPU 预取 | 3,598 | -4.5% |
| 再加双槽 H2D | 3,669 | -2.6% |
| 再加 CPU 预取和双槽 H2D | 3,778 | +0.3% |

组合方案的差异处于本轮波动范围，没有为此保留线程池和双缓冲分支。候选实现及其 4 项 GPU 数据所有权测试保存在冻结快照中。

## EnvPool：三个游戏

隔离测试 EnvPool 1.2.6，全部后端固定 8 环境、CPU 0–7，原生后端使用 4 线程。环境测试包含预处理、重置和观察交付；PPO/DQN 列使用相同的完整更新图。表中单位均为 transitions/s。

| 游戏 | wrapper 仅环境 | ALE 仅环境 | EnvPool 仅环境 | ALE PPO | EnvPool PPO | ALE DQN | EnvPool DQN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pong | 9,734 | 11,386 | 10,699 | 5,702 | 5,362 | 3,737 | 3,611 |
| Breakout | 6,160 | 4,683 | 6,717 | 3,745 | 4,196 | 2,745 | 3,062 |
| SpaceInvaders | 8,342 | 8,905 | 8,724 | 4,996 | 4,930 | 3,423 | 3,414 |

EnvPool 没有在三个游戏上统一胜过 ALE，收益主要出现在 Breakout。当前教学入口继续使用 wrapper 和显式 ALE；候选 adapter 保留在本轮产物中，不新增 EnvPool 依赖。正式采用还需要独立记录其学习得分，不能沿用其他后端的 checkpoint 结果。

语义检查覆盖三个游戏的种子复现、时间截断最终帧、训练丢命与评估完整回合配置，并各用 1,500 个向量步骤逐项对照原始接口。EnvPool 的显式 reset 会重开整局，候选在终止后通过下一步自动重置续接丢命回合，并保留 final observation；设置 sticky action=0.25、frame skip=4、灰度 84×84、4 帧、训练 reward clipping。它的画面处理与 reset 仍属于独立协议。参见 [EnvPool Atari 配置](https://envpool.readthedocs.io/en/stable/env/atari.html)及[显式 reset 测试](https://github.com/sail-sg/envpool/blob/main/envpool/atari/atari_envpool_test.py)。

## 测量与验证

- RTX 5090、Core Ultra 9 285K；PyTorch 2.14.0+cu130、CUDA 13.0、Gymnasium 1.3.0、ALE 0.12.1。固定 CPU 0–7，PyTorch/OpenMP/MKL/OpenCV 线程数为 1。
- DQN：填满 100,000 条回放后预热 1,024 transitions；计时 16,384 transitions、4,096 次更新，epsilon=0.1。
- PPO 更新图比较：预热 2,048、计时 40,960 transitions；环境后端比较计时 20,480，其余相同。
- 环境单独比较：预热 2,048、计时 32,768 transitions。主对照每项重复 3 次，使用现有编译缓存；预热不等同于全新缓存的首次编译时间。
- 213 项 GPU 可用环境下的回归测试通过；新增对照覆盖参数、梯度、Adam 状态、任务切换、GPM 更新计数、checkpoint 恢复及完整更新图与 eager 尾批交替执行。
- wrapper / ALE 共 48 个教学训练及独立评估作业通过；核对普通顺序、EWC、GPM 的初始化、任务头、顺序、预算与最终 checkpoint 有限性。此 smoke 只证明流程可执行。

smoke 冻结后仅增加了 PPO 尾批测试参数和文档，训练运行代码一致。基线为 `5811524`；候选同时包含 `473a951` 的两处 import 排版修改，没有运行逻辑变化。完整原始数值、预热时间、PyTorch allocated/reserved 显存和源码哈希见 [机器可读记录](learner_performance.json)。大产物保存于 `outputs/benchmarks/learner_20260909T080251Z`；包含已拒绝的候选及源快照，最终代码与测试位于其中的 `final/source/`。阶段主机计时不能直接视作 CUDA kernel 时间。

复测当前实现：

```bash
python scripts/benchmark_dqn_runtime.py --compile-dqn --env-backend ale \
  --num-envs 8 --env-threads 4 --replay-size 100000 --warmup 1024 \
  --transitions 16384 --batch-size 32 --seed 0 --output outputs/dqn_graph.json
python scripts/benchmark_ppo_runtime.py --configuration ale --num-envs 8 \
  --env-threads 4 --torch-threads 1 --warmup-transitions 2048 \
  --transitions 40960 --batch-size 32 --seed 0
```

测速入口可用 `--no-capture-updates` 比较局部编译路径；DQN 另有 `--no-defer-metrics`。上述命令保持训练 minibatch 不变，正式训练入口无需新增开关。
