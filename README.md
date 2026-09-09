# Atari 强化学习教程

用 DQN 和 PPO 学习 Pong、Breakout 和 Space Invaders，比较联合训练与顺序训练，并观察新任务学习带来的遗忘。

## 教学内容

| 案例 | 学习重点 |
| --- | --- |
| 单任务 DQN / PPO | 从游戏交互中学习动作价值或策略 |
| 联合多任务训练 | 交替训练多个游戏，共享特征网络 |
| 普通顺序训练 | 依次学习新游戏，观察旧任务的得分变化 |
| 顺序训练 + EWC | 限制旧任务中重要参数的变化 |
| 顺序训练 + GPM | 根据旧任务特征约束更新方向 |

持续学习部分以 PPO 为主，DQN 用于扩展比较。

默认读者熟悉 Python、PyTorch、梯度下降和基本矩阵运算。正文围绕案例展开，概念补充、数值例子与方法来源放在 notebook 附录中，按需查阅。

## 开始学习

使用 Python 3.12 或更新版本，在仓库目录安装依赖：

```bash
python -m pip install -e .
```

安装后，用 Jupyter 或 VS Code 打开 [notebook](Atari_RL_Complete_Tutorial.ipynb)，按顺序阅读并执行示例。

Notebook 中的训练调用默认带有 `--dry-run`，只打印命令。也可以在终端预览全部案例：

```bash
python scripts/run_experiments.py train teaching \
  --seed 0 --num-envs 8 --dqn-num-envs 8 --env-backend async --compile-ppo --compile-dqn --dry-run
```

先加上 `--smoke` 运行短流程检查；移除 `--smoke` 和 `--dry-run` 后运行完整预算。实验调度器每次在 `outputs/runs/` 新建目录，其中保存源码快照、哈希、`checkpoints/`、`outputs/` 和 `logs/`。也可用 `--run-dir` 指定新目录；评估已有运行时传入同一目录。其他选项见 `python scripts/run_experiments.py --help`。

教学配置默认启用 PPO/DQN 编译，使用 `reduce-overhead`；可用 `--no-compile-ppo`、`--no-compile-dqn` 对照 eager 路径。PPO 三种训练协议均支持同一游戏的批量采样，DQN 批量采样仍按累计 transition 数安排更新。

两种算法默认均为每个游戏 8 个 async wrapper 环境，训练 minibatch 为 32。教学入口的 `--num-envs` 控制 PPO，`--dqn-num-envs` 控制 DQN；底层训练脚本仍默认单环境。PPO 每环境采样 128 步，共收集 1,024 条 transition 后更新 4 epochs；DQN 达到学习起点后，每累计 4 条 transition 更新一次。`--steps` 统计所有环境合计的交互步数。

CUDA 上的学习器进一步将损失、反向传播和 Adam 更新放入同一个 CUDA Graph。图内保留算子编译，并关闭嵌套图捕获；DQN 的训练指标按更新顺序成组读取，减少 CPU/GPU 同步。

CUDA 使用 fused Adam，默认采样路径将 PyTorch/OpenCV 设为单线程；DQN 使用连续 CPU 回放存储。已有测试未支持将双缓冲采样、回放预取或 EnvPool 设为默认。ALE 线程数保持 4，作业默认串行；增加线程或并发作业应结合游戏和可用 CPU 核单独选择。

原生 ALE 将模拟器推进和图像预处理放在 C++ 中，可用以下命令先检查全部案例：

```bash
python scripts/run_experiments.py train teaching --seed 0 --smoke \
  --env-backend ale --num-envs 8 --dqn-num-envs 8 --env-threads 4
```

ALE 的灰度、最大池化和堆帧初始化与 wrapper 路径不同，因此它是独立的训练配置。checkpoint 记录这一协议，独立评估自动采用匹配的预处理和原始奖励；ALE 视频显示智能体看到的灰度画面。默认仍使用 wrapper 路径。

`--max-workers 2 --cpus-per-job 4` 可让两个独立作业共享 GPU，并为它们分配互不重叠的 CPU 核；每个模型的独立评估等待对应训练完成。单个 PPO 作业始终先收集 rollout，再更新参数。

评估入口 `scripts/evaluate.py` 同时生成分数和示例 MP4，按 checkpoint 选择观测协议。录像逐帧编码，编码失败会使作业报错。底层训练入口可用 `--save-video` 记录训练画面，每两个向量步记录一帧。

## 阅读代码与结果

- `algorithms/`：DQN、PPO、EWC 与 GPM。
- `environments/`：Atari 环境与图像预处理。
- `training/`：数据缓冲、DQN/PPO 采样与更新、回合评估。
- `scripts/`：训练、评估和实验调度入口。
- `tests/`：采样、更新与评估的测试。

PPO 的网络、策略和损失定义位于 `algorithms/ppo.py`，采样与更新由 `training/ppo_runtime.py` 中的 `PPOCollector` 和 `PPOLearner` 执行。Notebook 的小型数值示例统一调用 `scripts/tutorial_examples.py`。

各案例的训练步数、已有得分和图表见 [结果记录](results/README.md)。
计算性能、参数比较和测速命令见 [运行性能记录](results/performance.md)。
DQN 编译与满容量回放的测速结果见 [DQN 计算性能](results/dqn_performance.md)。
本轮完整计算流程、原生 ALE 与批量采样的比较见 [综合计算优化](results/computation_performance.md)。
进一步的更新图、回放预取和 EnvPool 比较见 [完整学习器性能](results/learner_performance.md)。
代码清理、当前 8 环境配置及视频输出的复测见 [教学代码与性能复测](results/cleanup_performance.md)。
性能和得分记录保留当时的配置；当前教学默认值以本页及实验调度器为准。DQN 的环境数改变后，需要重新运行正式训练，才能记录新配置的学习结果。
