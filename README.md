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
  --seed 0 --num-envs 8 --env-backend async --compile-ppo --dry-run
```

移除 `--dry-run` 后开始训练。模型、结果和日志分别保存在 `checkpoints/`、`outputs/`、`logs/`；重新运行相同配置前，请先另存已有记录。其他选项见 `python scripts/run_experiments.py --help`。

## 阅读代码与结果

- `algorithms/`：DQN、PPO、EWC 与 GPM。
- `environments/`：Atari 环境与图像预处理。
- `training/`：数据缓冲、PPO 采样与更新、回合评估。
- `scripts/`：训练、评估和实验调度入口。
- `tests/`：采样、更新与评估的测试。

各案例的训练步数、已有得分和图表见 [结果记录](results/README.md)。
