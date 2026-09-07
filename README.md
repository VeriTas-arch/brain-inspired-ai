# Atari RL Playground

一个面向课程教学的 Atari 强化学习与持续学习仓库。仓库保留可读的 PyTorch 手工实现，便于学生检查 DQN、PPO、多任务头和 EWC 的关键步骤；TorchRL 作为后续逐步迁移与对照验证的唯一新增强化学习框架依赖。

当前没有引入 Stable-Baselines3，也没有引入 EnvPool 等外部加速框架。

## 当前范围

- DQN 与 PPO 的单任务训练
- 共享视觉骨干、按游戏划分输出头的联合训练和顺序训练
- 基于逐样本平方梯度估计对角 Fisher 的 EWC
- 每个训练阶段完成后，对所有已见任务进行确定性评估
- 完整保存网络、任务头、优化器以及 EWC 状态
- pytest 回归测试与 Ruff 静态检查

TorchRL 目前只建立了依赖和导入测试。现有教学算法尚未整体改写为 TorchRL trainer；后续可以分别评估其 TensorDict、replay buffer、collector 和 objective 组件，避免一次性替换后失去可读的课程基线。

## 安装

项目要求 Python 3.12 或更高版本。`pyproject.toml` 只声明经过验证的最低版本，不设置依赖上界。安装项目及开发工具：

```bash
python -m pip install -e ".[dev]"
```

兼容旧的安装命令：

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` 只是指向 `pyproject.toml` 的兼容入口，不再维护第二份版本列表。

## 验证

```bash
pytest
ruff check .
python scripts/demo.py
```

`test_framework.py` 仍可作为旧课程材料的兼容入口：

```bash
python test_framework.py
```

## 环境契约

训练与评估共享相同的视觉与时间预处理：

- ALE 内建 `frameskip=1`
- 项目中的 `MaxAndSkipEnv` 执行一次 `frame_skip=4`
- 观测缩放到 84×84、灰度化并堆叠 4 帧
- 像素在环境和 replay buffer 中保持 `uint8`，采样后才转为浮点

训练环境额外使用 episodic-life 和符号奖励裁剪。评估环境不使用这两项训练技巧，因此报告的是完整游戏上的原始奖励。DQN 和 PPO 的评估动作都是确定性的。

## 训练

单任务：

```bash
python scripts/train_single.py \
  --game Pong-v5 \
  --algorithm dqn \
  --steps 500000
```

顺序持续学习：

```bash
python scripts/train_continual.py \
  --games Pong-v5 Breakout-v5 SpaceInvaders-v5 \
  --algorithm ppo \
  --steps-per-game 50000
```

加入 EWC：

```bash
python scripts/train_continual.py \
  --games Pong-v5 Breakout-v5 SpaceInvaders-v5 \
  --algorithm ppo \
  --use-ewc \
  --ewc-lambda 0.4 \
  --steps-per-game 50000
```

联合多任务训练：

```bash
python scripts/train_multitask.py \
  --games Pong-v5 Breakout-v5 SpaceInvaders-v5 \
  --algorithm dqn \
  --steps 150000
```

训练视频默认关闭，可通过 `--save-video` 开启。DQN 默认在 10,000 个 agent steps 后开始更新，使每任务 50,000 步的课程配置能够实际发生学习。持续学习入口会把阶段×任务矩阵和逐任务遗忘量写入 `outputs/continual/.../continual_evaluation.json`；评估次数可用 `--eval-episodes` 调整。

## 评估

单任务 checkpoint：

```bash
python scripts/evaluate.py \
  --mode single \
  --model checkpoints/single/Pong-v5_dqn.pt \
  --algorithm dqn \
  --game Pong-v5 \
  --episodes 10 \
  --json-out outputs/single/Pong-v5_dqn/eval/metrics.json
```

持续学习 checkpoint：

```bash
python scripts/evaluate.py \
  --mode continual \
  --model checkpoints/continual/ppo_ewcTrue.pt \
  --algorithm ppo \
  --games Pong-v5 Breakout-v5 SpaceInvaders-v5 \
  --ewc \
  --episodes 5 \
  --json-out outputs/continual/ppo_ewcTrue/eval/metrics.json
```

批处理脚本同时提供 GPU 默认版和 `_cpu.sh` 版本。

## 如何解释 EWC 结果

EWC 是稳定性正则项，不是任务冲突求解器。它可以限制对旧任务重要参数的漂移，因此可能减缓遗忘；当新旧任务在共享骨干上需要相反的更新方向时，提高 EWC 强度通常只会把问题转化为“旧任务保留更多、但新任务学得更慢”。多头输出解决了动作空间不同的问题，但没有消除共享表征中的梯度冲突。

持续学习实验应保存阶段×任务得分矩阵 `R[i, j]`，并至少分开报告：

- 旧任务保持：任务 `j` 的历史最佳得分与最终得分之差
- 新任务可塑性：首次完成任务 `j` 训练后的 `R[j, j]`
- 联合折中：同一方法的保持与可塑性，而不是只看最终平均分

不同 Atari 游戏的原始奖励尺度不同，不应直接把它们相加后解释为单一性能指标。随机数据上的训练损失也不能证明发生了遗忘。下一阶段应先固定 seeds、评估预算和得分矩阵，再比较 EWC 与能够处理冲突的候选方法，例如小型 episodic replay；只有在基线协议稳定后再考虑更复杂的梯度投影或蒸馏方法。

## 代码结构

```text
algorithms/               DQN、PPO、多头模型与 EWC
environments/             Atari 环境及训练/评估预处理
utils/                    replay/rollout buffer 与可视化
scripts/train_single.py   单任务训练
scripts/train_continual.py 顺序持续学习
scripts/train_multitask.py 联合多任务训练
scripts/evaluate.py       checkpoint 评估
tests/                    正确性与兼容性回归测试
pyproject.toml            唯一依赖与工具配置源
```

## 当前已知边界

- 训练循环仍是教学用途的同步、单环境实现；尚未开始性能优化。
- replay buffer 仍按 transition 保存 `state` 和 `next_state`，后续可在不引入额外加速库的前提下优化布局。
- 简化的环境接口目前将 Gymnasium 的 `terminated` 与 `truncated` 合并为 `done`；在开展时间限制敏感的实验前应进一步拆分。
- 现有 EWC 使用对角经验 Fisher，它不能表达参数之间的相关性，也不保证解决正向迁移或任务冲突。

## 参考资料

- [DQN](https://www.nature.com/articles/nature14236)
- [PPO](https://arxiv.org/abs/1707.06347)
- [EWC](https://arxiv.org/abs/1612.00796)
- [Gymnasium](https://gymnasium.farama.org/)
- [TorchRL](https://docs.pytorch.org/rl/)

## License

MIT
