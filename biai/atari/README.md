# Atari 强化学习

本课先用深度 Q 网络（DQN）和近端策略优化（PPO）学习单个游戏，再比较多个游戏上的联合训练与顺序训练。顺序学习部分加入对重要参数施加约束的 EWC 和限制梯度方向的 GPM，观察旧游戏得分能否保持。

阅读前需要熟悉神经网络训练。状态、动作、奖励和回合等强化学习概念会在 Notebook 中介绍。

## 准备环境

沿用课程的 `brain_ai` 环境。首次使用时，先运行 `conda create -n brain_ai python=3.12`，然后激活环境：

```bash
conda activate brain_ai
```

在已激活的环境中安装依赖：

<!-- course-dependencies -->
```bash
python -m pip install ipykernel ipython numpy matplotlib torch ale-py gymnasium imageio-ffmpeg opencv-python pillow tqdm
```
<!-- /course-dependencies -->

## 运行 Notebook

用 VS Code 打开解压后的课程目录，并安装 Microsoft 提供的 Python 和 Jupyter 扩展。打开 [atari.ipynb](atari.ipynb)，选择 `brain_ai` 内核，从顶部按顺序运行。

Notebook 会展示已有曲线和动图，训练调用默认带有 `--dry-run`，只打印命令。先阅读实验设置和参考结果，需要重新训练时再执行下方命令。

## 参考素材与本地结果

`assets/` 保存参考分数、图表和动图，各案例的预算及素材来源见[参考结果说明](../../assets/README.md)。本地训练记录、模型和录像写入课程根目录的 `results/`，与参考素材分开保存。

## 运行训练案例

以下命令在课程根目录的终端中执行。默认配置包含 12 个案例，覆盖单任务、联合训练和顺序训练；顺序训练包含普通训练、EWC、GPM 三组。先预览案例与参数：

```bash
python -m biai.atari.scripts.run_experiments train teaching --dry-run
```

用短训练检查环境、模型保存和评估是否正常：

```bash
python -m biai.atari.scripts.run_experiments train teaching --smoke
```

完成后移除 `--smoke`，即可运行完整训练。命令会自动选择可用设备，并使用多个并行环境；完整训练适合在有 GPU 的服务器上运行。

默认案例的训练步数不完全相同。加入 `--matched-budget` 后，会增加 Space Invaders 的单任务对照，共 14 个案例，每个游戏在各训练方式下都使用 50 万步交互：

```bash
python -m biai.atari.scripts.run_experiments train teaching --matched-budget --dry-run
```

这条命令也只预览。将 `--dry-run` 换成 `--smoke` 可运行短训练，移除它则运行完整对照。已有结果默认保留；需要重跑时加上 `--force`，所选案例的训练与评估成功后才会替换对应结果。

比较结果时，分别查看每个游戏的得分。顺序训练还需要对照各阶段的表现，判断方法是否在保留旧游戏能力的同时学会了新游戏。

## 阅读代码

Notebook 对应的 Python 代码位于 `biai/atari/`，按以下目录组织：

| 目录 | 内容 |
| --- | --- |
| `algorithms/` | DQN、PPO、EWC 和 GPM |
| `environments/` | 游戏环境与图像预处理 |
| `training/` | 经验回放、采样、网络更新和回合评估 |
| `scripts/` | 训练、评估、绘图和性能测试入口 |

各案例的参数在 [run_experiments.py](scripts/run_experiments.py) 的 `build_teaching_jobs` 中定义。Notebook 附录中的数值示例见 [tutorial_examples.py](scripts/tutorial_examples.py)。

## 源码与反馈

本课程的源码与构建工具见 [brain-inspired-ai](https://github.com/VeriTas-arch/brain-inspired-ai)。

如果想了解课程材料的组织方式，以及如何从源码生成每节课的压缩包，可以查看[构建说明](https://github.com/VeriTas-arch/brain-inspired-ai/blob/main/BUILDING.md)。欢迎提出问题和改进建议。
