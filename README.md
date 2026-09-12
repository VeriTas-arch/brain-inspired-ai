# 脑启发的人工智能

Brain-Inspired Artificial Intelligence（BIAI）课程代码。每个主题对应一份 Jupyter Notebook，从神经网络的学习规则讲到持续学习、元学习和强化学习。

项目名称为 `biai-course`，Python 包名为 `biai`。阅读前需要了解 Python、张量运算和梯度下降。

## 课程目录

| 主题 | Notebook | 学习内容 |
| --- | --- | --- |
| 多层感知机 | [MLP](biai/mlp/mlp.ipynb) | 单层与双层网络、激活函数对照、BP 与 Oja/GHA 特征学习 |
| 卷积神经网络 | [CNN](biai/cnn/cnn.ipynb) | MNIST/CIFAR-10 的 MLP、CNN、ResNet 对照与特征图 |
| 持续学习 | [MNIST 与 EWC](biai/continual_mnist/continual_mnist.ipynb) | Task-IL/Class-IL 独立训练、EWC 强度、固定容量回放与稳定性/可塑性 |
| 元学习 | [MAML 与 FO-MAML](biai/meta_learning/meta_learning.ipynb) | Omniglot/Mini-ImageNet、MAML/FO-MAML、N-way K-shot 对照 |
| 强化学习 | [Atari](biai/atari/atari.ipynb) | DQN、PPO、联合与顺序训练、EWC、GPM |

前四份 Notebook 直接展示模型与训练步骤。Atari Notebook 调用配套 Python 实现，并展示已有的[参考结果](assets/README.md)。

## 安装与阅读

使用 Python 3.12 或更新版本，在仓库根目录安装：

```bash
python -m pip install -e .
python -m jupyterlab
```

也可以用 VS Code 打开 Notebook，选择安装了本项目依赖的 Python 环境。按单元格顺序阅读和执行；数据与素材路径由已安装的 `biai` 包定位，无需切换 Notebook 的工作目录。

MLP、CNN、MNIST 持续学习和元学习 Notebook 的训练单元格会实际运行。Atari Notebook 的训练调用默认使用 `--dry-run`，只预览命令；参考曲线和动图无需重新训练即可查看。

## 数据与运行结果

MNIST、CIFAR-10 和 Omniglot 在首次执行对应数据单元格时下载，之后复用根目录 `data/` 下的缓存。元学习 Notebook 还支持 Mini-ImageNet（RGB 84×84，64/16/20 个训练/验证/测试类别），选择后从 [learn2learn 的固定数据记录](https://zenodo.org/records/7978538) 下载约 1.8 GB 缓存。完整 N-way/K-shot 矩阵在该 Notebook 的批量实验节启用。

Omniglot 使用 [torchvision 的官方数据接口](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.Omniglot.html)：`background` 用于训练，`evaluation` 用于评估。图片只存入本地数据缓存，克隆课程仓库时不包含数据集。下载来源与数据说明见 [Omniglot 原始项目](https://github.com/brendenlake/omniglot)。

- `data/`：本地数据缓存，不进入 Git。
- `results/<case>/`：本地训练记录、模型和录像，不进入 Git。现有 Atari 案例沿用原名称；新增课程的持久化结果使用主题前缀，例如 `mlp-…`。
- `assets/`：进入 Git 的教材参考结果、图表和动图。目前提供 Atari 参考结果；新课参考素材按主题放入子目录。
- `ref/`：本地原始课程材料，不参与安装，也不进入 Git。运行课程无需解压其中的 Omniglot 压缩包。

## 运行 Atari 案例

Atari 包含单任务 DQN/PPO、联合多任务训练、普通顺序训练、顺序训练加 EWC，以及顺序训练加 GPM。单任务使用 Pong 和 Breakout；持续学习依次使用 Pong、Breakout 和 Space Invaders。

以下命令在仓库根目录执行。先预览全部 12 个案例：

```bash
python -m biai.atari.scripts.run_experiments train teaching --dry-run
```

用短训练检查训练、模型保存和独立评估：

```bash
python -m biai.atari.scripts.run_experiments train teaching --smoke
```

检查通过后运行完整训练：

```bash
python -m biai.atari.scripts.run_experiments train teaching
```

若要在三个游戏上比较单任务、联合和顺序训练的效果，使用每游戏预算匹配的十四案例矩阵（默认每游戏 500,000 步）：

```bash
python -m biai.atari.scripts.run_experiments train teaching --matched-budget --dry-run
```

先加 `--smoke` 并去掉 `--dry-run` 验证完整流程，短训练保存到 `results/atari-matched-smoke/`；再运行正式预算，保存到 `results/atari-matched/`。可在 Atari Notebook 的预算匹配实验节读取结果。

教学入口默认使用种子 `0`，DQN 和 PPO 均使用 8 个环境采样。训练预算和已有分数见[参考结果](assets/README.md)，单独运行某个案例可使用 Atari Notebook 中的入口。

短训练结果保存在 `results/smoke/`。重跑已有案例时加上 `--force`；只有训练与评估成功后才替换选定案例的旧结果。训练和 `--force` 都不会覆盖 `assets/` 中的参考素材。其他参数见 `python -m biai.atari.scripts.run_experiments --help`。

## 阅读代码

| 位置 | 内容 |
| --- | --- |
| [biai/](biai/) | 按课程主题组织的 Notebook 与 Python 代码 |
| [biai/paths.py](biai/paths.py) | 仓库、数据、结果和参考素材的位置 |
| [biai/reproducibility.py](biai/reproducibility.py) | 共用的随机种子与确定性设置 |
| [Atari 算法](biai/atari/algorithms/) | DQN、PPO、EWC 和 GPM |
| [Atari 环境](biai/atari/environments/) | 游戏环境与图像预处理 |
| [Atari 训练组件](biai/atari/training/) | 经验回放、采样、网络更新和回合评估 |
| [Atari 运行入口](biai/atari/scripts/) | 训练、评估、绘图和性能测试 |
| [tests/](tests/) | 课程 Notebook 检查与 Atari 回归测试 |

Atari 正式配置由 [run_experiments.py](biai/atari/scripts/run_experiments.py) 中的 `build_teaching_jobs` 定义；数值示例对应 [tutorial_examples.py](biai/atari/scripts/tutorial_examples.py)。
