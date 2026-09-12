# 脑启发的人工智能

本仓库配套 Brain-Inspired Artificial Intelligence（BIAI）课程，包含五份 Jupyter Notebook，依次介绍神经网络的学习规则、图像分类、持续学习、元学习和强化学习。

阅读前需要了解 Python、张量运算和梯度下降。

## 课程目录

| 主题 | Notebook | 学习内容 |
| --- | --- | --- |
| 多层感知机 | [MLP](biai/mlp/mlp.ipynb) | 单层与双层网络、激活函数对照、BP 与 Oja/GHA 特征学习 |
| 卷积神经网络 | [CNN](biai/cnn/cnn.ipynb) | MNIST/CIFAR-10 的 MLP、CNN、ResNet 对照与特征图 |
| 持续学习 | [MNIST 持续学习](biai/continual_mnist/continual_mnist.ipynb) | 比较 Task-IL 与 Class-IL，观察 EWC 和样本回放对新旧任务的影响 |
| 元学习 | [MAML 与 FO-MAML](biai/meta_learning/meta_learning.ipynb) | Omniglot/Mini-ImageNet、MAML/FO-MAML、N-way K-shot 对照 |
| 强化学习 | [Atari](biai/atari/atari.ipynb) | DQN、PPO、联合与顺序训练、EWC、GPM |

前四份 Notebook 直接展示模型和训练循环。Atari 的训练代码较长，放在配套 Python 文件中，Notebook 负责讲解方法、调用训练和展示[参考结果](assets/README.md)。

## 安装与阅读

使用 Python 3.12 或更新版本，在仓库根目录安装依赖：

```bash
python -m pip install -e .
```

用 VS Code 打开仓库目录，并安装 Python 和 Jupyter 扩展。打开课程目录中的 Notebook，选择刚才安装依赖的 Python 环境作为内核，然后从顶部按顺序执行单元格。代码会自动找到仓库中的数据和素材，无需手动切换工作目录。

前四份 Notebook 执行到训练单元格时会开始训练。Atari 默认使用 `--dry-run` 预览命令，移除该选项后才启动训练；已有曲线和动图可以直接查看。

## 数据与运行结果

MNIST、CIFAR-10 和 Omniglot 在首次执行数据加载单元格时下载，之后直接读取 `data/` 下的缓存。数据集不随仓库分发。

Omniglot 通过 [torchvision](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.Omniglot.html) 下载。代码将 `background` 中约 80% 的字符类别用于训练，其余用于验证；`evaluation` 中的类别用于最终测试。数据背景见 [Omniglot 原始项目](https://github.com/brendenlake/omniglot)。

元学习默认使用 Omniglot。将 `DATASET_NAME` 改为 `"mini-imagenet"` 后，会从 [learn2learn 提供的数据](https://zenodo.org/records/7978538) 下载约 1.8 GB 的 Mini-ImageNet 图像缓存。图像为 84 × 84 的 RGB 彩色图，训练、验证和测试分别包含 64、16、20 个类别。不同 N-way/K-shot 设置的运行方式见该 Notebook 的比较实验一节。

运行时主要会用到以下目录：

- `data/`：下载的数据集缓存。
- `results/`：本地保存的训练记录、模型和录像；Atari 为每个案例建立一个子目录。
- `assets/`：随仓库提供的 Atari 参考结果、图表和动图。

`data/` 和 `results/` 已被 Git 忽略，自己的训练结果不会覆盖 `assets/` 中的参考素材。

## 运行 Atari 案例

默认案例分别用 DQN 和 PPO 进行单任务、联合与顺序训练，顺序训练另有 EWC 和 GPM 两组对照。单任务使用 Pong 和 Breakout；联合训练使用三个游戏，顺序训练则依次学习 Pong、Breakout 和 Space Invaders。

以下命令在仓库根目录执行。先预览全部 12 个案例：

```bash
python -m biai.atari.scripts.run_experiments train teaching --dry-run
```

先用短训练检查训练、模型保存和独立评估流程，结果保存在 `results/smoke/`：

```bash
python -m biai.atari.scripts.run_experiments train teaching --smoke
```

检查通过后运行完整训练，结果按案例保存在 `results/` 下：

```bash
python -m biai.atari.scripts.run_experiments train teaching
```

上面 12 个案例的训练步数不完全相同。若想在相同交互次数下比较训练方式，可以加入 `--matched-budget`：它增加 Space Invaders 的单任务对照，共运行 14 个案例，并让每个游戏在各训练方式下都获得 500,000 步交互。

```bash
python -m biai.atari.scripts.run_experiments train teaching --matched-budget --dry-run
```

这条命令仍只预览。将 `--dry-run` 换成 `--smoke` 可执行短训练，结果保存在 `results/atari-matched-smoke/`；直接移除 `--dry-run` 则执行完整训练，结果保存在 `results/atari-matched/`。完成后可在 Atari Notebook 的相同交互预算对照一节查看结果。

这些命令默认使用随机种子 `0`，DQN 和 PPO 均使用 8 个并行环境。各案例的训练步数和已有分数见[参考结果](assets/README.md)；若只想运行其中一个案例，可以使用 Atari Notebook 中的对应命令。

已有结果默认保留。需要重跑时加上 `--force`，程序会在所选案例的训练与评估都成功后替换旧结果。其他参数见 `python -m biai.atari.scripts.run_experiments --help`。

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

想了解 Atari 各案例使用了哪些参数，可以查看 [run_experiments.py](biai/atari/scripts/run_experiments.py) 中的 `build_teaching_jobs`；Notebook 附录中的数值示例实现在 [tutorial_examples.py](biai/atari/scripts/tutorial_examples.py) 中。
