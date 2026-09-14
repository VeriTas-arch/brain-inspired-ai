# 脑启发的人工智能

本仓库配套 Brain-Inspired Artificial Intelligence（BIAI）课程。从环境配置与 PyTorch 入门开始，依次学习神经网络的学习规则、图像分类、持续学习、元学习和强化学习。

环境配置与 PyTorch 入门部分需要基本的 Python 知识。后续课程会用到张量运算、自动求导和神经网络训练。

## 课程目录

| 主题与说明 | Notebook | 学习内容 |
| --- | --- | --- |
| [环境配置](biai/intro/README.md) | [PyTorch 入门](biai/intro/intro.ipynb) | Conda、Jupyter、服务器 Git 克隆、张量、自动求导与小网络训练 |
| [多层感知机](biai/mlp/README.md) | [MLP](biai/mlp/mlp.ipynb) | 单层与双层网络、激活函数对照、BP 与 Oja/GHA 特征学习 |
| [卷积神经网络](biai/cnn/README.md) | [CNN](biai/cnn/cnn.ipynb) | MNIST/CIFAR-10 的 MLP、CNN、ResNet 对照与特征图 |
| [持续学习](biai/continual_mnist/README.md) | [MNIST 持续学习](biai/continual_mnist/continual_mnist.ipynb) | 比较 Task-IL 与 Class-IL，观察 EWC 和样本回放对新旧任务的影响 |
| [元学习](biai/meta_learning/README.md) | [MAML 与 FO-MAML](biai/meta_learning/meta_learning.ipynb) | Omniglot/Mini-ImageNet、MAML/FO-MAML、N-way K-shot 对照 |
| [强化学习](biai/atari/README.md) | [Atari](biai/atari/atari.ipynb) | DQN、PPO、联合与顺序训练、EWC、GPM |

PyTorch 入门、MLP、CNN、持续学习和元学习的 Notebook 直接展示模型和训练步骤。Atari 的训练代码较长，放在配套 Python 文件中，Notebook 负责讲解方法、调用训练和展示[参考结果](assets/README.md)。

## 安装与阅读

### 使用每课压缩包

每节课单独下载并解压，先阅读包内的 `README.md`。首次使用按[环境配置说明](biai/intro/README.md)创建 `brain_ai` 环境，后续各课沿用这个环境，按本课说明补充依赖即可。

用 VS Code 打开解压目录，打开根目录的 Notebook 并选择 `brain_ai` 内核。Notebook 与 `biai/` 并列，不需要运行 `pip install -e .`，也不需要克隆课程仓库。换课时打开新的课程目录并启动新的内核；数据和结果保留在各自的课程目录中。

### 使用完整仓库

完整仓库中的 Notebook 按主题存放。使用 Python 3.12 或更新版本，在仓库根目录安装依赖和本地包：

```bash
python -m pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

用 VS Code 打开仓库目录，并安装 Python 和 Jupyter 扩展。打开课程目录中的 Notebook，选择刚才安装依赖的 Python 环境作为内核，然后从顶部按顺序执行单元格。代码会自动找到仓库中的数据和素材，无需手动切换工作目录。

运行 PyTorch 入门、图像分类、持续学习和元学习的训练单元格时，会直接开始训练。Atari 默认使用 `--dry-run` 预览命令，移除该选项后才启动训练；已有曲线和动图可以直接查看。

## 数据与运行结果

MNIST、CIFAR-10 和 Omniglot 在首次执行数据加载单元格时下载，之后直接读取 `data/` 下的缓存。数据集不随仓库分发。

Omniglot 通过 [torchvision](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.Omniglot.html) 下载。代码将 `background` 中约 80% 的字符类别用于训练，其余用于验证；`evaluation` 中的类别用于最终测试。数据背景见 [Omniglot 原始项目](https://github.com/brendenlake/omniglot)。

元学习默认使用 Omniglot。将 `DATASET_NAME` 改为 `"mini-imagenet"` 后，会从 [learn2learn 提供的数据](https://zenodo.org/records/7978538) 下载约 1.8 GB 的 Mini-ImageNet 图像缓存。图像为 84 × 84 的 RGB 彩色图，训练、验证和测试分别包含 64、16、20 个类别。不同 N-way/K-shot 设置的运行方式见该 Notebook 的比较实验一节。

运行时主要会用到以下目录：

- `data/`：下载的数据集缓存。
- `results/`：本地保存的训练记录、模型和录像；Atari 为每个案例建立一个子目录。
- `assets/`：随仓库提供的 Atari 参考结果、图表和动图。

`data/` 和 `results/` 已被 Git 忽略，自己的训练结果不会覆盖 `assets/` 中的参考素材。

## 阅读代码

从课程目录中的 README 和 Notebook 开始阅读，具体的代码入口见各课说明。仓库的主要代码与测试位置如下：

| 位置 | 内容 |
| --- | --- |
| [biai/](biai/) | 按课程主题组织的 Notebook 与 Python 代码 |
| [biai/paths.py](biai/paths.py) | 仓库、数据、结果和参考素材的位置 |
| [biai/reproducibility.py](biai/reproducibility.py) | 共用的随机种子与确定性设置 |
| [tests/](tests/) | Notebook 检查与代码测试 |

## 源码与反馈

本课程的源码与构建工具见 [brain-inspired-ai](https://github.com/VeriTas-arch/brain-inspired-ai)。

如果想了解课程材料的组织方式，以及如何从源码生成每节课发布的压缩包，可以查看[构建说明](BUILDING.md)。欢迎提出问题和改进建议。
