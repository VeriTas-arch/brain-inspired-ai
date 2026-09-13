# 少样本学习与 MAML

模型无关元学习（MAML）尝试学到一组容易适应新任务的初始参数。本课将它与一阶近似 FO-MAML 比较，观察模型用少量带标签图像更新后，能否识别新的类别。

阅读前需要熟悉卷积网络和自动求导，并了解训练集、验证集与测试集的划分。

## 准备环境

沿用课程的 `brain_ai` 环境。首次使用时，先运行 `conda create -n brain_ai python=3.12`，然后激活环境：

```bash
conda activate brain_ai
```

在已激活的环境中安装依赖：

<!-- course-dependencies -->
```bash
python -m pip install notebook ipykernel ipython numpy matplotlib torch torchvision pillow
```
<!-- /course-dependencies -->

## 运行 Notebook

用 VS Code 打开解压后的课程目录，并安装 Microsoft 提供的 Python 和 Jupyter 扩展。打开 [meta_learning.ipynb](meta_learning.ipynb)，选择 `brain_ai` 内核，从顶部按顺序运行。

默认使用 Omniglot，首次运行会下载到 `data/omniglot-py/`。每个任务从若干类别中抽取图像，一部分用于调整参数，另一部分用于评估。训练、验证和测试任务来自不同类别。

先运行默认设置，再尝试改变每个任务的类别数或每类用于学习的图像数。MAML 和 FO-MAML 默认各进行 2000 次外循环更新，即反复采样一批任务来更新共享的初始参数。

将 `DATASET_NAME` 改为 `"mini-imagenet"` 后，可改用 Mini-ImageNet。首次使用会下载约 1.8 GB 的缓存到 `data/mini-imagenet/`，类别划分和输入尺寸见 Notebook。

## 查看结果

训练曲线显示采样任务上的损失与准确率，最终评估比较模型在新类别上适应前后的平均准确率，并给出置信区间。对照两种方法时，同时查看准确率和运行时间。
