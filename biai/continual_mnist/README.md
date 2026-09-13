# MNIST 持续学习

本课把手写数字分成五个任务，让模型依次学习。实验比较普通顺序训练、弹性权重巩固（EWC）和样本回放，观察学习新任务时，旧任务的准确率如何变化。

阅读前需要能够训练和评估图像分类器，并了解交叉熵损失与梯度下降。

## 准备环境

沿用课程的 `brain_ai` 环境。首次使用时，先运行 `conda create -n brain_ai python=3.12`，然后激活环境：

```bash
conda activate brain_ai
```

在已激活的环境中安装依赖：

<!-- course-dependencies -->
```bash
python -m pip install notebook ipykernel ipython numpy matplotlib torch torchvision
```
<!-- /course-dependencies -->

## 运行 Notebook

用 VS Code 打开解压后的课程目录，并安装 Microsoft 提供的 Python 和 Jupyter 扩展。打开 [continual_mnist.ipynb](continual_mnist.ipynb)，选择 `brain_ai` 内核，从顶部按顺序运行。

首次运行会下载 MNIST，缓存保存在课程根目录的 `data/`。各方法使用相同的初始化、任务顺序和训练预算。

实验包含两种评估方式：任务增量学习（Task-IL）在预测时提供任务编号，只需区分该任务的两个数字；类别增量学习（Class-IL）不提供任务编号，需要从已经学过的所有数字中判断。

## 查看结果

每学完一个任务，程序都会评估此前学过的任务，并在 Notebook 中显示学习曲线和准确率矩阵。阅读矩阵时，先看新任务刚学完的准确率，再沿同一任务查看后续阶段的变化，这样可以区分“没有学好”和“学会后遗忘”。
