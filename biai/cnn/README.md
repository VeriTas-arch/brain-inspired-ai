# 卷积神经网络与图像分类

本课先在 MNIST 上训练卷积神经网络（CNN），再在 CIFAR-10 上比较多层感知机（MLP）、CNN 和残差网络（ResNet）。除了分类准确率，还会查看卷积层的特征图。

阅读前需要熟悉 MLP 的前向计算和训练循环，并了解训练集、验证集与测试集的用途。

## 准备环境

沿用课程的 `brain_ai` 环境。首次使用时，先运行 `conda create -n brain_ai python=3.12`，然后激活环境：

```bash
conda activate brain_ai
```

在已激活的环境中安装依赖：

<!-- course-dependencies -->
```bash
python -m pip install ipykernel ipython numpy matplotlib torch torchvision
```
<!-- /course-dependencies -->

## 运行 Notebook

用 VS Code 打开解压后的课程目录，并安装 Microsoft 提供的 Python 和 Jupyter 扩展。打开 [cnn.ipynb](cnn.ipynb)，选择 `brain_ai` 内核，从顶部按顺序运行。

课程包已经包含 MNIST 和 CIFAR-10，默认从课程根目录的 `data/` 读取，不需要联网。CIFAR-10 以无损压缩的 `tar.xz` 保存，第一次运行到 CIFAR-10 单元格时自动解包，之后直接复用解包后的数据。若要与网络来源重新获取的数据对照，将 Notebook 顶部的 `USE_NETWORK_DATA` 改为 `True`；数据会下载到 `data/network/`，不会覆盖包内副本。先运行 MNIST 实验，了解卷积层如何处理图像，再运行 CIFAR-10 上的三种网络对照。

## 查看结果

学习曲线显示训练与验证指标的变化，测试准确率用于比较训练后的模型。特征图展示卷积层对同一幅图像的不同响应，可以结合图像中的边缘和局部纹理阅读。

## 源码与反馈

本课程的源码与构建工具见 [brain-inspired-ai](https://github.com/VeriTas-arch/brain-inspired-ai)。

如果想了解课程材料的组织方式，以及如何从源码生成每节课的压缩包，可以查看[构建说明](https://github.com/VeriTas-arch/brain-inspired-ai/blob/main/BUILDING.md)。欢迎提出问题和改进建议。
