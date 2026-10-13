# Amari–Hopfield 联想记忆模型

本课用两个 28 × 28 的二值图像演示联想记忆，像素编码为 -1 或 1。网络按照 Hebb 学习规则把存储模式写入权重矩阵，再从像素翻转、局部遮挡或混合得到的初始状态出发恢复图像。Notebook 展示模式恢复结果、能量变化，并正式比较异步更新与同步更新。

阅读前需要了解向量外积、全连接网络和 Hebb 学习规则。

## 准备环境

沿用课程的 `brain_ai` 环境。首次使用时，先运行 `conda create -n brain_ai python=3.12`，然后激活环境：

```bash
conda activate brain_ai
```

在已激活的环境中安装依赖：

<!-- course-dependencies -->
```bash
python -m pip install ipykernel ipython numpy matplotlib torch
```
<!-- /course-dependencies -->

## 运行 Notebook

用 VS Code 打开解压后的课程目录，并安装 Microsoft 提供的 Python 和 Jupyter 扩展。打开 [amari_hopfield_model.ipynb](amari_hopfield_model.ipynb)，选择 `brain_ai` 内核，从顶部按顺序运行。

课程包已经包含演示使用的两组 `.npy` 图像，不需要联网。Notebook 调用 [hopfield_model.py](hopfield_model.py) 中的网络实现；也可以直接运行：

```bash
python -m biai.amari_hopfield_model.hopfield_model
```

## 观察重点

比较原图、初始状态和恢复结果，并检查异步更新过程中能量是否单调不增。在固定示例中，两种规则每轮计算相同数量的局部场；异步更新收敛到存储模式，同步更新进入二周期。不明确的初始状态没有对应的目标图像；它的收敛结果取决于它落入哪个吸引域。

可以调整翻转概率、遮挡范围或存储模式数量，观察恢复质量何时开始下降。

## 源码与反馈

本课程的源码与构建工具见 [brain-inspired-ai](https://github.com/VeriTas-arch/brain-inspired-ai)。欢迎提出问题和改进建议。
