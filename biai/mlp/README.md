# 多层感知机与学习规则

本课在 MNIST 手写数字上比较单层分类器和带隐藏层的多层感知机（MLP）。随后改变激活函数，并比较反向传播与两种局部学习规则 Oja、GHA：前者根据分类误差更新权重，后两者利用输入与神经元活动学习特征。

阅读前需要熟悉张量、梯度和 PyTorch 的参数更新步骤。

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

用 VS Code 打开解压后的课程目录，并安装 Microsoft 提供的 Python 和 Jupyter 扩展。打开 [mlp.ipynb](mlp.ipynb)，选择 `brain_ai` 内核，从顶部按顺序运行。

课程包已经包含 MNIST，默认直接从课程根目录的 `data/` 读取，不需要联网。若要与网络来源重新获取的数据对照，将 Notebook 顶部的 `USE_NETWORK_DATA` 改为 `True`；数据会下载到 `data/network/`，不会覆盖包内副本。程序从训练数据中划出验证集，每轮训练后计算验证指标，训练结束后评估测试集。

## 查看结果

先比较单层网络与 MLP 的学习曲线，再查看激活函数和学习规则的对照实验。修改参数时，一次改变一个因素，便于判断差异来自哪里。

曲线和汇总表都显示在 Notebook 中。分类准确率反映特征能否用于识别数字，特征之间的相似程度则帮助判断不同隐藏单元是否学到了重复内容。

## 源码与反馈

本课程的源码与构建工具见 [brain-inspired-ai](https://github.com/VeriTas-arch/brain-inspired-ai)。

如果想了解课程材料的组织方式，以及如何从源码生成每节课的压缩包，可以查看[构建说明](https://github.com/VeriTas-arch/brain-inspired-ai/blob/main/BUILDING.md)。欢迎提出问题和改进建议。
