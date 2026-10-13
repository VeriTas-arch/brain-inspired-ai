# Amari–Hopfield 双神经元相平面

本课用双神经元模型观察激活函数、增益与耦合权重如何影响状态演化。五组实验分别展示相平面、时间序列和能量景观，比较不同参数下的平衡点与收敛过程。sigmoid 与 tanh 两组均将相平面和时间序列并排显示，子图标题标明各自的参数。

阅读前需要了解常微分方程、零流线、平衡点和 Jacobian 矩阵。

## 准备环境

沿用课程的 `brain_ai` 环境。首次使用时，先运行 `conda create -n brain_ai python=3.12`，然后激活环境：

```bash
conda activate brain_ai
```

在已激活的环境中安装依赖：

<!-- course-dependencies -->
```bash
python -m pip install ipykernel ipython numpy matplotlib torch scipy
```
<!-- /course-dependencies -->

## 运行 Notebook

用 VS Code 打开解压后的课程目录，并安装 Microsoft 提供的 Python 和 Jupyter 扩展。打开 [amari_hopfield_phase_plane.ipynb](amari_hopfield_phase_plane.ipynb)，选择 `brain_ai` 内核，从顶部按顺序运行。

Notebook 调用 [phase_plane.py](phase_plane.py) 中的双神经元方程、数值积分、平衡点搜索和绘图函数。所有实验只依赖 NumPy、SciPy 和 Matplotlib，不需要联网。图片直接显示在 Notebook 中，不另存文件。

## 实验内容

| 实验 | 模型与观察内容 |
| --- | --- |
| 1 | 低增益 arctan：两个稳定结点与一个鞍点 |
| 2 | 高增益 arctan：增益对零流线形状的影响 |
| 3 | sigmoid：正耦合相平面与提高增益后的时间序列 |
| 4 | tanh：相互抑制的相平面与临界反馈下的缓慢衰减 |
| 5 | Hopfield 能量景观：两个对称的能量低谷 |

## 观察重点

比较低增益和高增益下零流线、平衡点与吸引域的变化；区分稳定结点、鞍点和非双曲平衡点；再联系状态轨迹与能量低谷。能量图采用特定的 arctan 参数，不能直接用于解释其他参数设置。

## 源码与反馈

本课程的源码与构建工具见 [brain-inspired-ai](https://github.com/VeriTas-arch/brain-inspired-ai)。欢迎提出问题和改进建议。
