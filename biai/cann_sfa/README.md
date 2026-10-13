# 连续吸引子网络与慢适应

本课用一群神经元的活动峰表示圆环上的位置。外部目标移动时，活动峰可以平稳地跟随，也可以在目标附近来回摆动；撤去目标后，活动峰还可能继续传播。我们用连续吸引子神经网络（CANN）和慢适应（SFA）模型模拟这三种情况。

阅读前需要了解 NumPy 数组、常微分方程和 Matplotlib 绘图。

## 准备环境

沿用课程的 `brain_ai` 环境。首次使用时创建环境，然后激活并安装依赖：

```bash
conda create -n brain_ai python=3.12
conda activate brain_ai
```

<!-- course-dependencies -->
```bash
python -m pip install ipykernel ipython numpy matplotlib torch
```
<!-- /course-dependencies -->

默认仿真使用 CPU，不需要 GPU 或外部数据集。

## 运行 Notebook

在 VS Code 中安装 Python 和 Jupyter 扩展，打开课程目录和 [cann_sfa.ipynb](cann_sfa.ipynb)，选择 `brain_ai` 内核，从上到下执行。独立课程包无需安装 `biai-course`；使用完整仓库时，先在仓库根目录运行 `python -m pip install -e .`。

Notebook 中的三组实验都有静态图和可播放的动画。静态图显示活动峰位置、跟踪误差和活动热图，动画显示活动峰与输入峰的形状。行波实验撤去输入后，中间的图改为显示活动峰走过的累计位置。最后一组对照去掉慢反馈，比较跟踪误差是否仍然振荡。

参数放在 `configs` 单元格中，每组使用 256 个神经元、0.1 ms 步长和 1400 ms 时长。修改参数后，从这个单元格开始重新运行即可。

## 阅读代码

[cann_sfa_triple_regimes.py](cann_sfa_triple_regimes.py) 提供仿真和绘图：

- `RegimeConfig` 与 `REGIMES`：参数和三组示例。
- `CANN1DSFA.update`：周期卷积、指数 Euler 更新和活动峰位置。
- `simulate_regime`：从零状态运行，返回轨迹、输入和跟踪误差。
- `plot_regime` 与 `animate_regime`：绘制静态图和动画，在 Notebook 中显示。

数值更新方法和周期卷积的计算见 Notebook 附录，课末提供了几项可继续尝试的参数实验。
