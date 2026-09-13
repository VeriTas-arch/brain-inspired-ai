# 环境配置与 PyTorch 入门

本课从配置 `brain_ai` 环境开始，练习使用 Notebook 和 Git，再用 PyTorch 定义并训练一个小网络。需要了解 Python 的变量、循环和函数。训练数据由代码生成，使用 CPU 即可完成。

## 1. 创建 Python 环境

在终端创建环境：

```bash
conda create -n brain_ai python=3.12
```

创建完成后激活环境：

```bash
conda activate brain_ai
```

## 2. 安装 Jupyter 与 PyTorch

在已激活的环境中执行：

<!-- course-dependencies -->
```bash
python -m pip install notebook ipykernel ipython numpy matplotlib torch
```
<!-- /course-dependencies -->

`notebook` 提供 Jupyter Notebook，`ipykernel` 负责在所选 Python 环境中执行单元格。后续使用服务器 GPU 时，可按 [PyTorch 安装页](https://pytorch.org/get-started/locally/)选择适合服务器的安装命令。

检查导入是否成功：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

输出的第一行是 PyTorch 版本，第二行表示是否能使用 CUDA。

## 3. 创建并运行一个 Notebook

用 VS Code 打开解压后的课程目录，安装 Microsoft 提供的 Python 和 Jupyter 扩展。通过命令面板选择 `Jupyter: Create New Jupyter Notebook`，将文件保存为 `hello.ipynb`，与本课 README 放在同一目录。

内核是执行 Notebook 代码的 Python 进程。点击右上角的内核选择器，选择 `brain_ai`，让 Notebook 使用刚才安装依赖的环境。

在第一个代码单元格输入：

```python
import sys
import torch

print(sys.executable)
print(torch.tensor([1.0, 2.0, 3.0]) * 2)
```

按 `Shift+Enter` 执行。解释器路径中应包含 `brain_ai`，张量结果应为 `[2., 4., 6.]`。再添加一个 Markdown 单元格，写下这个运算的含义，然后保存。

运行过的变量会留在内核中，修改代码不会自动更新已有输出。练习结束后，重启内核并从头运行所有单元格，确认程序能按顺序执行。

操作说明见 [VS Code 官方文档](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)和[菜鸟教程的图文步骤](https://www.runoob.com/jupyter-notebook/jupyter-notebook-vscode.html)。也可以在本机课程目录运行 `python -m notebook`，通过终端给出的地址打开浏览器界面，详见 [Jupyter 使用说明](https://jupyter.org/install)。

## 4. 在服务器上克隆一个远程仓库

在 VS Code 中安装 Remote - SSH 扩展，并连接服务器。

在远程终端中运行：

```bash
hostname
pwd
git --version
mkdir -p ~/brain_ai_practice
cd ~/brain_ai_practice
git clone https://github.com/octocat/Hello-World.git
cd Hello-World
ls
git status
git remote -v
```

这些命令在服务器上执行。`hostname` 和 `pwd` 显示机器名与当前目录；`git clone` 把仓库下载到服务器的 `Hello-World/` 目录。进入该目录后，`git status` 查看文件修改状态，`git remote -v` 显示远程仓库地址。

完成这个练习后，也可以在服务器运行课程 Notebook：上传并解压课程包，在服务器配置 `brain_ai` 环境，再打开 Notebook 并选择服务器上的内核。

阅读 [GitHub Git 入门](https://github.com/git-guides)和 [git clone 用法](https://github.com/git-guides/git-clone)。补充阅读：[Git 使用文章](https://zhuanlan.zhihu.com/p/369486197)。

## 5. 开始使用 PyTorch

打开 [intro.ipynb](intro.ipynb)，从顶部按顺序运行。先练习张量运算和自动求导，再训练一个网络拟合曲线，比较训练前后的预测。

图表显示在 Notebook 中。最后将模型参数保存到课程根目录的 `results/intro/model.pt`，重新加载后检查预测是否一致。再次运行保存单元格会更新这个文件。

继续阅读 [PyTorch 入门](https://docs.pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html)和[定义神经网络](https://docs.pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html)。后者使用图像网络，可以对照查看 `__init__` 与 `forward` 的分工。

## 参考资料

- 安装 jupyter，创建并运行一个 notebook：
  - <https://www.runoob.com/jupyter-notebook/jupyter-notebook-vscode.html>
- 在服务器中 git clone 一个远程仓库：
  - <https://github.com/git-guides>
  - <https://zhuanlan.zhihu.com/p/369486197>
- 学习使用 pytorch：
  - <https://docs.pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html>
- pytorch 创建一个神经网络模型
  - <https://docs.pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html>
