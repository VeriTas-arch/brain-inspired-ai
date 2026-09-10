# Atari 强化学习教程

用 DQN 和 PPO 训练 Pong、Breakout 和 Space Invaders，观察智能体怎样学会玩游戏，以及学习新游戏后是否会忘记旧游戏。

方法说明、训练示例和游戏动图见[教程 notebook](Atari_RL_Complete_Tutorial.ipynb)，得分与学习曲线见[实验结果](assets/README.md)。阅读前需要了解 Python、PyTorch 和梯度下降。

## 学习内容

| 案例 | 关注的问题 |
| --- | --- |
| 单任务 DQN / PPO | 动作价值和策略分别怎样从交互中学到？ |
| 联合多任务训练 | 多个游戏交替训练时，共享网络的表现怎样？ |
| 普通顺序训练 | 学完新游戏后，旧游戏的得分怎样变化？ |
| 顺序训练 + EWC | 限制重要参数的变化，能否减轻遗忘？ |
| 顺序训练 + GPM | 约束参数更新方向，能保留多少旧任务能力？ |

## 安装与阅读

使用 Python 3.12 或更新版本，在仓库目录安装依赖：

```bash
python -m pip install -e .
```

用 Jupyter 或 VS Code 打开 notebook，选择安装了这些依赖的 Python 环境。正文从单任务训练讲到持续学习，公式和数值例子在附录中。

仓库附带的参考结果可直接查看。Notebook 中的训练调用默认带有 `--dry-run`，执行单元格只会显示命令。

## 运行训练

以下命令均在仓库根目录执行。先预览全部 12 个案例：

```bash
python -m scripts.run_experiments train teaching --dry-run
```

用短训练检查各案例能否完成训练、保存模型和独立评估：

```bash
python -m scripts.run_experiments train teaching --smoke
```

检查通过后，运行完整训练：

```bash
python -m scripts.run_experiments train teaching
```

教学入口默认使用种子 `0`，DQN 和 PPO 均使用 8 个环境采样。各案例的训练预算见[实验结果](assets/README.md)；单独运行某个案例可使用 notebook 中的对应入口。

训练记录、模型和录像保存在 `results/<case>/`，短训练结果保存在 `results/smoke/`。这些文件不进入 Git。重跑已有案例时加上 `--force`；训练和评估成功后才替换旧结果。其他参数见 `python -m scripts.run_experiments --help`。

教材参考数据和展示素材保存在 `assets/`，可以用来对照自己的实验。运行训练或使用 `--force` 都不会覆盖这些参考结果。

## 阅读代码

| 目录 | 内容 |
| --- | --- |
| [algorithms/](algorithms/) | DQN、PPO、EWC 和 GPM |
| [environments/](environments/) | Atari 环境与图像预处理 |
| [training/](training/) | 经验回放、采样、网络更新和回合评估 |
| [scripts/](scripts/) | 训练、评估、绘图和性能测试入口 |
| [tests/](tests/) | 环境、算法及训练流程的测试 |

修改训练配置可从 [run_experiments.py](scripts/run_experiments.py) 的 `build_teaching_jobs` 开始；notebook 中的数值示例对应 [tutorial_examples.py](scripts/tutorial_examples.py)。
