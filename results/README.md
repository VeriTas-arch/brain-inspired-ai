# 教学结果

结果直接按教学案例组织，每个案例保留一份当前结果。时间、代码版本和依赖信息放在各案例的 `run.json` 中。

```text
results/
  single-ppo-pong/
    config.json                  实际训练配置
    run.json                     训练来源和作业状态
    learning.json                过程评估的实际步数与逐回合得分
    training_summary.json        训练统计和最终模型哈希
    checkpoints/final.pt         本地模型，不进入 Git
    figures/                     训练图表
    train.log                    本地日志
    evaluation/
      evaluation.json            独立评估的逐回合得分
      run.json                   评估来源与模型哈希
      figures/
      videos/                    GIF 进入 Git，完整 MP4 留在本地
      evaluate.log
  continual-ppo-gpm/
    checkpoints/                 stage-01.pt、stage-02.pt、final.pt
    initialization.json          初始骨干和新任务头的参数摘要
    ...
  figures/                       跨案例折线图
  performance/                   本地性能报告，Git 忽略
  smoke/                         短流程验证产物，Git 忽略
```

共 12 个案例：PPO、DQN 各有 Pong 和 Breakout 单任务、联合训练、普通顺序、EWC 和 GPM。非零 seed 的案例名加上 `-seed<N>`，不同种子分别保存。

## 重跑与覆盖

默认遇到已有结果会报错，提示使用 `--force`。指定后先运行到临时目录；全部作业和产物检查通过后，才替换本次选中的案例。教学矩阵包含独立评估，会等评估成功后再替换。失败时保留旧结果及临时现场，`.pending-*` 临时目录由 Git 忽略。

```bash
python -m scripts.run_experiments train teaching --force --dry-run
python -m scripts.run_experiments evaluate teaching --force --dry-run
```

移除 `--dry-run` 才实际执行。正式长训练前先使用 `--smoke`；已有 smoke 记录同样通过 `--force` 替换。`--results-dir` 可选择另一结果根目录。

单独重评估仅替换案例的 `evaluation/`，保留训练配置与模型。底层训练脚本也支持 `--force` 和 `--output-dir`；直接训练成功后保存模型，独立评估另行运行。每个过程评估点只记录得分，单任务和联合训练保存最终模型，持续学习保存阶段末模型。最后一个阶段直接使用 `final.pt`，这些模型不承诺恢复完整环境与回放状态。

## 当前教学配置与来源

所有案例使用 seed 0、8 个 async wrapper 环境、minibatch 32、确定性训练和评估。评估使用 checkpoint 中记录的 `gymnasium_wrappers_v1` 观察协议及原始奖励，每游戏 10 个完整回合。

Pong 单任务训练 2M 步，Breakout 单任务各 500k 步，联合训练各 1.5M 步。PPO 顺序预算为 1,000,448 / 500,000 / 500,000；DQN 顺序每游戏 500k。每个 GPM 案例还使用 98,304 条边界采样。

Breakout 单任务使用固定 10 个评估点的重跑结果，其最终参数及共有评估点的逐回合得分与原训练一致。Pong 当前有 8 个过程评估点，顺序案例保留其实际记录。当前训练配置对后续单任务及顺序训练默认安排每任务 10 个过程评估点，不补造已有数据。

各案例的 `config.json`、`run.json` 和 `evaluation/run.json` 分别记录配置、训练来源和评估来源。它们不必来自同一次调度或同一提交。训练直接使用仓库代码，记录 commit 和未提交修改状态，不保存源码副本；未提交修改无法仅凭 commit 还原。[验证记录](verification.json)保存迁移核验、初始化配对及 Breakout 重跑检查。

## 最终独立评估

下表为 10 回合平均原始奖励。单任务行合并了分别训练的 Pong 和 Breakout 模型；没有 Space Invaders 单任务配置。各游戏奖励尺度不同，不跨游戏求平均。

| 算法 | 配置 | Pong | Breakout | Space Invaders |
| --- | --- | ---: | ---: | ---: |
| PPO | 单任务 | 8.9 | 4.2 | — |
| PPO | 联合 | -16.0 | 6.7 | 287.5 |
| PPO | 普通顺序 | -19.9 | 0.5 | 242.0 |
| PPO | 顺序 + EWC | -12.9 | 9.8 | 334.5 |
| PPO | 顺序 + GPM | 11.8 | 9.4 | 262.5 |
| DQN | 单任务 | 7.2 | 10.4 | — |
| DQN | 联合 | -15.5 | 4.2 | 337.0 |
| DQN | 普通顺序 | -21.0 | 3.7 | 273.5 |
| DQN | 顺序 + EWC | -21.0 | 0.1 | 426.5 |
| DQN | 顺序 + GPM | -19.3 | 1.7 | 216.5 |

## 阶段轨迹

顺序始终为 Pong → Breakout → Space Invaders。Pong 的三列分别对应完成这三个训练阶段后的得分；Breakout 两列为刚学完及完成最后任务后的得分。Space Invaders 是最后新任务，见上表。

| 算法/方法 | Pong 阶段 1 | Pong 阶段 2 | Pong 阶段 3 | Breakout 刚学完 | Breakout 最终 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PPO 普通顺序 | 12.6 | -20.8 | -19.9 | 7.6 | 0.5 |
| PPO EWC | 12.6 | -19.7 | -12.9 | 7.8 | 9.8 |
| PPO GPM | 12.6 | 9.7 | 11.8 | 6.0 | 9.4 |
| DQN 普通顺序 | -15.0 | -21.0 | -21.0 | 22.0 | 3.7 |
| DQN EWC | -15.0 | -20.7 | -21.0 | 22.0 | 0.1 |
| DQN GPM | -15.0 | -15.2 | -19.3 | 5.6 | 1.7 |

## 教学上可以说明什么

PPO 普通顺序清楚地展示了遗忘：Pong 从 12.6 降到 −19.9，Breakout 从 7.6 降到 0.5。GPM 的 Pong 最终为 11.8，较刚学完仅低 0.8；Breakout 为 9.4，Space Invaders 为 262.5。在这轮匹配初始化的运行中，它保留了旧任务表现。后续任务得分如上；由于没有 Space Invaders 单任务或未训练基线，不能据此判断该新任务已学充分。

PPO EWC 只部分缓解了 Pong 遗忘，最终为 −12.9；Breakout 和 Space Invaders 的最终得分分别为 9.8、334.5。不能仅凭最后一个游戏得分更高，就判断持续学习整体更好。

DQN 顺序案例在首个 Pong 阶段仅达到 −15.0，尚未形成强策略；因此不适合作为“保住已学会 Pong”的主要展示。它的 Breakout 轨迹更有解释力：普通顺序 22.0 → 3.7，EWC 22.0 → 0.1；当前 EWC 配置没有保住该旧任务。GPM 刚学完 Breakout 只有 5.6，最终 1.7，较小的下降同时伴随较弱的新任务学习，不能只比较遗忘量就称其更优。

联合训练仍在三个游戏间分配总预算，并非每个游戏均获得单任务预算。PPO 的实际分配为 514,048 / 506,880 / 479,072，DQN 为 502,376 / 499,560 / 498,064。因此，联合 Pong 的低分与 2M 单任务的差距不能全部归因于多任务干扰。

## 单任务学习曲线

![单任务学习曲线](figures/single_learning.png)

PPO 的 Breakout 在约 350k 步记录到 11.2，最终为 4.2；DQN 在 350k 步达到 29.7，400k 步降至 4.6，最终为 10.4。中间波动无法从原来的两个评估点中看出。Pong 中，DQN 的最终均分为 7.2，PPO 为 8.9。

这些比较仅对应 seed 0 的本次案例，不建立跨种子的算法排名。[图表与录像说明](figures/README.md)记录素材路径、截取方式和来源哈希。

重新生成当前教学图表：

```bash
python -m scripts.build_teaching_assets
```

使用 `--results-dir` 选择另一组完整教学结果；加上 `--videos` 可从本地模型和完整录像生成 GIF。重跑或重新评估后，应重新生成图表和 GIF，使展示内容与新数据对应。
