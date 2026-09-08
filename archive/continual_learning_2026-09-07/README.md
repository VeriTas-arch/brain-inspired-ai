# 持续学习探索归档（2026-09-07）

维护范围收敛为原 EWC 与硬投影 GPM。策略功能回放、SGP、GPM read-scaling、expected-Fisher 不再作为活跃候选；这项维护决定不表示这些方法普遍无效。单任务、普通 PPO/DQN、多头和联合训练教学基线继续保留。

## 内容与恢复

`source.zip` 保存整理前的算法、训练运行时、环境、工具、脚本、测试、实验协议及仓库说明，使用原相对路径。`source_sha256.json` 逐文件记录 SHA-256；旁边的协议和结果 Markdown/JSON 是原文副本。SGP 与 GPM 共用的历史协议也原样保存在这里，活跃 GPM 说明见 [experiments/README.md](../../experiments/README.md)。

需要检查历史代码时，把 ZIP 解压到独立目录，勿覆盖当前维护目录。按对应输出目录的 `manifest.json` 和 `source_snapshot/` 识别实际运行版本；本归档是收尾时源码，不代替各次运行的冻结源码。重建运行还需要原有环境依赖和下列本地实验数据。

## 原地保留的证据

| 路径（相对仓库根目录） | 状态 |
|---|---|
| `outputs/policy_replay_v1/`、`logs/policy_replay_v1/` | 四分支历史比较；seed-61001 的 anchor 与原 EWC 结果继续供维护方案引用 |
| `outputs/subspace_v1/`、`logs/subspace_v1/` | GPM/SGP 历史比较；GPM 结果继续作为参考 |
| `outputs/subspace_read_v1/`、`logs/subspace_read_v1/` | read-scaling 已关闭 |
| `outputs/expected_fisher_v1/`、`logs/expected_fisher_v1/` | Fisher 诊断门槛未通过，未进行新任务训练 |

同系列 smoke、取消运行的部分产物也保留原地。所有 checkpoint、日志、源码快照和结果文件均未移动或改写；这些目录仍沿用仓库的 Git 忽略规则，ZIP 不包含大型数据。归档不等于外部备份。

`verification.json` 记录归档哈希、原 EWC 不变、冻结 anchor 源码兼容性，以及整理前后 GPM 基与实际 Adam 更新的数值核对。本轮只整理维护边界，未新增训练结果。
