# 教学图表与录像

折线图读取各教学案例的 `learning.json` 和 `training_summary.json`。每个游戏单独比较原始奖励，不补造未测量的评估点。运行条件和得分解释见[结果记录](../README.md)。

GIF 位于对应案例的 `evaluation/videos/` 中，每段截取完整评估录像的前 10 秒，以 12 FPS 播放。它展示一次策略行为，不代表多回合平均得分。

- [PPO 单任务 Pong](../single-ppo-pong/evaluation/videos/pong.gif)
- [DQN 单任务 Pong](../single-dqn-pong/evaluation/videos/pong.gif)
- [PPO 共同第一阶段](../continual-ppo-finetune/evaluation/videos/pong-stage1.gif)
- [PPO 普通顺序训练](../continual-ppo-finetune/evaluation/videos/pong.gif)
- [PPO EWC](../continual-ppo-ewc/evaluation/videos/pong.gif)
- [PPO GPM](../continual-ppo-gpm/evaluation/videos/pong.gif)

`media.json` 记录 GIF、原始视频和模型的相对路径与哈希；`figures.json` 记录折线图所用的数据。完整 MP4 和模型留在本地，不随 Git 分发。
