# Atari RL Playground

An educational repository for Atari reinforcement learning and continual learning. It keeps
readable, hand-written PyTorch implementations so students can inspect the essential steps in
DQN, PPO, multi-head agents, and EWC. TorchRL is the only newly introduced reinforcement-learning
framework dependency and will be adopted incrementally for migration and reference testing.

Stable-Baselines3 and external acceleration frameworks such as EnvPool are not currently project
dependencies.

## Current Scope

- Single-task DQN and PPO training
- Joint and sequential training with a shared visual backbone and game-specific output heads
- Policy-only diagonal empirical EWC for PPO and squared TD-gradient importance for DQN
- Deterministic evaluation of every previously seen task after each training stage
- Complete checkpoints containing networks, task heads, optimizer state, and EWC state
- pytest regression tests and Ruff static checks

TorchRL currently has a dependency and import smoke test only. The teaching implementations have
not been replaced by a TorchRL trainer. Its TensorDict, replay buffer, collector, and objective
components can be evaluated independently later without removing the readable baseline all at
once.

## Installation

Python 3.12 or newer is required. `pyproject.toml` declares tested minimum versions without upper
bounds. Install the project and development tools with pip:

```bash
python -m pip install -e ".[dev]"
```

The current development baseline is PyTorch 2.14 with TorchRL 0.13.3 and TensorDict 0.13. It is not
compatible with the teaching server's current NVIDIA 550 driver. Upgrade that server to a driver
supported by the selected PyTorch wheel before deploying this development version; the previous
PyTorch 2.6 CUDA 12.4 stack is no longer a compatibility target.

The legacy installation command remains available:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` is only a compatibility entry point to `pyproject.toml`; it does not maintain a
second dependency list. This project does not use `uv`.

## Validation

```bash
pytest
ruff check .
python scripts/demo.py
```

`test_framework.py` remains as a compatibility entry point for older course material:

```bash
python test_framework.py
```

## Teaching Notebook

`Atari_RL_Complete_Tutorial.ipynb` is a thin teaching interface for prose, equations, short calls
into tested Python examples, experiment previews, and result display. It does not install packages
or contain independent training implementations. The executable examples live in
`scripts/tutorial_examples.py`, and full experiments remain available through the training,
evaluation, and experiment-runner scripts.

In a restricted execution sandbox, PyTorch may warn that it cannot initialize NVML even when CUDA
computation works. This is caused by unavailable NVIDIA management device nodes rather than by the
repository. Run `nvidia-smi` or GPU integration tests with elevated permissions when GPU telemetry
is required; do not suppress the warning in project code.

pytest and Ruff keep their repository-local caches under `.cache/pytest` and `.cache/ruff`.

## Environment Contract

Training and evaluation use the same visual and temporal preprocessing:

- ALE uses its built-in `frameskip=1`.
- The project applies `frame_skip=4` once through `MaxAndSkipEnv`.
- Observations are resized to 84×84, converted to grayscale, and stacked over four frames.
- Pixels remain `uint8` through environment and buffer storage and during device transfer; network
  preprocessing converts them to floating point.

Training additionally uses episodic-life termination and sign-based reward clipping. Evaluation
uses neither training-only technique, so it reports raw rewards over complete games. DQN and PPO
both use deterministic actions during evaluation.

## Training

Single-task training:

```bash
python scripts/train_single.py \
  --game Pong-v5 \
  --algorithm dqn \
  --steps 500000
```

Sequential continual learning:

```bash
python scripts/train_continual.py \
  --games Pong-v5 Breakout-v5 SpaceInvaders-v5 \
  --algorithm ppo \
  --steps-per-game 50000
```

Continual learning with EWC:

```bash
python scripts/train_continual.py \
  --games Pong-v5 Breakout-v5 SpaceInvaders-v5 \
  --algorithm ppo \
  --use-ewc \
  --ewc-lambda 0.4 \
  --steps-per-game 50000
```

Joint multi-task training:

```bash
python scripts/train_multitask.py \
  --games Pong-v5 Breakout-v5 SpaceInvaders-v5 \
  --algorithm dqn \
  --steps 150000
```

Training videos are disabled by default and can be enabled with `--save-video`. DQN starts
updating after 10,000 agent steps so the 50,000-step-per-task course configuration performs actual
optimization. Continual training writes the stage-by-task score matrix and per-task forgetting to
`outputs/continual/.../continual_evaluation.json`; configure the evaluation budget with
`--eval-episodes`. Multi-head DQN maintains a separate exploration rate for each task, so a new task
does not inherit the minimum epsilon reached by an earlier task. PPO uses `--batch-size` as its
minibatch size as well as DQN's replay-sample size.

## Evaluation

Evaluate a single-task checkpoint:

```bash
python scripts/evaluate.py \
  --mode single \
  --model checkpoints/single/Pong-v5_dqn/seed-0.pt \
  --algorithm dqn \
  --game Pong-v5 \
  --episodes 10 \
  --seed 0 \
  --json-out outputs/single/Pong-v5_dqn/seed-0/eval/metrics.json
```

Evaluate a continual-learning checkpoint:

```bash
python scripts/evaluate.py \
  --mode continual \
  --model checkpoints/continual/ppo_ewcTrue/seed-0.pt \
  --algorithm ppo \
  --games Pong-v5 Breakout-v5 SpaceInvaders-v5 \
  --ewc \
  --episodes 5 \
  --seed 0 \
  --json-out outputs/continual/ppo_ewcTrue/seed-0/eval/metrics.json
```

The visualization script reads these `metrics.json` files and the
`continual_evaluation.json` files produced during training:

```bash
python scripts/visualize_results.py --type evaluation --results-dir outputs
python scripts/visualize_results.py --type continual --results-dir outputs
```

Each game is displayed on a separate raw-reward axis. The script does not average raw rewards
across games or infer that EWC is effective from an experiment directory name.

Standalone evaluation allows up to 30,000 agent steps per episode by default, which exceeds ALE's
wrapped 108,000-frame time limit at frame skip 4. If an explicit lower `--max-steps` limit is reached
before the environment terminates or truncates, evaluation fails instead of recording a partial
episode return as a complete score.

## Experiment Matrices

The standard experiment matrices are defined in one Python entry point instead of separate
CPU/GPU and parallel/sequential shell scripts. Runs are sequential by default:

```bash
python scripts/run_experiments.py train single
python scripts/run_experiments.py train continual --device cpu
python scripts/run_experiments.py evaluate multitask
```

Use `--parallel` to launch every job in the selected matrix concurrently. In automatic or CUDA
mode, parallel jobs are assigned to visible GPUs round-robin; sequential jobs use the first visible
GPU. Use `--device cpu` to hide CUDA from child processes. Training output and all parallel-job
output are written under `logs/`.

Inspect a matrix without starting any training or evaluation:

```bash
python scripts/run_experiments.py train continual --parallel --dry-run
```

The runner also accepts `--games`, `--algorithms`, `--steps`, `--episodes`, `--max-steps`,
`--ewc-mode`, `--ewc-lambda`, and `--seed`; run it with `--help` for the complete interface. Training
and evaluation default to seed 0 and jointly seed Python, NumPy, PyTorch, and each Atari environment.
Use `--seeds 0 1 2` to expand a matrix over independent, seed-specific output, checkpoint, and log
paths.

## Interpreting EWC Results

EWC is a stability regularizer, not a task-conflict solver. It can limit movement in parameters
that were important to previous tasks and may therefore reduce forgetting. When old and new tasks
require opposing updates in the shared backbone, increasing EWC strength generally changes the
trade-off to better retention but slower new-task learning. Multi-head outputs resolve differing
action spaces but do not remove gradient conflicts in shared representations.

Continual-learning experiments should retain the stage-by-task score matrix `R[i, j]` and report at
least the following quantities separately:

- Old-task retention: the difference between task `j`'s best historical score and final score
- New-task plasticity: `R[j, j]`, measured immediately after training task `j`
- Joint trade-off: retention and plasticity for the same method, rather than only a final average

EWC runs also write `ewc_diagnostics` into `continual_evaluation.json`. It records the estimator,
whether that estimator is an empirical Fisher, the protected modules, sample count, and nonzero
coverage. PPO uses policy negative log-likelihood, so it protects policy-sensitive backbone and
actor directions and explicitly excludes the critic. DQN has no policy likelihood; its importance
weights are squared per-sample TD-MSE gradients and are therefore labeled as a surrogate rather
than a Fisher estimate. These boundaries are not evidence that value-function preservation is
unnecessary.

Raw rewards from different Atari games have different scales and should not be summed and
interpreted as a single performance measure. Training loss on random data is also not evidence of
forgetting. The next experimental phase should first freeze seeds, evaluation budgets, and the
score matrix, then compare EWC with a conflict-aware candidate such as a small episodic replay
baseline. More complex gradient projection or distillation methods should be considered only after
the baseline protocol is stable.

## Repository Layout

```text
algorithms/                DQN, PPO, multi-head agents, and EWC
environments/              Atari environments and train/evaluation preprocessing
utils/                     replay/rollout buffers and visualization utilities
scripts/train_single.py    single-task training
scripts/train_continual.py sequential continual learning
scripts/train_multitask.py joint multi-task training
scripts/evaluate.py        checkpoint evaluation
scripts/run_experiments.py standard experiment matrices and process execution
scripts/tutorial_examples.py short, testable examples used by the notebook
tests/                     correctness and compatibility regression tests
pyproject.toml             single source of dependency and tool configuration
```

## Known Limitations

- Training loops remain synchronous, single-environment teaching implementations. The first data
  path optimizations preserve `uint8` pixels through transfer and reuse preallocated rollout
  storage; environment parallelism has not started.
- The replay buffer stores both `state` and `next_state` per transition. Its layout can later be
  optimized without introducing another acceleration framework.
- Training handles Gymnasium `terminated` and `truncated` separately: DQN bootstraps time-limit
  transitions, while PPO bootstraps the final observation and then closes the GAE segment.
- PPO EWC uses a policy-only diagonal empirical Fisher, while DQN uses diagonal squared TD-gradient
  importance. Neither represents parameter correlations or guarantees positive transfer or
  resolution of task conflicts.

## References

- [DQN](https://www.nature.com/articles/nature14236)
- [PPO](https://arxiv.org/abs/1707.06347)
- [EWC](https://arxiv.org/abs/1612.00796)
- [Gymnasium](https://gymnasium.farama.org/)
- [TorchRL](https://docs.pytorch.org/rl/)

## License

MIT
