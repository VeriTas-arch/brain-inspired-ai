# Atari implementation guidelines

These rules supplement the [repository guidelines](../../AGENTS.md) for `biai/atari/` and related
regression tests and reference results. Code paths below are relative to the repository root.

## Teaching interface

- Keep executable logic in Python; the notebook calls tested functions and displays results.
  Do not duplicate algorithms, training loops or calculations. Training previews use `--dry-run`.
- `build_teaching_jobs` in `biai/atari/scripts/run_experiments.py` defines configurations and
  experiment combinations. Keep notebook previews in sync; do not duplicate this logic in shell scripts.

## Atari training and evaluation

- Keep PPO collection and optimization in `biai/atari/training/ppo_runtime.py`, and complete-episode
  evaluation in `biai/atari/training/evaluation.py`. In-training and standalone evaluation use
  deterministic actions, raw rewards and the same episode-length limit.
- Count transitions across all environments and honor the requested budget. Compute PPO GAE along
  each environment's trajectory. DQN batching must preserve cumulative update counts.
- With shared-memory environments, save the current observation before stepping. Recover `final_obs`
  under same-step autoreset. Truncated transitions bootstrap from the final observation, not the reset.
- Record the observation protocol in checkpoints. Native ALE has separate preprocessing; evaluation
  must match the protocol used for training.
- Preserve eager DQN and synchronous/eager PPO paths. Formal profiles use async environments and
  `--compile-ppo` / `--compile-dqn`. Use `reduce-overhead` and `fullgraph=True` where supported.
  DQN compilation covers inference, TD loss, gradient clipping and GPM projection. Retain fused Adam.
- CUDA learners capture the fixed-shape loss/backward/Adam update. Disable only nested Inductor
  graph capture, retaining tensor compilation. Warmup must restore parameters, optimizer state,
  RNG and GPM counters. Invalidate captured updates after loading optimizer state or changing task modules.

## Results and publication

- Store runs under ignored `results/<case>/`. Record configuration, seed, creation time and
  dependency versions without invoking Git.
- Reject existing output by default. For `--force`, stage selected cases and validate every job,
  artifact and dependent evaluation before replacement. Preserve old results and failed work on error.
- Bound job concurrency and assign disjoint CPU allocations.
- Save standalone evaluation in `<case>/evaluation/` with its own `run.json`; re-evaluation replaces
  only this directory, preserving training records and models.
- Retain final single/joint models and continual task-boundary models; name the last checkpoint
  `final.pt`. Evaluation points save scores, not checkpoints. Compare initialization with parameter digests.
- Limit cleanup to verified temporary artifacts from the current task and retain useful summaries.
  Preserve historical results unless their cleanup is explicitly within the requested scope.
- `python -m biai.atari.scripts.build_teaching_assets` redraws reference figures. Use
  `--results-dir results` to export complete local teaching results and `--videos` to regenerate GIFs
  from local checkpoints and MP4s. Check notebook explanations and score tables against exported data.

## Performance validation

For Atari speed measurements, use
`biai/atari/scripts/benchmark_ppo_runtime.py` or `biai/atari/scripts/benchmark_dqn_runtime.py`.
Hold training semantics and minibatch size fixed. Report warmup, timed transitions, hardware,
throughput and PyTorch allocated/reserved memory. Assess learning with episode scores;
GPU utilization helps diagnose stalls but does not measure training speed.
