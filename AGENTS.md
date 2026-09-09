# Repository Guidelines

## Teaching material

- This repository teaches DQN, PPO, joint and sequential training, EWC, and GPM. Favor readable
  PyTorch implementations and changes that serve these examples.
- `README.md` provides a short introduction, setup instructions, and links. The notebook explains
  the methods and works through examples. `results/` holds scores, figures, and experiment details.
- Keep the notebook's main text focused on cases and training/evaluation entry points. Put
  derivations, numerical examples, and notes on differences from the papers in linked appendices.
  Exercises should remain brief suggestions for exploration.
- Write lessons in plain Chinese. Introduce terms and symbols before using them, connect each
  example to the concept it illustrates, and explain results through specific observations.
  Remove repeated introductions, slogans, and audit or changelog language from lesson prose.
- Put executable logic in Python modules and scripts. Notebook cells call tested functions and
  display artifacts; do not duplicate algorithms, training loops, result calculations, or
  installation instructions there. Training previews default to `--dry-run`.
- `build_teaching_jobs` in `scripts/run_experiments.py` defines the formal configurations. Update
  notebook previews when those configurations change. Every teaching method, including GPM,
  starts from fresh initialization without an external pretrained checkpoint.

## Environment and dependencies

- Use Python 3.12 or newer. Run project commands through direnv when available.
- Install packages with `python -m pip`. Declare all dependencies, including pytest and Ruff, in
  `pyproject.toml`; do not add `requirements.txt`, separate dev extras, `uv`, or `uv.lock`.
- Test against the PyTorch, TorchRL, and TensorDict stack declared in `pyproject.toml`. Use tested
  minimum versions; add upper bounds only when an incompatibility has been demonstrated.
- TorchRL is the adopted external RL framework. Introduce it incrementally and test each change.
  Adding another RL framework or acceleration backend requires an explicit project decision.

## Implementation and performance

- Keep single-task, sequential, and joint protocols separate. Share code when behavior and data
  semantics match; avoid abstractions that hide differences between protocols.
- PPO collection and optimization belong in `training/ppo_runtime.py`. Complete episode evaluation
  belongs in `training/evaluation.py`. Both in-training and standalone evaluation use deterministic
  actions, raw rewards, and the same episode-length limit. Reuse these in protocol scripts.
- With shared-memory vector environments, save the current observation before stepping. Under
  same-step autoreset, recover Gymnasium's `final_obs`; a truncated transition must bootstrap from
  its final observation, never the reset observation.
- Count PPO transitions across all environments and honor the requested total. Compute GAE
  separately along each environment's trajectory.
- Maintain a working synchronous/eager PPO path. Formal profiles use `--compile-ppo` and async
  environments where supported. Compile hot paths with `mode="reduce-overhead"` and use
  `fullgraph=True` wherever possible. DQN currently runs eagerly.
- Measure PPO speed with `scripts/benchmark_ppo_runtime.py`, keeping training semantics and
  minibatch size fixed. Report warmup, timed transitions, minibatch size, hardware, throughput, and
  PyTorch allocated/reserved memory. Use throughput to assess speed, episode scores to assess
  learning, and GPU utilization to help diagnose stalls.

## Runs and results

- Define experiment matrices in `scripts/run_experiments.py` instead of adding shell-script variants.
  Before a long run, smoke-test every affected configuration and its standalone evaluation with
  the intended runtime options.
- Formal teaching runs use seed 0 and deterministic training and evaluation. When comparing
  continual-learning methods, verify that network and task-head initialization, task order, and
  budgets match. Check these conditions directly as well as setting the seed.
- Give each run a new directory. Preserve earlier artifacts and frozen source snapshots. Before
  reusing a result, verify the source code, resolved configuration, and checkpoint it came from.
- Store small summaries and figures in `results/`. Large artifacts belong in the ignored
  `outputs/`, `checkpoints/`, and `logs/` directories; historical material belongs in `archive/`.
- Report old-task retention, new-task learning, and the stage-by-task score matrix. EWC penalizes
  parameter changes; assess its effect on both old and new tasks. Keep negative results and label
  historical configurations. Conclusions from a single seed apply to that run. Compare each
  game's scores separately because reward scales differ.

## Validation

Before handing off code changes, run:

```bash
direnv exec . ruff check .
direnv exec . ruff format --check .
direnv exec . pytest -q
```

For prose-only edits, check the diff, links, and notebook structure; do not launch training or GPU
tests. Report unrelated pre-existing failures without editing unrelated files to make checks pass.

The sandbox may hide NVIDIA management devices even when CUDA works. If NVML reports a warning,
first check `torch.cuda.is_available()` and run a small synchronized CUDA operation. Use elevated
`nvidia-smi` when telemetry or driver verification is needed. If both succeed, report a sandbox
limitation; do not suppress the warning in project code or change dependencies to address it.
