# Repository Guidelines

## Environment and dependency policy

- Use Python 3.12 or newer.
- Run project commands through the active direnv environment when available, for example
  `direnv exec . pytest`.
- Use `python -m pip` for installation. Do not use `uv` and do not add `uv.lock`.
- Keep dependency declarations in `pyproject.toml`; `requirements.txt` is only a compatibility
  entry point.
- Specify tested minimum versions such as `tqdm>=4.70`. Do not add upper bounds unless a concrete
  incompatibility has been demonstrated.
- TorchRL is the only external reinforcement-learning framework dependency currently adopted.
  Do not add Stable-Baselines3, EnvPool, or another acceleration framework without an explicit
  project decision.
- Develop and validate against the current PyTorch 2.14, TorchRL 0.13.3, and TensorDict 0.13 stack.
  PyTorch 2.6 is not a compatibility target. The teaching server's NVIDIA 550 driver must be
  upgraded before this development version can be deployed there.

## GPU and NVML behavior

- The restricted Codex sandbox may not expose the `/dev/nvidia*` management devices required by
  NVML. In that case, PyTorch can emit `Can't initialize NVML`, and `nvidia-smi` can fail even
  while CUDA computation remains available.
- Do not treat this warning alone as a driver failure, suppress it in project code, or change
  project dependencies to work around it.
- First verify CUDA with `torch.cuda.is_available()` and a small synchronized CUDA tensor
  operation. When GPU telemetry or a definitive driver check is required, rerun `nvidia-smi` or
  the relevant test with elevated permissions outside the restricted sandbox.
- A normal elevated `nvidia-smi` result and a successful CUDA tensor operation mean the warning
  is an execution-sandbox limitation rather than a repository defect.

## Code organization

- Preserve the readable PyTorch DQN, PPO, multi-head, and EWC implementations as teaching
  baselines. TorchRL adoption should be incremental and separately testable.
- Python modules and scripts are the only executable source of truth. The teaching notebook is a
  thin presentation layer for prose, formulas, calls into tested Python functions, and artifact
  display; do not duplicate algorithms, training loops, installation logic, or result computation
  in notebook cells.
- Keep single-task, sequential continual-learning, and joint multi-task training protocols
  separate unless a genuinely shared maintained contract has been identified.
- Extract repeated evaluation, serialization, and plotting behavior when semantics are identical;
  avoid generic wrappers that only hide protocol differences.
- Use `scripts/run_experiments.py` for experiment matrices. Add Python options or job definitions
  there instead of introducing CPU/GPU or parallel/sequential shell-script variants.
- Evaluation must use deterministic actions and raw evaluation rewards. Do not average raw scores
  across Atari games with different reward scales.
- EWC is a stability regularizer, not a task-conflict solver. Report retention, new-task
  plasticity, and the stage-by-task score matrix separately.

## Validation

Run the following before handing off code changes:

```bash
direnv exec . ruff check .
direnv exec . ruff format --check .
direnv exec . pytest -q
```

If pytest passes in the restricted sandbox with only an NVML warning, report the warning as an
environment limitation. Use elevated execution only when the task requires GPU telemetry or a
warning-free GPU integration check.
