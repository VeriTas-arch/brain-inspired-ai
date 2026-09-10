# Repository guidelines

## Documentation and scope

This repository teaches DQN, PPO, joint training, sequential training, EWC and GPM.
Keep the Python implementations readable and share code only when the protocols have the same
behavior and data semantics.

- `README.md` is the student's starting point: installation, running examples and code navigation.
  The notebook explains methods and cases; `assets/README.md` explains the reference results.
- Write teaching material in plain Chinese. Introduce terms before using them and explain results
  with specific observations. Keep maintenance procedures, performance audit notes and changelogs
  out of lesson prose. Put derivations and numerical examples in notebook appendices; keep exercises brief.
- Keep executable logic in Python. Notebook cells call tested functions and display artifacts;
  do not duplicate algorithms, training loops, result calculations or installation instructions.
  Training previews use `--dry-run`.
- `build_teaching_jobs` in `scripts/run_experiments.py` defines the formal configurations and
  experiment matrix. Update notebook previews when it changes. Do not add shell scripts that duplicate it.

## Environment

- Use Python 3.12 or newer and run commands through direnv when available.
- Install with `python -m pip`. Keep all dependencies, including pytest and Ruff, in `pyproject.toml`.
  Do not add `requirements.txt`, dev extras, `uv` or `uv.lock`. Use tested minimum versions; add
  upper bounds only for demonstrated incompatibilities.
- TorchRL is the adopted RL framework. Test integrations against the declared PyTorch, TorchRL
  and TensorDict versions. Adding a framework or acceleration backend requires a project decision.

## Training and evaluation

- All teaching methods start from fresh initialization. Formal runs use seed 0 and deterministic
  training and evaluation. For continual-learning comparisons, also verify matching network and
  task-head initialization, task order and budgets; setting the seed alone is insufficient.
- Keep PPO collection and optimization in `training/ppo_runtime.py`, and complete-episode evaluation
  in `training/evaluation.py`. In-training and standalone evaluation must use deterministic actions,
  raw rewards and the same episode-length limit.
- With shared-memory environments, save the current observation before stepping. Recover `final_obs`
  under same-step autoreset; truncated transitions bootstrap from the final observation, not the reset.
- Count transitions across all environments and honor the requested budget. Compute PPO GAE along
  each environment's trajectory. DQN batching must preserve cumulative update counts.
- Replay sampling stays uniform without replacement. Use a seeded private generator separate from
  exploration, task selection and EWC sampling; keep the replay store on CPU.
- Native ALE uses a separate observation protocol. Record it in checkpoints and use matching
  preprocessing in evaluation; do not evaluate wrapper-trained weights with native ALE preprocessing.
- Keep working eager DQN and synchronous/eager PPO paths. Formal profiles use async environments
  and `--compile-ppo` / `--compile-dqn`. Use `reduce-overhead` and `fullgraph=True` where supported;
  DQN compilation covers inference, TD loss, gradient clipping and GPM projection. Retain fused Adam.
- CUDA learners capture the fixed-shape loss/backward/Adam update. Disable nested Inductor graph
  capture only; retain tensor compilation. Warmup must restore parameters, optimizer state, RNG and
  GPM counters. Invalidate captured updates after loading optimizer state or changing task modules.

## Local runs and reference results

- Store local runs under ignored `results/<case>/`, without date directories or source copies.
  Record configuration, seed, Git commit, dirty status and dependency versions. Do not recreate
  `outputs/`, `archive/`, source manifests or compatibility links to old paths.
- Reject existing output by default. `--force` replaces selected cases only: write to a temporary
  directory and check all selected jobs and artifacts before replacement. Teaching runs must finish
  dependent evaluation first. Preserve old results and failed work on error. Keep job concurrency
  bounded and CPU allocations disjoint.
- Standalone evaluation belongs in `<case>/evaluation/`, with its own `run.json`. Re-evaluation
  replaces only this directory, preserving training records and models.
- Retain final single/joint models and continual task-boundary models; use `final.pt` for the last
  stage. Evaluation points save scores, not checkpoints. Compare initialization through parameter
  digests. Remove obsolete smoke and historical artifacts after verification, keeping useful summaries.
- Track `assets/reference_results.json`, selected figures and GIFs in Git. Reference displays must
  work without local runs. Training and `--force` must never modify these assets.
- `python -m scripts.build_teaching_assets` redraws reference figures. An explicit
  `python -m scripts.build_teaching_assets --results-dir results` exports the complete teaching results.
  Add `--videos` to regenerate GIFs from local checkpoints and MP4s. After export, check notebook
  explanations and score tables against the new data.
- Report old-task retention, new-task learning and stage-by-task scores. Keep negative results;
  distinguish a weak starting policy from successful retention. Compare games separately and limit
  single-seed conclusions to that run. EWC's penalty can affect both old and new tasks.

## Validation

Before handing off code changes, run:

```bash
direnv exec . ruff check .
direnv exec . ruff format --check .
direnv exec . pytest -q
```

For prose-only changes, check the diff, links and notebook structure; do not run training or GPU tests.
Report unrelated failures without changing unrelated code to make checks pass.

Before a long run, smoke-test every affected configuration and its standalone evaluation with the
intended runtime options. Measure speed with `scripts/benchmark_ppo_runtime.py` and
`scripts/benchmark_dqn_runtime.py`, holding training semantics and minibatch size fixed. Report warmup,
timed transitions, hardware, throughput and PyTorch allocated/reserved memory. Use episode scores to
assess learning; GPU utilization helps diagnose stalls but does not measure training speed.

If NVML fails in the sandbox, check `torch.cuda.is_available()` and a small synchronized CUDA operation.
Use elevated `nvidia-smi` for device telemetry. If both work, report the sandbox limitation instead of
suppressing the warning or changing dependencies.
