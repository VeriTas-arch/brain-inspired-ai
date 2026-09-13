# Repository guidelines

## Scope and teaching material

This repository supports the Brain-Inspired Artificial Intelligence (BIAI) course: MLP, CNN,
MNIST continual learning with EWC and replay, Omniglot/Mini-ImageNet meta-learning, and Atari RL.
The distribution is `biai-course`; Python modules and lesson notebooks live under `biai`.

- Use `README.md` for installation, running examples and code navigation. Default to VS Code
  with the Python and Jupyter extensions. Each lesson's README must also work as the root README
  of its standalone ZIP, including its dependency installation and notebook instructions.
  Use `assets/README.md` to explain Atari reference results.
- Write teaching prose in plain Chinese and code comments/docstrings in English. Introduce terms
  before using them, explain results through specific observations, and avoid repeated caveats.
  Keep maintenance procedures, audit notes and changelogs out of lessons.
- Put derivations and numerical examples in notebook appendices. Keep exercises brief and end
  lessons with optional extensions. Assignment and lab-report requirements belong in separate
  course instructions.
- Keep the four introductory notebooks' implementations readable and inline. Extract shared code only when
  there is demonstrated reuse with the same behavior and data semantics, such as paths and
  reproducibility. Do not add a mandatory `lesson.py` or a framework around every lesson.
- Keep Atari executable logic in Python. Its notebook calls tested functions and displays results;
  do not duplicate algorithms, training loops or calculations. Training previews use `--dry-run`.
- Use `biai.paths` for data, results and assets so paths do not depend on the notebook working
  directory. Download datasets into ignored `data/`; keep datasets and `ref/` archives out of Git.
- `build_teaching_jobs` in `biai/atari/scripts/run_experiments.py` defines Atari configurations and
  experiment combinations. Keep notebook previews in sync; do not duplicate this logic in shell scripts.

## Environment

- Use Python 3.12 or newer. Run commands through direnv when available.
- Install with `python -m pip`. Declare all dependencies, including pytest and Ruff, in
  `pyproject.toml`. Do not add `requirements.txt`, dev extras, `uv` or `uv.lock`.
  Use tested minimum versions; add upper bounds only for demonstrated incompatibilities.
- RL algorithms and training loops use PyTorch directly. Adding a framework or acceleration
  backend requires a project decision.

## Reproducibility and comparisons

- Start each teaching method from fresh initialization. Formal runs use seed 0 and deterministic
  training and evaluation. For continual-learning comparisons, verify matching network and task-head
  initialization, task order and budgets; setting the seed alone is insufficient.
- Sample replay uniformly without replacement, using a seeded private generator separate from
  exploration, task selection and EWC sampling. Keep the replay store on CPU.
- Show new-task learning, old-task retention and stage-by-task scores together. Preserve negative
  results and distinguish a weak starting policy from successful retention. Compare Atari games
  separately, limit single-seed conclusions to that run, and consider EWC's effect on both old and new tasks.

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

## Local results and reference assets

- Store local runs under ignored `results/<case>/`. Record configuration, seed, creation time and
  dependency versions without invoking Git. Do not add dated subdirectories or source copies, or recreate
  `outputs/`, `archive/`, source manifests or compatibility links to old paths.
- Reject existing output by default. For `--force`, write to a temporary directory and check all
  selected jobs and artifacts before replacing the selected cases. Teaching runs must complete
  dependent evaluation first. Preserve old results and failed work on error.
- Bound job concurrency and assign disjoint CPU allocations.
- Save standalone evaluation in `<case>/evaluation/` with its own `run.json`. Re-evaluation replaces
  only this directory, preserving training records and models.
- Retain final single/joint models and continual task-boundary models; name the last checkpoint
  `final.pt`. Evaluation points save scores, not checkpoints. Compare initialization with parameter
  digests. After verification, remove obsolete smoke and historical artifacts while keeping useful summaries.
- Track `assets/reference_results.json`, selected figures and GIFs in Git. Reference displays must
  work without local runs. Training and `--force` must never modify these assets.
- `python -m biai.atari.scripts.build_teaching_assets` redraws reference figures.
  Add `--results-dir results` to export the complete local teaching results, and `--videos` to
  regenerate GIFs from local checkpoints and MP4s. After export, check notebook explanations and
  score tables against the new data.

## Validation

### Lesson exports

- Maintain lesson topics, file selection and dependency names in `courses.toml`; dependency version
  constraints belong in `pyproject.toml`. Add new topics under `biai/` with their own README and notebook.
- Export with `direnv exec . python scripts/export_course.py intro --version 0.1.0`;
  replace the topic to export another lesson. Add `--number 0` to assign a number at release time.
  Keep lesson titles and READMEs independent of numbering. Each export goes to ignored `dist/<topic>/`,
  containing `<release>.zip` and the matching expanded `<release>/` directory for inspection.
  `--output-dir` changes the parent of these topic directories. An existing ZIP or expanded directory
  is rejected; use a new release version to retain previous exports. Publish the ZIP only after the
  expanded directory is ready, and remove the new expanded directory if publication fails.
- The exporter uses the current working tree, including uncommitted edits. ZIP comments record the
  release label, source commit and dirty status. Review the working tree before a formal release.
- Student ZIPs put the lesson README and notebook at the root, alongside the selected `biai/` modules.
  Rewrite and check local links after relocation. Include only selected reference assets, never local
  datasets, results, checkpoints, caches or Git history. Do not modify source notebooks or assets on export.
- The README dependency block is replaced with constraints from `pyproject.toml` during export.
  Students install dependencies and run from the extracted root; installing `biai-course` is unnecessary.
- Check extracted packages in fresh Python processes, verify that imports resolve inside the package,
  and run the affected lesson's short execution checks. Keep release mechanics out of student lessons.

### Required checks

Before handing off code changes, run:

```bash
direnv exec . ruff check .
direnv exec . ruff format --check .
direnv exec . pytest -q
```

For prose-only changes, check the diff, links and affected notebook structure; do not run training
or GPU tests. Report unrelated failures without changing unrelated code to make checks pass.

Before a long run, smoke-test every affected configuration and its standalone evaluation with the
intended runtime options. For Atari speed measurements, use
`biai/atari/scripts/benchmark_ppo_runtime.py` or `biai/atari/scripts/benchmark_dqn_runtime.py`.
Hold training semantics and minibatch size fixed. Report warmup, timed transitions, hardware,
throughput and PyTorch allocated/reserved memory. Assess learning with episode scores;
GPU utilization helps diagnose stalls but does not measure training speed.

If NVML fails in the sandbox, check `torch.cuda.is_available()` and a small synchronized CUDA
operation. Use elevated `nvidia-smi` for device telemetry. If both work, report the sandbox
limitation instead of suppressing the warning or changing dependencies.
