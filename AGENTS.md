# Repository guidelines

The `biai-course` distribution contains BIAI course notebooks and Python modules under `biai/`.
For Atari implementation, regression or reference-result changes, follow
[biai/atari/AGENTS.md](biai/atari/AGENTS.md). Test changes also follow
[tests/AGENTS.md](tests/AGENTS.md). The validation requirements below apply repository-wide.

## Teaching material

- Keep installation, examples and code navigation in READMEs; explain Atari reference results in
  [assets/README.md](assets/README.md). Each lesson README must work as its standalone ZIP's
  entry point, including dependency installation and notebook instructions. Default to VS Code
  with Python and Jupyter extensions.
- Keep contributor procedures in [BUILDING.md](BUILDING.md) and maintenance constraints in
  scoped AGENTS files. Keep maintenance procedures, audit notes and changelogs out of lessons.
- Write teaching prose in plain Chinese and code comments/docstrings in English. Introduce terms
  before using them, explain specific observations and avoid repeated caveats.
- Put derivations and numerical examples in notebook appendices. Keep exercises brief, end lessons
  with optional extensions, and place assignment/report requirements in separate course instructions.
- Keep MLP, CNN, MNIST continual-learning and meta-learning implementations readable and inline in
  their notebooks. Extract helpers only for demonstrated reuse with matching behavior and data
  semantics, such as paths and reproducibility; do not require a `lesson.py` or per-lesson framework.

## Environment and data

- Use Python 3.12 or newer and run commands through direnv when available.
- Install with `python -m pip`; declare dependencies, including pytest and Ruff, in `pyproject.toml`.
  Use tested minimum versions and upper bounds only for demonstrated incompatibilities.
  Do not add `requirements.txt`, dev extras, `uv` or `uv.lock`.
- RL algorithms and training loops use PyTorch directly. Adding a framework or acceleration
  backend requires a project decision.
- Use `biai.paths` so data, results and assets do not depend on the notebook working directory.
  Keep downloaded datasets in ignored `data/` and local results in ignored `results/`; exclude
  datasets and `ref/` archives from Git. Do not add dated result subdirectories, source copies,
  `outputs/`, `archive/`, source manifests or compatibility links to old paths.
- Track `assets/reference_results.json`, selected figures and GIFs. Reference displays must work
  without local runs; training and `--force` must never modify these assets.

## Reproducibility and comparisons

- Start each method from fresh initialization. Formal runs use seed 0 and deterministic training
  and evaluation. Match network/task-head initialization, task order and budgets across continual
  methods; setting the seed alone is insufficient.
- Sample replay uniformly without replacement with a seeded private generator separate from
  exploration, task selection and EWC sampling. Keep the replay store on CPU.
- Show new-task learning, old-task retention and stage-by-task scores together. Preserve negative
  results and distinguish weak initial learning from successful retention. Compare Atari games
  separately, keep single-seed conclusions specific to that run, and assess EWC on old and new tasks.

## Lesson exports

- Maintain lesson topics, titles, file selection and dependency names in `courses.toml`, and version
  constraints in `pyproject.toml`. Each topic has a README and notebook under `biai/`.
  Follow [BUILDING.md](BUILDING.md) for commands and layout; keep titles and READMEs independent of numbering.
- Use `pyproject.toml`'s `[project].version` as the sole release version for the repository and all
  lessons. Keep export names, packaged project metadata and ZIP comments consistent with it.
- Reject existing release ZIPs or expanded directories; use a new version. Publish the ZIP only
  after the expanded directory is ready, and remove that new directory if publication fails.
- Export the current working tree, including uncommitted edits. Record project version, source commit
  and dirty status in the ZIP comment; review the working tree before a formal release.
- Relocate the lesson README and notebook to the ZIP root beside selected `biai/` modules, rewriting
  and checking local links. Populate README dependencies from `pyproject.toml`; extracted lessons
  must run without installing `biai-course`. Do not modify source notebooks or assets on export.
- Include only selected reference assets; exclude datasets, local results, checkpoints, caches,
  Git history and maintenance documents. Verify package-local imports and the affected lesson's
  short execution checks in fresh Python processes.

## Validation

Before handing off code changes, run from the repository root:

```bash
direnv exec . ruff check .
direnv exec . ruff format --check .
direnv exec . pytest -q
```

For prose-only changes, check the diff, links and affected notebook structure; do not run training
or GPU tests. Report unrelated failures without changing unrelated code to make checks pass.
Before a long run, smoke-test every affected configuration and its standalone evaluation using the
intended runtime options.

If CUDA or NVML is unavailable in the sandbox, compare with an elevated process using
`torch.cuda.is_available()`, a small synchronized CUDA operation and `nvidia-smi`. If these work,
report the access limitation and validate GPU changes there. Skipped GPU tests are not completed
GPU validation; do not change dependencies to hide an access limitation.
