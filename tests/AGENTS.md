# Test maintenance guidelines

Follow the [root validation and GPU-access rules](../AGENTS.md#validation) and, for Atari regressions,
[Atari implementation rules](../biai/atari/AGENTS.md). Run commands from the repository root.

## Organization

- Use `test_course_notebooks.py` for notebook structure, links and offline execution,
  `test_course_exports.py` for standalone packages, and `test_*_semantics.py` for lesson-specific
  numerical and learning-protocol checks.
- In `atari/`, separate experiment configuration (`test_experiment_jobs.py`), process scheduling
  and cleanup (`test_job_runner.py`), and publication/rollback (`test_run_publication.py`).
  Other tests follow their owning algorithm, runtime or output module.
- Reuse [notebook_helpers.py](notebook_helpers.py) through ordinary functions, not test functions or
  fixture internals. Locate notebook code by definitions rather than cell numbers. Keep small fake
  environments local when observation, reward or episode semantics differ.
- Preserve mathematical reference checks, matched initialization/budgets, transition boundaries,
  checkpoint round trips and failure recovery when reorganizing tests.

## Execution

- Use deterministic offline data and small notebook budgets. Follow the
  [root export checks](../AGENTS.md#lesson-exports) for independent package execution.
- Mark CUDA-dependent parameter combinations with `cuda`, keeping CPU combinations selectable.
  Use `integration` for complete notebooks, real Atari environments, subprocesses or video encoders;
  the markers may overlap. Select development subsets with `pytest -m` using either marker or
  `"not cuda and not integration"`; subset runs do not replace the root handoff checks.
- Use [conftest.py](conftest.py) to restore runtime settings and `fresh_compiler_state` for compiler
  isolation between tests, preserving reuse within a test. Release resources on failure and success.
- Scope warning allowances by test, message, category and source module; document compatibility
  details beside the code/filter. Preserve compiled/eager numerical, gradient, task-switch and
  checkpoint comparisons. For strict warning checks with verified GPU access, run:

```bash
direnv exec . pytest -q -W error
```
