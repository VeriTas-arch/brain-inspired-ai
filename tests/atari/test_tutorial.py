"""Teaching examples and the thin notebook interface."""

import ast
import json
import math
from pathlib import Path

from biai.atari.scripts.tutorial_examples import (
    run_dqn_update_demo,
    run_ewc_penalty_demo,
    run_gae_demo,
    runtime_summary,
)


def test_runtime_summary_reports_python_torch_and_cuda() -> None:
    summary = runtime_summary()

    assert summary["torch"]
    assert summary["python"]
    assert isinstance(summary["cuda_available"], bool)
    assert isinstance(summary["cuda_usable"], bool)
    if summary["cuda_available"]:
        assert summary["cuda_usable"] or summary["cuda_error"] is not None


def test_dqn_demo_runs_the_maintained_update_path() -> None:
    result = run_dqn_update_demo(batch_size=2)

    assert result["batch_shapes"]["states"] == (2, 4, 84, 84)
    assert math.isfinite(result["loss"])


def test_gae_demo_runs_the_maintained_estimator() -> None:
    result = run_gae_demo()

    assert len(result["advantages"]) == 3
    assert result["returns"][-1] == 1.0


def test_ewc_demo_uses_a_representative_batch_and_detects_weight_movement() -> None:
    result = run_ewc_penalty_demo(batch_size=1)

    assert result["penalty_at_reference"] == 0.0
    assert result["penalty_after_move"] > 0.0


def _notebook() -> dict:
    path = Path(__file__).resolve().parents[2] / "biai/atari/atari.ipynb"
    return json.loads(path.read_text(encoding="utf-8"))


def test_notebook_is_a_thin_interface_to_python_functionality() -> None:
    notebook = _notebook()
    code = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )

    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.For, ast.While))
        for node in ast.walk(ast.parse(code))
    )
    assert "biai.atari.scripts.tutorial_examples" in code
    assert "biai.atari.scripts.run_experiments" in code
    assert "pip install" not in code
    assert "subprocess" not in code
    assert "sys.path" not in code


def test_notebook_previews_all_teaching_cases_without_launching_jobs(monkeypatch):
    from biai.atari.scripts import run_experiments

    def unexpected_execution(*args, **kwargs):
        raise AssertionError("Notebook previews must not launch training or evaluation")

    monkeypatch.setattr(run_experiments, "run_jobs", unexpected_execution)
    calls = []
    namespace = {}
    for cell in _notebook()["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for statement in tree.body:
            if (isinstance(statement, ast.ImportFrom) and statement.module == "biai.paths") or (
                isinstance(statement, ast.Assign)
                and isinstance(statement.value, (ast.Tuple, ast.BinOp))
            ):
                exec(
                    compile(ast.Module(body=[statement], type_ignores=[]), "settings", "exec"),
                    namespace,
                )
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "run_experiment_matrix":
                    arguments = eval(
                        compile(ast.Expression(node.args[0]), "preview", "eval"), namespace
                    )
                    assert "--dry-run" in arguments
                    run_experiments.main(arguments)
                    calls.append(arguments)
    assert len(calls) == 14
    assert {call[1] for call in calls} == {"single", "multitask", "continual", "teaching"}
    methods = {call[call.index("--method") + 1] for call in calls if "--method" in call}
    assert methods == {"finetune", "ewc", "gpm"}


def test_matched_budget_report_requires_complete_evaluation_and_equal_budgets(tmp_path):
    import pytest

    from biai.atari.scripts.run_experiments import build_teaching_jobs
    from biai.atari.scripts.tutorial_examples import matched_budget_results

    assert "14/14" in matched_budget_results(tmp_path)
    for job in build_teaching_jobs(phase="evaluate", seed=0, matched_budget=True):
        arguments = job.arguments
        mode = arguments[arguments.index("--mode") + 1]
        algorithm = arguments[arguments.index("--algorithm") + 1]
        case = tmp_path / job.case_id
        (case / "evaluation").mkdir(parents=True)
        config = dict(protocol=mode, method=job.case_id.rsplit("-", 1)[-1])
        metrics = dict(mode=mode, algorithm=algorithm)
        summary = dict(checkpoint_sha256="0" * 64)
        if mode == "single":
            metrics.update(game=arguments[arguments.index("--game") + 1], avg_reward=1.0)
            summary["total_steps"] = 500000
        else:
            games = ["Pong-v5", "Breakout-v5", "SpaceInvaders-v5"]
            metrics["games"] = {game: dict(avg_reward=1.0) for game in games}
            summary["task_steps"] = {game: 500000 for game in games}
        for filename, content in [
            ("config.json", config),
            ("run.json", {}),
            ("training_summary.json", summary),
            ("evaluation/evaluation.json", metrics),
        ]:
            (case / filename).write_text(json.dumps(content))
    table = matched_budget_results(tmp_path)
    assert table.count("500000") == 30
    assert "SpaceInvaders-v5" in table
    path = tmp_path / "single-ppo-pong/training_summary.json"
    bad = json.loads(path.read_text())
    bad["total_steps"] = 499999
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="Mismatched training budgets"):
        matched_budget_results(tmp_path)
