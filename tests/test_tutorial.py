"""Teaching examples and the thin notebook interface."""

import ast
import json
import math
from pathlib import Path

from scripts.tutorial_examples import (
    run_dqn_update_demo,
    run_ewc_penalty_demo,
    run_gae_demo,
    runtime_summary,
)


def test_runtime_summary_reports_installed_frameworks() -> None:
    summary = runtime_summary()

    assert summary["torch"]
    assert summary["torchrl"]
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
    return json.loads(Path("Atari_RL_Complete_Tutorial.ipynb").read_text(encoding="utf-8"))


def test_notebook_is_a_thin_interface_to_python_functionality() -> None:
    notebook = _notebook()
    code = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )

    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.For, ast.While))
        for node in ast.walk(ast.parse(code))
    )
    assert "scripts.tutorial_examples" in code
    assert "scripts.run_experiments" in code
    assert "results/ppo_continual.png" in json.dumps(notebook)
    assert "pip install" not in code
    assert "subprocess" not in code
    assert "sys.path" not in code
    assert "def " not in code
    assert "class " not in code


def test_all_notebook_code_cells_compile_and_have_no_saved_outputs() -> None:
    notebook = _notebook()

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        compile(source, f"notebook-cell-{index}", "exec")
        assert cell["execution_count"] is None
        assert cell["outputs"] == []


def test_notebook_previews_all_teaching_cases_without_launching_jobs(monkeypatch):
    from scripts import run_experiments

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
            if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Tuple):
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
    assert len(calls) == 12
    assert {call[1] for call in calls} == {"single", "multitask", "continual", "teaching"}
    methods = {call[call.index("--method") + 1] for call in calls if "--method" in call}
    assert methods == {"finetune", "ewc", "gpm"}


def test_committed_results_cover_every_case_and_preserve_score_evidence():
    from scripts.tutorial_examples import teaching_results

    snapshot = json.loads(Path("results/teaching.json").read_text())
    cases = {"single", "multitask", "finetune", "ewc", "gpm"}
    assert {(r["case"], r["algorithm"]) for r in snapshot["records"]} == {
        (case, algorithm) for case in cases for algorithm in ("ppo", "dqn")
    }
    assert snapshot["comparison_gates"]["comparison_gates_pass"]
    for record in snapshot["records"]:
        assert len(record["source_sha256"]) == 64
        data = record["data"]
        for evaluation in data.get("evaluations", []):
            assert len(evaluation["rewards"]) == 10
            assert math.isclose(sum(evaluation["rewards"]) / 10, evaluation["mean_raw_reward"])
        for row in data.get("stage_episode_rewards", []):
            for game, rewards in row["rewards"].items():
                assert len(rewards) == 10
                score = data["score_matrix"][row["stage"] - 1]["scores"][game]
                assert math.isclose(sum(rewards) / 10, score)
    for case in cases:
        table = teaching_results(case)
        assert "PPO" in table and "DQN" in table
    assert "15.7 → -21 → 2.9" in teaching_results("ewc")
    assert "15.7 → 13.3 → 10.2" in teaching_results("gpm")
