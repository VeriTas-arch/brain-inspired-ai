"""Case records and notebook assets remain usable without local models."""

import hashlib
import json
import math
import subprocess

import pytest

from biai.atari.scripts.run_experiments import build_teaching_jobs
from biai.atari.scripts.tutorial_examples import REFERENCE_RESULTS, TEACHING_ASSETS
from biai.atari.training.results import (
    parameter_digest,
    prepare_case,
    publish_outputs,
    result_directory,
    run_metadata,
)


def test_run_metadata_does_not_invoke_git(monkeypatch):
    def reject_subprocess(*args, **kwargs):
        pytest.fail("Runtime metadata must not invoke external commands")

    monkeypatch.setattr(subprocess, "run", reject_subprocess)
    metadata = run_metadata()
    assert "created_at" in metadata
    assert {"python", "torch", "gymnasium", "ale-py"} == metadata["versions"].keys()
    assert "commit" not in metadata and "dirty" not in metadata


@pytest.mark.parametrize("seed", (0, 17))
def test_training_and_evaluation_share_explicit_case_directories(seed):
    jobs = build_teaching_jobs(phase="train", seed=seed)
    training, evaluation = jobs[:12], jobs[12:]
    assert len({job.case_id for job in training}) == 12
    for train, evaluate in zip(training, evaluation, strict=True):
        assert train.case_id == evaluate.case_id
        assert train.arguments[train.arguments.index("--output-dir") + 1] == train.case_id
        assert (
            evaluate.arguments[evaluate.arguments.index("--model") + 1]
            == f"{train.case_id}/checkpoints/final.pt"
        )
        assert (
            evaluate.arguments[evaluate.arguments.index("--json-out") + 1]
            == f"{train.case_id}/evaluation/evaluation.json"
        )
        assert evaluate.depends_on == (train.log_name,)


def test_case_configuration_cannot_be_overwritten(tmp_path):
    config = dict(protocol="single", algorithm="ppo", games=["Pong-v5"], seed=0)
    prepare_case(tmp_path, **config)
    before = (tmp_path / "config.json").read_bytes()
    with pytest.raises(FileExistsError):
        prepare_case(tmp_path, **{**config, "seed": 17})
    assert (tmp_path / "config.json").read_bytes() == before


def test_default_output_uses_project_results_independently_of_cwd(monkeypatch, tmp_path):
    from biai.atari.training import results

    monkeypatch.setattr(results, "RESULTS_DIR", tmp_path / "project/results")
    monkeypatch.chdir(tmp_path)
    directory = prepare_case(None, protocol="single", algorithm="ppo", games=["Pong-v5"], seed=0)
    assert directory.resolve() == tmp_path / "project/results/single-ppo-pong"
    assert not (tmp_path / "results").exists()
    assert json.loads((directory / "run.json").read_text())["case"] == "single-ppo-pong"
    assert not (directory.parent / "run.json").exists()


def test_force_replaces_results_only_after_success(tmp_path):
    directory = tmp_path / "case"
    directory.mkdir()
    (directory / "old.pt").write_bytes(b"previous model")
    with pytest.raises(FileExistsError, match="--force"):
        with result_directory(directory):
            pytest.fail("Existing output must be rejected before work begins")
    with result_directory(directory, force=True) as staging:
        (staging / "new.pt").write_bytes(b"new model")
        assert (directory / "old.pt").read_bytes() == b"previous model"
    assert not (directory / "old.pt").exists()
    assert (directory / "new.pt").read_bytes() == b"new model"
    assert not list(tmp_path.glob(".pending-*"))


def test_failed_force_keeps_previous_results_and_failed_work(tmp_path):
    directory = tmp_path / "case"
    directory.mkdir()
    (directory / "model.pt").write_bytes(b"previous model")
    with pytest.raises(RuntimeError, match="training failed"):
        with result_directory(directory, force=True) as staging:
            (staging / "partial.json").write_text("{}")
            raise RuntimeError("training failed")
    assert (directory / "model.pt").read_bytes() == b"previous model"
    assert (staging / "partial.json").is_file()


def test_failed_publish_restores_previous_directory(monkeypatch, tmp_path):
    from pathlib import Path

    directory, staging = tmp_path / "case", tmp_path / "staging"
    for path, value in ((directory, "old"), (staging, "new")):
        path.mkdir()
        (path / "model.pt").write_text(value)
    rename = Path.rename

    def fail_new_move(path, target):
        if path == staging:
            raise OSError("move failed")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_new_move)
    with pytest.raises(OSError, match="move failed"):
        publish_outputs({staging: directory}, force=True)
    assert (directory / "model.pt").read_text() == "old"
    assert (staging / "model.pt").read_text() == "new"


def test_backup_cleanup_failure_does_not_report_successful_publish_as_unpublished(
    monkeypatch, tmp_path, capsys
):
    from biai.atari.training import results

    directory = tmp_path / "case"
    directory.mkdir()
    (directory / "model.pt").write_text("old")

    def fail_cleanup(path):
        raise PermissionError(f"Cannot remove backup: {path}")

    monkeypatch.setattr(results.shutil, "rmtree", fail_cleanup)
    with pytest.raises(PermissionError, match="Cannot remove backup"):
        with result_directory(directory, force=True) as staging:
            (staging / "model.pt").write_text("new")
    assert (directory / "model.pt").read_text() == "new"
    backup = next(tmp_path.glob(".pending-*-previous"))
    assert (backup / "model.pt").read_text() == "old"
    assert "Unpublished" not in capsys.readouterr().err


def test_parameter_digest_detects_changes():
    import torch

    module = torch.nn.Linear(3, 2)
    before = parameter_digest(module)
    with torch.no_grad():
        module.weight[0, 0] += 1
    assert parameter_digest(module) != before


def test_readers_ignore_pending_and_non_teaching_results(tmp_path):
    from biai.atari.scripts.tutorial_examples import load_teaching_results

    case = tmp_path / "single-ppo-pong"
    (case / "evaluation").mkdir(parents=True)
    for name, data in {
        "config.json": {"protocol": "single"},
        "run.json": {"commit": "source"},
        "training_summary.json": {"checkpoint_sha256": "model"},
        "evaluation/evaluation.json": {},
    }.items():
        (case / name).write_text(json.dumps(data))
    for name in (".pending-case", "smoke", "performance"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "config.json").write_text("incomplete output")
    result = load_teaching_results(tmp_path)
    assert [model["name"] for model in result["models"]] == [case.name]


def test_published_figures_and_media_match_the_current_cases():
    figures_dir = TEACHING_ASSETS / "figures"
    figures = json.loads((figures_dir / "figures.json").read_text())
    for name, digest in figures["inputs"].items():
        assert hashlib.sha256((TEACHING_ASSETS / name).read_bytes()).hexdigest() == digest
    for name, digest in figures["figures"].items():
        assert hashlib.sha256((figures_dir / name).read_bytes()).hexdigest() == digest
    media = json.loads((TEACHING_ASSETS / "videos/media.json").read_text())
    assert len(media["records"]) == 6
    for record in media["records"]:
        assert (
            hashlib.sha256((TEACHING_ASSETS / record["gif"]).read_bytes()).hexdigest()
            == record["gif_sha256"]
        )
    reference = json.loads(REFERENCE_RESULTS.read_text())
    assert len(reference["configurations"]) == 12
    for case, config in reference["configurations"].items():
        assert config["seed"] == 0
        for group in ("runs", "evaluation_runs"):
            assert "jobs" not in reference[group][case]


def test_committed_results_cover_every_case_and_preserve_score_evidence():
    from biai.atari.scripts.tutorial_examples import load_teaching_results, teaching_results

    snapshot = load_teaching_results()
    assert len(snapshot["models"]) == 12
    for record in snapshot["models"]:
        data = record["metrics"]
        games = data["games"] if "games" in data else {data["game"]: data}
        for score in games.values():
            assert len(score["rewards"]) == 10
            assert math.isclose(sum(score["rewards"]) / 10, score["avg_reward"])
    for data in snapshot["continual"].values():
        for row in data["stage_episode_rewards"]:
            for game, rewards in row["rewards"].items():
                assert len(rewards) == 10
                score = data["score_matrix"][row["stage"] - 1]["scores"][game]
                assert math.isclose(sum(rewards) / 10, score)
    for case in {"single", "multitask", "finetune", "ewc", "gpm"}:
        table = teaching_results(case)
        assert "PPO" in table and "DQN" in table


@pytest.mark.parametrize("method", ("ewc", "gpm"))
def test_teaching_table_renders_available_stage_scores(monkeypatch, method):
    from biai.atari.scripts import tutorial_examples

    snapshot = {
        "continual": {
            "example": {
                "method": method,
                "algorithm": "ppo",
                "games": ["pong"],
                "score_matrix": [{"scores": {"pong": value}} for value in (1, None, -2.5)],
            }
        }
    }
    monkeypatch.setattr(tutorial_examples, "load_teaching_results", lambda *args: snapshot)
    assert tutorial_examples.teaching_results(method).splitlines()[2:] == [
        "| PPO | pong | 1 → -2.5 |"
    ]


def test_multiple_seeds_have_disjoint_case_paths():
    jobs = [job for seed in (0, 17) for job in build_teaching_jobs(phase="train", seed=seed)[:12]]
    assert len({job.case_id for job in jobs}) == 24
    assert len({job.arguments[job.arguments.index("--output-dir") + 1] for job in jobs}) == 24


def test_reference_figures_work_without_local_results(tmp_path, monkeypatch):
    from biai.atari.scripts.build_teaching_assets import build_figures
    from biai.atari.scripts.tutorial_examples import load_teaching_results, teaching_results

    reference = tmp_path / "reference_results.json"
    reference.write_bytes(REFERENCE_RESULTS.read_bytes())
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "results").exists()
    assert len(load_teaching_results(reference)["models"]) == 12
    assert "PPO" in teaching_results("single", reference)
    output = build_figures(reference)
    assert len(list(output.glob("*.png"))) == 3


@pytest.fixture
def export_source(tmp_path):
    import shutil

    data = json.loads(REFERENCE_RESULTS.read_text())
    root = tmp_path / "local"
    for model in data["models"]:
        name = model["name"]
        case = root / name
        (case / "evaluation").mkdir(parents=True)
        values = {
            "config.json": data["configurations"][name],
            "run.json": {**data["runs"][name], "jobs": [{"pid": 999}]},
            "training_summary.json": {
                **data["training"][name],
                **data["continual"].get(name, {}),
                "checkpoint_sha256": model["sha256"],
            },
            "evaluation/evaluation.json": model["metrics"],
            "evaluation/run.json": data["evaluation_runs"][name],
        }
        key = f"{name}/learning.json"
        if key in data["learning_curves"]:
            values["learning.json"] = data["learning_curves"][key]
        for filename, value in values.items():
            (case / filename).write_text(json.dumps(value))
    media = json.loads((TEACHING_ASSETS / "videos/media.json").read_text())
    for record in media["records"]:
        case = record["checkpoint"].split("/")[0]
        source = TEACHING_ASSETS / record["gif"]
        filename = source.name.removeprefix(case + "-")
        record["gif"] = f"{case}/evaluation/videos/{filename}"
        target = root / record["gif"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (root / "figures").mkdir()
    (root / "figures/media.json").write_text(json.dumps(media))
    return root


def test_explicit_export_preserves_reference_data(export_source, tmp_path):
    from biai.atari.scripts.build_teaching_assets import export_reference

    output = export_reference(export_source, tmp_path / "assets")
    assert json.loads((output / "reference_results.json").read_text()) == json.loads(
        REFERENCE_RESULTS.read_text()
    )
    assert len(list((output / "videos").glob("*.gif"))) == 6
    assert len(list((output / "figures").glob("*.png"))) == 3


@pytest.mark.parametrize("corruption", ("gif", "evaluation"))
def test_failed_export_preserves_published_assets(export_source, tmp_path, corruption):
    from biai.atari.scripts.build_teaching_assets import export_reference

    output = tmp_path / "assets"
    output.mkdir()
    (output / "reference_results.json").write_text("previous reference")
    if corruption == "gif":
        next(export_source.glob("*/evaluation/videos/*.gif")).write_bytes(b"changed")
    else:
        path = export_source / "single-ppo-pong/evaluation/run.json"
        data = json.loads(path.read_text())
        data["checkpoint_sha256"] = "changed"
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="changed|mismatch"):
        export_reference(export_source, output)
    assert (output / "reference_results.json").read_text() == "previous reference"


@pytest.mark.parametrize(
    "destination", ("assets", "assets/figures", "biai", "tests", ".git", ".", "..", "alias")
)
def test_training_rejects_protected_output_paths(monkeypatch, tmp_path, destination):
    from biai.atari.training import results

    project = tmp_path / "project"
    (project / "assets").mkdir(parents=True)
    (project / "alias").symlink_to(project / "assets", target_is_directory=True)
    monkeypatch.setattr(results, "PROJECT_ROOT", project)
    monkeypatch.setattr(results, "ASSETS_DIR", project / "assets")
    with pytest.raises(ValueError, match="Protected output"):
        results.prepare_case(
            project / destination, protocol="single", algorithm="dqn", games=["Pong-v5"], seed=0
        )
    assert not list(tmp_path.rglob("config.json"))


@pytest.mark.parametrize(
    "module,extra",
    (
        ("train_single", ["--steps", "0"]),
        ("train_multitask", ["--steps", "0"]),
        ("train_continual", ["--steps-per-game", "0"]),
        ("evaluate", ["--model", "missing.pt", "--game", "Pong-v5"]),
    ),
)
def test_cli_rejects_protected_output_before_staging(monkeypatch, tmp_path, module, extra):
    import runpy
    import sys

    from biai.atari.training import results
    from biai.paths import PROJECT_ROOT

    project = tmp_path / "project"
    assets = project / "assets"
    assets.mkdir(parents=True)
    marker = assets / "reference_results.json"
    marker.write_text("reference")
    monkeypatch.setattr(results, "PROJECT_ROOT", project)
    monkeypatch.setattr(results, "ASSETS_DIR", assets)
    option = (
        ["--json-out", str(assets / "evaluation.json")]
        if module == "evaluate"
        else ["--output-dir", str(assets)]
    )
    monkeypatch.setattr(sys, "argv", [module, *option, "--force", *extra])
    with pytest.raises(ValueError, match="Protected output"):
        runpy.run_path(
            str(PROJECT_ROOT / "biai/atari/scripts" / f"{module}.py"), run_name="__main__"
        )
    assert marker.read_text() == "reference"
    assert not list(project.glob(".pending-*"))
