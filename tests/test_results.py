"""Case records and notebook assets remain usable without local models."""

import hashlib
import json
import math

import pytest

from scripts.run_experiments import build_teaching_jobs
from scripts.tutorial_examples import TEACHING_RESULTS
from training.results import parameter_digest, prepare_case, publish_output, result_directory


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


def test_default_output_has_no_batch_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    directory = prepare_case(None, protocol="single", algorithm="ppo", games=["Pong-v5"], seed=0)
    assert directory.resolve() == tmp_path / "results/single-ppo-pong"
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
        publish_output(staging, directory, force=True)
    assert (directory / "model.pt").read_text() == "old"
    assert (staging / "model.pt").read_text() == "new"


def test_parameter_digest_detects_changes():
    import torch

    module = torch.nn.Linear(3, 2)
    before = parameter_digest(module)
    with torch.no_grad():
        module.weight[0, 0] += 1
    assert parameter_digest(module) != before


def test_readers_ignore_pending_and_non_teaching_results(tmp_path):
    from scripts.tutorial_examples import load_teaching_results

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
    figures_dir = TEACHING_RESULTS / "figures"
    figures = json.loads((figures_dir / "figures.json").read_text())
    for name, digest in figures["inputs"].items():
        assert hashlib.sha256((TEACHING_RESULTS / name).read_bytes()).hexdigest() == digest
    for name, digest in figures["figures"].items():
        assert hashlib.sha256((figures_dir / name).read_bytes()).hexdigest() == digest
    media = json.loads((figures_dir / "media.json").read_text())
    assert len(media["records"]) == 6
    for record in media["records"]:
        assert (
            hashlib.sha256((TEACHING_RESULTS / record["gif"]).read_bytes()).hexdigest()
            == record["gif_sha256"]
        )
    cases = sorted(TEACHING_RESULTS.glob("*/config.json"))
    assert len(cases) == 12
    for config in cases:
        for path in (config.parent / "run.json", config.parent / "evaluation/run.json"):
            run = json.loads(path.read_text())
            assert all(job["state"] == "completed" for job in run["jobs"])


def test_committed_results_cover_every_case_and_preserve_score_evidence():
    from scripts.tutorial_examples import load_teaching_results, teaching_results

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
    from scripts import tutorial_examples

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
    monkeypatch.setattr(tutorial_examples, "load_teaching_results", lambda: snapshot)
    assert tutorial_examples.teaching_results(method).splitlines()[2:] == [
        "| PPO | pong | 1 → -2.5 |"
    ]


def test_multiple_seeds_have_disjoint_case_paths():
    jobs = [job for seed in (0, 17) for job in build_teaching_jobs(phase="train", seed=seed)[:12]]
    assert len({job.case_id for job in jobs}) == 24
    assert len({job.arguments[job.arguments.index("--output-dir") + 1] for job in jobs}) == 24
