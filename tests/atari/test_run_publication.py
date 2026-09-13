"""Run publication, evaluation replacement, and rollback boundaries."""

import pytest

from biai.atari.scripts.run_experiments import build_jobs, main


@pytest.mark.integration
@pytest.mark.parametrize("smoke", (True, False))
@pytest.mark.parametrize("succeeds", (True, False))
def test_force_publish_waits_for_successful_evaluation(monkeypatch, tmp_path, smoke, succeeds):
    import json

    from biai.atari.scripts import run_experiments as runner

    project = tmp_path / "project"
    (project / "biai/atari/scripts").mkdir(parents=True)
    for package in ("biai", "biai/atari", "biai/atari/scripts"):
        (project / package / "__init__.py").write_text("")
    (project / "biai/atari/scripts/train_example.py").write_text(
        "import hashlib,json\n"
        "from pathlib import Path\n"
        "Path('case/checkpoints').mkdir(parents=True)\n"
        "Path('case/videos').mkdir()\n"
        "Path('case/checkpoints/final.pt').write_text('model')\n"
        "Path('case/videos/example.mp4').write_text('video')\n"
        "Path('case/config.json').write_text('{}')\n"
        "Path('case/training_summary.json').write_text(json.dumps("
        "{'checkpoint_sha256':hashlib.sha256(b'model').hexdigest()}))\n"
    )
    evaluate = (
        "from pathlib import Path; "
        "assert Path('case/checkpoints/final.pt').read_text() == 'model'; "
        "Path('case/evaluation').mkdir(); "
        + (
            "Path('case/evaluation/evaluation.json').write_text('{}')"
            if succeeds
            else "raise SystemExit(3)"
        )
    )
    monkeypatch.setattr(runner, "PROJECT_ROOT", project)
    directory = tmp_path / "results"
    (directory / "case").mkdir(parents=True)
    (directory / "case/previous.pt").write_bytes(b"old model")
    (directory / "unselected").mkdir()
    (directory / "unselected/model.pt").write_bytes(b"unselected model")
    jobs = [
        runner.Job(
            "train",
            ("-m", "biai.atari.scripts.train_example", "--output-dir", "case"),
            "train.log",
            True,
            case_id="case",
        ),
        runner.Job(
            "evaluate", ("-c", evaluate), "evaluate.log", True, ("train.log",), case_id="case"
        ),
    ]
    args = runner._build_parser().parse_args(
        [
            "train",
            "teaching",
            "--device",
            "cpu",
            "--results-dir",
            str(directory),
            *(["--smoke"] if smoke else []),
        ]
    )
    with pytest.raises(FileExistsError, match="--force"):
        runner._dispatch_jobs(jobs, args)
    assert not list(directory.glob(".pending-*"))
    args.force = True
    if succeeds:
        runner._dispatch_jobs(jobs, args)
        record = json.loads((directory / "case/run.json").read_text())
        assert record.get("temporary_models_and_videos_removed", False) is smoke
        assert record["jobs"][0]["state"] == "completed"
        assert (directory / "case/evaluation/evaluation.json").is_file()
        assert (directory / "case/checkpoints/final.pt").exists() is not smoke
        assert (directory / "case/videos/example.mp4").exists() is not smoke
        assert not (directory / "case/previous.pt").exists()
        assert not list(directory.glob(".pending-*"))
        assert not (directory / "run.json").exists()
    else:
        with pytest.raises(RuntimeError, match="exit code 3"):
            runner._dispatch_jobs(jobs, args)
        assert (directory / "case/previous.pt").read_bytes() == b"old model"
        assert not (directory / "case/config.json").exists()
        failed = next(directory.glob(".pending-*/work"))
        assert (failed / "case/checkpoints/final.pt").read_text() == "model"
        assert (failed / ".logs/case/evaluate.log").is_file()
    assert (directory / "unselected/model.pt").read_bytes() == b"unselected model"


@pytest.mark.integration
def test_force_evaluation_keeps_the_case_model_and_training_record(monkeypatch, tmp_path):
    import json

    from biai.atari.scripts import run_experiments as runner

    project = tmp_path / "project"
    (project / "biai/atari/scripts").mkdir(parents=True)
    for package in ("biai", "biai/atari", "biai/atari/scripts"):
        (project / package / "__init__.py").write_text("")
    (project / "biai/atari/scripts/evaluate.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "assert Path(sys.argv[sys.argv.index('--model')+1]).read_text() == 'model'\n"
        "output=Path(sys.argv[sys.argv.index('--json-out')+1])\n"
        "output.parent.mkdir(parents=True)\noutput.write_text('{\"new\":true}')\n"
    )
    monkeypatch.setattr(runner, "PROJECT_ROOT", project)
    case = tmp_path / "results/single-ppo-pong"
    (case / "checkpoints").mkdir(parents=True)
    (case / "checkpoints/final.pt").write_text("model")
    (case / "run.json").write_text('{"commit":"training-source"}')
    (case / "evaluation").mkdir()
    (case / "evaluation/evaluation.json").write_text('{"old":true}')
    (case / "evaluation/stale.png").write_bytes(b"old plot")
    main(
        (
            "evaluate",
            "single",
            "--games",
            "Pong-v5",
            "--algorithms",
            "ppo",
            "--device",
            "cpu",
            "--results-dir",
            str(case.parent),
            "--force",
        )
    )
    assert (case / "checkpoints/final.pt").read_text() == "model"
    assert json.loads((case / "run.json").read_text()) == {"commit": "training-source"}
    assert json.loads((case / "evaluation/evaluation.json").read_text()) == {"new": True}
    assert not (case / "evaluation/stale.png").exists()
    assert (
        json.loads((case / "evaluation/run.json").read_text())["checkpoint"]
        == "../checkpoints/final.pt"
    )


@pytest.mark.parametrize("first_exists", (True, False))
@pytest.mark.parametrize("error_type", (OSError, KeyboardInterrupt))
def test_batch_publish_failure_restores_all_selected_cases(
    monkeypatch, tmp_path, first_exists, error_type
):
    import json
    from pathlib import Path

    from biai.atari.scripts import run_experiments as runner
    from biai.atari.training.results import file_digest

    cases = ("single-dqn-breakout", "single-dqn-pong")
    for case in cases:
        if case == cases[0] and not first_exists:
            continue
        (tmp_path / case).mkdir()
        (tmp_path / case / "old.pt").write_text(case)
    jobs = build_jobs(
        "train", "single", games=("Breakout-v5", "Pong-v5"), algorithms=("dqn",), steps=32
    )
    args = runner._build_parser().parse_args(
        ["train", "single", "--device", "cpu", "--results-dir", str(tmp_path), "--force"]
    )

    def finished_jobs(jobs, *, run_dir, **kwargs):
        rows = []
        for job in jobs:
            directory = run_dir / job.case_id
            (directory / "checkpoints").mkdir(parents=True)
            model = directory / "checkpoints/final.pt"
            model.write_text("new " + job.case_id)
            (directory / "config.json").write_text("{}")
            (directory / "training_summary.json").write_text(
                json.dumps({"checkpoint_sha256": file_digest(model)})
            )
            log = Path(".logs") / job.case_id / "train.log"
            (run_dir / log).parent.mkdir(parents=True)
            (run_dir / log).write_text("completed")
            rows.append(dict(log=str(log), state="completed"))
        (run_dir / "run.json").write_text(
            json.dumps(dict(created_at="", commit="", dirty=False, versions={}, jobs=rows))
        )

    rename = Path.rename

    def fail_second_publish(source, target):
        if source.parent.name == "work" and source.name == cases[1]:
            raise error_type("second publish failed")
        return rename(source, target)

    monkeypatch.setattr(runner, "run_jobs", finished_jobs)
    monkeypatch.setattr(Path, "rename", fail_second_publish)
    with pytest.raises(error_type, match="second publish failed"):
        runner._dispatch_jobs(jobs, args)
    work = next(tmp_path.glob(".pending-*/work"))
    for case in cases:
        if case == cases[0] and not first_exists:
            assert not (tmp_path / case).exists()
        else:
            assert (tmp_path / case / "old.pt").read_text() == case
        assert not (tmp_path / case / "checkpoints").exists()
        assert (work / case / "checkpoints/final.pt").read_text() == "new " + case


@pytest.mark.parametrize("destination", ("root", "assets", "case_symlink"))
def test_runner_rejects_protected_output_before_launch(monkeypatch, tmp_path, destination):
    from biai.atari.scripts import run_experiments as runner
    from biai.atari.training import results

    project = tmp_path / "project"
    assets = project / "assets"
    assets.mkdir(parents=True)
    monkeypatch.setattr(results, "PROJECT_ROOT", project)
    monkeypatch.setattr(results, "ASSETS_DIR", assets)
    jobs = build_jobs("train", "single", games=("Pong-v5",), algorithms=("dqn",), steps=32)
    root = project if destination == "root" else assets
    if destination == "case_symlink":
        root = project / "results"
        root.mkdir()
        (root / jobs[0].case_id).symlink_to(assets, target_is_directory=True)
    args = runner._build_parser().parse_args(
        ["train", "single", "--device", "cpu", "--results-dir", str(root), "--force"]
    )
    monkeypatch.setattr(runner, "run_jobs", lambda *a, **kw: pytest.fail("Jobs must not start"))
    with pytest.raises(ValueError, match="Protected output"):
        runner._dispatch_jobs(jobs, args)
    assert not list(project.rglob(".pending-*"))
