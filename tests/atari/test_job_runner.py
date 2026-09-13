"""Process scheduling, device allocation, cleanup, and run records."""

import pytest

from biai.atari.scripts.run_experiments import _job_environment


def test_child_device_assignment_supports_cpu_and_gpu_round_robin(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert _job_environment(0, job_index=3, parallel=True)["CUDA_VISIBLE_DEVICES"] == ""
    assert _job_environment(2, job_index=3, parallel=True)["CUDA_VISIBLE_DEVICES"] == "1"
    assert _job_environment(2, job_index=3, parallel=False)["CUDA_VISIBLE_DEVICES"] == "0"


@pytest.mark.integration
def test_sequential_runner_records_completion_failure_and_pending_jobs(tmp_path):
    import json

    from biai.atari.scripts.run_experiments import Job, run_jobs

    jobs = [
        Job("ok", ("-c", "print('done')"), "ok.log", True),
        Job("fail", ("-c", "raise SystemExit(3)"), "fail.log", True),
        Job("pending", ("-c", "print('not run')"), "pending.log", True),
    ]
    with pytest.raises(RuntimeError, match="exit code 3"):
        run_jobs(jobs, device="cpu", parallel=False, log_dir=tmp_path)
    status = json.loads((tmp_path / "run.json").read_text())
    assert [job["state"] for job in status["jobs"]] == ["completed", "failed", "pending"]
    assert status["jobs"][0]["exit_code"] == 0
    assert status["jobs"][1]["exit_code"] == 3
    assert (tmp_path / "ok.log").read_text().strip() == "done"


@pytest.mark.parametrize("exit_race", (False, True))
def test_runner_cleanup_continues_after_a_signal_failure(monkeypatch, tmp_path, exit_race):
    import json
    import signal
    import subprocess
    from unittest.mock import Mock

    from biai.atari.scripts import run_experiments as runner

    original = KeyboardInterrupt("stop requested")
    failure = OSError("signal failed")
    processes = [Mock(pid=101, returncode=None), Mock(pid=102, returncode=None)]
    logs = []

    def start(*args, **kwargs):
        logs.append(kwargs["stdout"])
        return processes[len(logs) - 1]

    def wait(process, timeout=None):
        if process is processes[0] and timeout is not None and exit_race:
            raise subprocess.TimeoutExpired("fake process", timeout)
        process.returncode = -signal.SIGTERM
        return process.returncode

    def killpg(pid, sig):
        if pid == 101:
            if exit_race and sig == signal.SIGKILL:
                raise ProcessLookupError("already exited")
            if not exit_race:
                raise failure

    for process in processes:
        process.poll.return_value = None
        process.wait.side_effect = lambda timeout=None, process=process: wait(process, timeout)
    monkeypatch.setattr(runner, "run_metadata", lambda: {})
    monkeypatch.setattr(runner.os, "sched_getaffinity", lambda pid: {0, 1})
    monkeypatch.setattr(runner.os, "killpg", killpg)
    monkeypatch.setattr(runner.subprocess, "Popen", start)
    monkeypatch.setattr(runner.time, "sleep", Mock(side_effect=original))
    jobs = [runner.Job(str(i), ("-c", "pass"), f"{i}.log", True) for i in range(2)]
    with pytest.raises(KeyboardInterrupt if exit_race else OSError) as caught:
        runner.run_jobs(jobs, device="cpu", parallel=True, cpus_per_job=1, log_dir=tmp_path)
    if exit_race:
        assert caught.value is original
    else:
        assert caught.value is failure
        assert caught.value.__context__ is original
    assert processes[1].returncode is not None
    assert all(log.closed for log in logs)
    records = json.loads((tmp_path / "run.json").read_text())["jobs"]
    assert records[0]["state"] == ("cancelled" if exit_race else "cleanup_failed")
    assert records[1]["state"] == "cancelled"
    assert all(record["finished_at"] for record in records)


@pytest.mark.integration
def test_bounded_runner_enforces_dependencies_and_cpu_quotas(tmp_path):
    import json

    from biai.atari.scripts.run_experiments import Job, run_jobs

    script = "import os,time,json; print(json.dumps({'cpus':len(os.sched_getaffinity(0))})); time.sleep(0.2)"
    jobs = [
        Job("eval a", ("-c", script), "eval-a.log", True, ("train-a.log",)),
        Job("train a", ("-c", script), "train-a.log", True),
        Job("train b", ("-c", script), "train-b.log", True),
        Job("eval b", ("-c", script), "eval-b.log", True, ("train-b.log",)),
    ]
    run_jobs(jobs, device="cpu", parallel=True, max_workers=2, cpus_per_job=1, log_dir=tmp_path)
    records = json.loads((tmp_path / "run.json").read_text())["jobs"]
    assert all(record["state"] == "completed" for record in records)
    assert records[0]["started_at"] >= records[1]["finished_at"]
    assert records[3]["started_at"] >= records[2]["finished_at"]
    events = sorted(
        [(r["started_at"], 1) for r in records] + [(r["finished_at"], -1) for r in records]
    )
    active = peak = 0
    for _, change in events:
        active += change
        peak = max(peak, active)
    assert peak == 2 and active == 0
    assert all(json.loads((tmp_path / job.log_name).read_text())["cpus"] == 1 for job in jobs)


@pytest.mark.integration
def test_new_runs_use_current_source_and_keep_artifacts_separate(monkeypatch, tmp_path):
    import json

    from biai.atari.scripts import run_experiments as runner

    project = tmp_path / "project"
    (project / "biai/atari/scripts").mkdir(parents=True)
    for package in ("biai", "biai/atari", "biai/atari/scripts"):
        (project / package / "__init__.py").write_text("")
    script = project / "biai/atari/scripts/train_example.py"
    monkeypatch.setattr(runner, "PROJECT_ROOT", project)
    job = runner.Job("example", ("-m", "biai.atari.scripts.train_example"), "example.log", True)
    for value in ("first", "second"):
        script.write_text(f"from pathlib import Path\nPath('result.txt').write_text('{value}')\n")
        directory = tmp_path / value
        runner.run_jobs(
            [job],
            device="cpu",
            parallel=False,
            log_dir=directory / "custom-logs",
            run_dir=directory,
        )
        record = json.loads((directory / "run.json").read_text())
        assert record["jobs"][0]["state"] == "completed"
        assert record["jobs"][0]["log"] == "custom-logs/example.log"
        assert not (directory / "source").exists()
        assert not (directory / "source_manifest.json").exists()
    assert (tmp_path / "first/result.txt").read_text() == "first"
    assert (tmp_path / "second/result.txt").read_text() == "second"
    with pytest.raises(ValueError, match="new run directory"):
        runner.run_jobs(
            [job],
            device="cpu",
            parallel=False,
            log_dir=tmp_path / "logs",
            run_dir=tmp_path / "first",
        )


@pytest.mark.parametrize("visible", ("3,1", "GPU-first,GPU-second", "MIG-GPU-first/1/0"))
def test_child_device_assignment_preserves_parent_visible_devices(monkeypatch, visible):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    devices = visible.split(",")
    for slot in range(3):
        assert (
            _job_environment(len(devices), slot, True)["CUDA_VISIBLE_DEVICES"]
            == devices[slot % len(devices)]
        )
        assert _job_environment(len(devices), slot, False)["CUDA_VISIBLE_DEVICES"] == devices[0]
    assert _job_environment(0, 0, False)["CUDA_VISIBLE_DEVICES"] == ""
