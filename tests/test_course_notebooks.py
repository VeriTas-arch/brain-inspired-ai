"""Validate notebook structure and run lessons with small offline datasets."""

import json
import os
import re
import subprocess
import sys

import nbformat
import pytest
import torch

from biai import paths
from tests.notebook_helpers import (
    assert_lesson_smoke,
    notebook_path,
    prepare_offline_data,
    run_notebook_smoke,
)

TOPICS = ("intro", "mlp", "cnn", "continual_mnist", "meta_learning", "atari")


@pytest.mark.parametrize("topic", TOPICS)
def test_notebook_format_code_and_local_links(topic):
    path = notebook_path(topic)
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, str(path), "exec")
            assert cell.execution_count is None
            assert cell.outputs == []
        else:
            for target in re.findall(r"\]\(([^)]+)\)", cell.source):
                if target.startswith(("https://", "http://", "#")):
                    continue
                assert (path.parent / target.split("#")[0]).exists(), target


@pytest.fixture
def offline_data(monkeypatch, tmp_path):
    return prepare_offline_data(monkeypatch, tmp_path / "data")


@pytest.mark.integration
@pytest.mark.parametrize("topic", TOPICS[1:-1])
@pytest.mark.parametrize("device", ("cpu", pytest.param("cuda", marks=pytest.mark.cuda)))
@pytest.mark.filterwarnings("error:__array__ implementation.*:DeprecationWarning")
def test_course_cells_train_evaluate_and_plot_offline(topic, device, offline_data, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    path = notebook_path(topic)
    monkeypatch.chdir(path.parent)
    namespace = run_notebook_smoke(path, device)
    assert_lesson_smoke(topic, namespace, offline_data)


@pytest.mark.integration
@pytest.mark.parametrize("topic", TOPICS[1:-1])
def test_course_network_switch_uses_separate_cache(topic, offline_data, monkeypatch):
    path = notebook_path(topic)
    monkeypatch.chdir(path.parent)
    namespace = run_notebook_smoke(path, "cpu", use_network_data=True)
    assert_lesson_smoke(topic, namespace, offline_data, use_network_data=True)


@pytest.mark.integration
def test_installed_package_and_atari_cli_work_outside_the_repository(tmp_path):
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "biai.atari.scripts.run_experiments",
            "train",
            "teaching",
            "--dry-run",
            "--results-dir",
            str(tmp_path / "results"),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.count("[24/24]") == 1
    assert "-m biai.atari.scripts.train_continual" in result.stdout
    assert "-m biai.atari.scripts.evaluate" in result.stdout
    assert not (tmp_path / "results").exists()
    metadata = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import json; from biai.paths import PROJECT_ROOT, DATA_DIR; "
                "print(json.dumps([str(PROJECT_ROOT), str(DATA_DIR)]))",
            ],
            cwd=tmp_path,
            env=environment,
            text=True,
        )
    )
    assert metadata == [str(paths.PROJECT_ROOT), str(paths.PROJECT_ROOT / "data")]
