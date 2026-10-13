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

TOPICS = (
    "intro",
    "mlp",
    "amari_hopfield_model",
    "cnn",
    "continual_mnist",
    "meta_learning",
    "atari",
)
DATA_TOPICS = ("mlp", "cnn", "continual_mnist", "meta_learning")


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
@pytest.mark.parametrize("topic", DATA_TOPICS)
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
@pytest.mark.parametrize("topic", DATA_TOPICS)
def test_course_network_switch_uses_separate_cache(topic, offline_data, monkeypatch):
    path = notebook_path(topic)
    monkeypatch.chdir(path.parent)
    namespace = run_notebook_smoke(path, "cpu", use_network_data=True)
    assert_lesson_smoke(topic, namespace, offline_data, use_network_data=True)


@pytest.mark.integration
def test_hopfield_notebook_recovers_patterns_and_lowers_energy(monkeypatch):
    topic = "amari_hopfield_model"
    path = notebook_path(topic)
    monkeypatch.chdir(path.parent)
    namespace = run_notebook_smoke(path, "cpu")

    assert (namespace["recovered_flip"] == namespace["image2"]).all()
    assert (namespace["recovered_mask"] == namespace["image1"]).all()
    assert namespace["closest_pattern_index"] != namespace["TARGET_PATTERN_INDEX"]
    assert namespace["multi_pattern_is_stable"]
    assert not namespace["multi_pattern_is_stored"]
    assert (namespace["comparison_async_states"][-1] == namespace["comparison_target"]).all()
    assert namespace["sync_two_cycle"]
    for name in ("energy_flip", "energy_mask", "energy_ambiguous", "multi_pattern_energy"):
        assert (namespace[name][1:] - namespace[name][:-1] <= 1e-5).all()
    assert (
        namespace["comparison_async_energy"][1:] <= namespace["comparison_async_energy"][:-1]
    ).all()


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
