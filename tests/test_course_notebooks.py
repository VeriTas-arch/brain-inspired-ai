"""Validate notebook structure and run lessons with small offline datasets."""

import json
import os
import re
import subprocess
import sys

import matplotlib.pyplot as plt
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
    "amari_hopfield_phase_plane",
    "cann_sfa",
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
def test_phase_plane_notebook_runs_five_figures_without_files(monkeypatch, tmp_path):
    topic = "amari_hopfield_phase_plane"
    path = notebook_path(topic)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    namespace = run_notebook_smoke(path, "cpu")

    figures = [
        namespace[name]
        for name in (
            "low_gain_figure",
            "sigmoid_figure",
            "tanh_figure",
            "high_gain_figure",
            "energy_figure",
        )
    ]
    assert len({id(figure) for figure in figures}) == 5
    for figure in figures[1:3]:
        assert len(figure.axes) == 2
        assert figure.axes[0].get_xlabel() == "x1"
        assert figure.axes[1].get_xscale() == "symlog"
    assert namespace["sigmoid_solution"].y[0, -1] == pytest.approx(2.0, abs=0.001)
    assert namespace["critical_tanh_solution"].y[0, -1] == pytest.approx(0.0274227, rel=1e-5)
    assert not list(tmp_path.rglob("*.png"))


@pytest.mark.integration
def test_cann_notebook_runs_without_writing_results(monkeypatch, tmp_path):
    import numpy as np

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    animations = []
    monkeypatch.setattr("IPython.display.display", lambda html: animations.append(html.data))
    namespace = run_notebook_smoke(notebook_path("cann_sfa"), "cpu")
    results = namespace["results"]
    late = results["smooth"]["ts"] > 700
    assert np.ptp(results["smooth"]["relative"][late]) < 0.01
    assert np.ptp(results["oscillatory"]["relative"][late]) > 0.5
    assert np.ptp(namespace["no_feedback"]["relative"][late]) < 0.01
    wave = results["travelling_wave"]
    assert np.ptp(wave["center"][late]) > 2 * np.pi
    assert wave["u"][-1].max() > 0.5
    assert len(animations) == 3
    assert all("data:image/png;base64" in html for html in animations)
    assert not list(tmp_path.iterdir())


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
