"""Exercise standalone lessons without installing the exported biai package."""

import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from zipfile import ZipFile

import nbformat
import pytest

from scripts import export_course as exporter

ROOT = Path(__file__).resolve().parents[1]
CATALOG = tomllib.loads((ROOT / "courses.toml").read_text())


@pytest.mark.parametrize("topic", CATALOG["lessons"])
def test_exported_lesson_runs_independently(topic, tmp_path):
    archive_path = exporter.export_course(topic, tmp_path)
    assert archive_path.parent == tmp_path / topic
    lesson_root = archive_path.with_suffix("")
    with ZipFile(archive_path) as archive:
        assert json.loads(archive.comment)["lesson"] == topic
        expanded_files = {
            path.relative_to(archive_path.parent).as_posix(): path.read_bytes()
            for path in lesson_root.rglob("*")
            if path.is_file()
        }
        assert set(expanded_files) == set(archive.namelist())
        for name, content in expanded_files.items():
            assert content == archive.read(name)
    notebook = lesson_root / f"{topic}.ipynb"
    nbformat.validate(nbformat.read(notebook, as_version=4))
    assert sorted(p.name for p in lesson_root.glob("*.ipynb")) == [notebook.name]
    assert sorted(p.name for p in (lesson_root / "biai").iterdir() if p.is_dir()) == [topic]
    for excluded in ("data", "results", ".git", "tests", ".envrc", "courses.toml"):
        assert not (lesson_root / excluded).exists()
    assert not list(lesson_root.rglob("*.pt"))
    assert not list(lesson_root.rglob("*.pyc"))
    assert not list(lesson_root.rglob("*.mp4"))
    assert (lesson_root / "assets").exists() == (topic == "atari")
    for path in lesson_root.rglob("*"):
        if path.suffix == ".md":
            text = path.read_text()
        elif path.suffix == ".ipynb":
            document = nbformat.read(path, as_version=4)
            text = "\n".join(c.source for c in document.cells if c.cell_type == "markdown")
            assert all(
                c.execution_count is None and not c.outputs
                for c in document.cells
                if c.cell_type == "code"
            )
        else:
            continue
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if not target.startswith(("https://", "http://", "#")):
                assert (path.parent / target.split("#")[0]).is_file(), (path, target)
    metadata = tomllib.loads((lesson_root / "pyproject.toml").read_text())["project"]
    for requirement in metadata["dependencies"]:
        assert f'"{requirement}"' in (lesson_root / "README.md").read_text()
    assert not any(item.startswith(("pytest", "ruff")) for item in metadata["dependencies"])

    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["MPLBACKEND"] = "Agg"
    # A new interpreter must resolve biai from the ZIP even when a developer install exists.
    script = r"""
import json
import runpy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat
import pytest
import torch

import biai
from biai import paths

root = Path.cwd()
topic = sys.argv[1]
assert Path(biai.__file__).resolve() == root / "biai/__init__.py"
assert paths.PROJECT_ROOT == root
torch.set_num_threads(1)
with pytest.MonkeyPatch.context() as patch:
    patch.setattr(plt, "show", lambda: plt.close("all"))
    if topic in ("intro", "atari"):
        namespace = {"__name__": "__main__"}
        for cell in nbformat.read(root / f"{topic}.ipynb", as_version=4).cells:
            if cell.cell_type == "code":
                exec(compile(cell.source, f"{topic}.ipynb", "exec"), namespace)
        if topic == "intro":
            assert namespace["final_loss"] < namespace["initial_loss"] * 0.1
            assert namespace["predictions_match"]
            assert namespace["model_path"].is_relative_to(root / "results")
            assert namespace["model_path"].is_file()
        else:
            assert not (root / "results").exists()
            from biai.atari.training.results import run_metadata
            patch.setenv("PATH", "")
            metadata = run_metadata()
            assert "commit" not in metadata and "dirty" not in metadata
            assert "torch" in metadata["versions"]
    else:
        # Reuse the owning lesson's offline datasets, budgets and semantic assertions.
        helpers = runpy.run_path(sys.argv[2])
        test = helpers["test_course_cells_train_evaluate_and_plot_offline"]
        test.__globals__["notebook_path"] = lambda _: root / f"{topic}.ipynb"
        data_dir = helpers["offline_data"].__wrapped__(patch, root)
        test(topic, "cpu", data_dir, patch)
print("EXPORTED_LESSON_OK", topic)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, topic, str(ROOT / "tests/test_course_notebooks.py")],
        cwd=lesson_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"EXPORTED_LESSON_OK {topic}" in result.stdout


@pytest.mark.parametrize("existing", ("zip", "directory", "both"))
def test_export_rejects_existing_outputs_and_keeps_source(tmp_path, existing):
    source = ROOT / "biai/intro/intro.ipynb"
    original = source.read_bytes()
    archive = exporter.export_course("intro", tmp_path)
    content = archive.read_bytes()
    expanded = archive.with_suffix("")
    if existing == "zip":
        shutil.rmtree(expanded)
    elif existing == "directory":
        archive.unlink()
    if expanded.exists():
        (expanded / "notes.txt").write_text("Review notes")
    with pytest.raises(FileExistsError):
        exporter.export_course("intro", tmp_path)
    if existing != "directory":
        assert archive.read_bytes() == content
    else:
        assert not archive.exists()
    if existing != "zip":
        assert (expanded / "notes.txt").read_text() == "Review notes"
    else:
        assert not expanded.exists()
    assert source.read_bytes() == original


@pytest.mark.parametrize("failure", ("copy", "publish"))
def test_failed_export_removes_partial_outputs(tmp_path, monkeypatch, failure):
    def fail(*args, **kwargs):
        if failure == "copy":
            (Path(args[1]) / "partial.txt").write_text("Partial copy")
        raise OSError("Export failed")

    if failure == "copy":
        monkeypatch.setattr(exporter.shutil, "copytree", fail)
    else:
        monkeypatch.setattr(exporter.os, "link", fail)
    with pytest.raises(OSError, match="Export failed"):
        exporter.export_course("intro", tmp_path)
    assert list((tmp_path / "intro").iterdir()) == []


def test_export_keeps_versions_together_without_changing_previous_output(tmp_path):
    first = exporter.export_course("intro", tmp_path, version="0.1.0", number=0)
    original = first.read_bytes()
    second = exporter.export_course("intro", tmp_path, version="0.1.1", number=1)
    assert first.parent == second.parent == tmp_path / "intro"
    assert first.name == "biai-00-intro-0.1.0.zip"
    assert second.name == "biai-01-intro-0.1.1.zip"
    assert first.read_bytes() == original
    assert first.with_suffix("").is_dir() and second.with_suffix("").is_dir()


def test_export_rejects_unpublished_link_before_writing(tmp_path, monkeypatch):
    real_read = Path.read_text

    def read_text(path, *args, **kwargs):
        text = real_read(path, *args, **kwargs)
        if path == ROOT / "biai/mlp/README.md":
            text += "\n[Later lesson](../cnn/cnn.ipynb)\n"
        return text

    monkeypatch.setattr(Path, "read_text", read_text)
    with pytest.raises(ValueError, match="Unpublished link"):
        exporter.export_course("mlp", tmp_path)
    assert not (tmp_path / "mlp").exists()
