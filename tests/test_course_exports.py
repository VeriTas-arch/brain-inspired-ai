"""Exercise standalone lessons without installing the exported biai package."""

import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import nbformat
import pytest

from scripts import export_course as exporter

ROOT = Path(__file__).resolve().parents[1]
CATALOG = tomllib.loads((ROOT / "courses.toml").read_text())


@pytest.fixture
def fake_dataset_packaging(monkeypatch):
    """Keep export contract tests small while preserving per-lesson dataset selection."""

    def package(dataset_names, catalog, data_root):
        if not dataset_names:
            return {}
        manifest = {"datasets": [{"id": name} for name in dataset_names]}
        files = {Path("data") / f"{name}.fixture": f"{name}\n".encode() for name in dataset_names}
        files[Path("data/DATASETS.json")] = json.dumps(manifest).encode()
        return files

    monkeypatch.setattr(exporter, "package_datasets", package)


@pytest.mark.integration
@pytest.mark.parametrize("topic", CATALOG["lessons"])
def test_exported_lesson_runs_independently(topic, tmp_path, fake_dataset_packaging):
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    archive_path = exporter.export_course(topic, tmp_path)
    assert archive_path.parent == tmp_path / topic
    assert archive_path.name == f"biai-{topic}-{version}.zip"
    lesson_root = archive_path.with_suffix("")
    with ZipFile(archive_path) as archive:
        assert json.loads(archive.comment)["lesson"] == topic
        assert json.loads(archive.comment)["version"] == version
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
    for excluded in ("results", ".git", "tests", ".envrc", "courses.toml"):
        assert not (lesson_root / excluded).exists()
    expected_datasets = CATALOG["lessons"][topic]["datasets"]
    assert (lesson_root / "data").exists() == bool(expected_datasets)
    if expected_datasets:
        manifest = json.loads((lesson_root / "data/DATASETS.json").read_text())
        assert [entry["id"] for entry in manifest["datasets"]] == expected_datasets
        assert not list((lesson_root / "data").rglob("*mini-imagenet*"))
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
    assert metadata["version"] == version
    for requirement in metadata["dependencies"]:
        assert f'"{requirement}"' in (lesson_root / "README.md").read_text()
    assert not any(item.startswith(("pytest", "ruff")) for item in metadata["dependencies"])

    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["MPLBACKEND"] = "Agg"
    # A new interpreter must resolve biai from the ZIP even when a developer install exists.
    script = r"""
import json
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
        sys.path.append(sys.argv[2])
        from notebook_helpers import prepare_offline_data, run_notebook_smoke, assert_lesson_smoke
        data_dir = prepare_offline_data(patch, root / "data")
        namespace = run_notebook_smoke(root / f"{topic}.ipynb", "cpu")
        assert_lesson_smoke(topic, namespace, data_dir)
print("EXPORTED_LESSON_OK", topic)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, topic, str(ROOT / "tests")],
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


def test_export_keeps_versions_together_without_changing_previous_output(tmp_path, monkeypatch):
    real_read = Path.read_text
    project_version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    version = "0.1.0"

    def read_text(path, *args, **kwargs):
        text = real_read(path, *args, **kwargs)
        if path == ROOT / "pyproject.toml":
            text = text.replace(f'version = "{project_version}"', f'version = "{version}"', 1)
        return text

    monkeypatch.setattr(Path, "read_text", read_text)
    first = exporter.export_course("intro", tmp_path)
    original = first.read_bytes()
    version = "0.1.1"
    second = exporter.export_course("intro", tmp_path)
    assert first.parent == second.parent == tmp_path / "intro"
    assert first.name == "biai-intro-0.1.0.zip"
    assert second.name == "biai-intro-0.1.1.zip"
    assert first.read_bytes() == original
    assert first.with_suffix("").is_dir() and second.with_suffix("").is_dir()
    for path, expected in ((first, "0.1.0"), (second, "0.1.1")):
        with ZipFile(path) as archive:
            assert json.loads(archive.comment)["version"] == expected
            metadata = tomllib.loads(archive.read(f"{path.stem}/pyproject.toml").decode())
            assert metadata["project"]["version"] == expected


@pytest.mark.parametrize("option,value", (("--version", "0.1.1"), ("--number", "0")))
def test_export_cli_rejects_release_overrides(tmp_path, monkeypatch, capsys, option, value):
    monkeypatch.setattr(
        sys, "argv", ["export_course.py", "intro", "--output-dir", str(tmp_path), option, value]
    )
    with pytest.raises(SystemExit) as error:
        exporter.main()
    assert error.value.code == 2
    assert f"unrecognized arguments: {option} {value}" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_export_rejects_unpublished_link_before_writing(
    tmp_path, monkeypatch, fake_dataset_packaging
):
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


def _md5_bytes(content):
    return hashlib.md5(content, usedforsecurity=False).hexdigest()


def test_package_datasets_verifies_and_extracts_supported_archives(tmp_path):
    data_root = tmp_path / "sources"
    data_root.mkdir()
    gzip_path = data_root / "digits.gz"
    with gzip.open(gzip_path, "wb") as stream:
        stream.write(b"digits")

    tar_path = data_root / "images.tar.gz"
    with tarfile.open(tar_path, "w:gz") as archive:
        content = b"images"
        member = tarfile.TarInfo("cifar/images.bin")
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))

    zip_path = data_root / "characters.zip"
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("characters/a.png", b"character")

    def source(path, archive_format):
        content = path.read_bytes()
        return {
            "path": path.name,
            "md5": _md5_bytes(content),
            "format": archive_format,
        }

    catalog = {
        "datasets": {
            "fixture": {
                "title": "Fixture",
                "source_url": "https://example.test/data",
                "citation_url": "https://example.test/citation",
                "sources": [
                    source(gzip_path, "gzip"),
                    source(tar_path, "tar-xz"),
                    source(zip_path, "zip"),
                ],
            }
        }
    }
    files = exporter.package_datasets(["fixture"], catalog, data_root)
    assert files[Path("data/digits")] == b"digits"
    tar_xz = files[Path("data/images.tar.xz")]
    with tarfile.open(fileobj=io.BytesIO(tar_xz), mode="r:xz") as archive:
        extracted = archive.extractfile("cifar/images.bin")
        assert extracted is not None and extracted.read() == b"images"
    assert files[Path("data/characters.zip")] == zip_path.read_bytes()
    assert files[Path("data/characters/a.png")] == b"character"
    manifest = json.loads(files[Path("data/DATASETS.json")])
    assert manifest["datasets"][0]["id"] == "fixture"
    assert manifest["datasets"][0]["packaged_files"] == 4


def test_export_stores_nested_xz_without_recompression(tmp_path, monkeypatch):
    monkeypatch.setattr(
        exporter,
        "package_datasets",
        lambda dataset_names, catalog, data_root: {Path("data/fixture.tar.xz"): b"fixture"},
    )
    archive_path = exporter.export_course("intro", tmp_path)
    with ZipFile(archive_path) as archive:
        info = archive.getinfo(f"{archive_path.stem}/data/fixture.tar.xz")
        assert info.compress_type == ZIP_STORED


def test_package_datasets_rejects_checksum_mismatch_before_staging(tmp_path):
    source = tmp_path / "bad.gz"
    source.write_bytes(b"not-a-valid-source")
    catalog = {
        "datasets": {
            "bad": {
                "title": "Bad",
                "source_url": "https://example.test/data",
                "citation_url": "https://example.test/citation",
                "sources": [{"path": source.name, "md5": "0" * 32, "format": "gzip"}],
            }
        }
    }
    with pytest.raises(ValueError, match="checksum mismatch"):
        exporter.package_datasets(["bad"], catalog, tmp_path)


def test_package_datasets_rejects_unsafe_archive_members(tmp_path):
    source = tmp_path / "unsafe.zip"
    with ZipFile(source, "w") as archive:
        archive.writestr("../escape.txt", b"escape")
    catalog = {
        "datasets": {
            "unsafe": {
                "title": "Unsafe",
                "source_url": "https://example.test/data",
                "citation_url": "https://example.test/citation",
                "sources": [
                    {"path": source.name, "md5": _md5_bytes(source.read_bytes()), "format": "zip"}
                ],
            }
        }
    }
    with pytest.raises(ValueError, match="Unsafe dataset archive path"):
        exporter.package_datasets(["unsafe"], catalog, tmp_path)


def test_package_datasets_rejects_unsafe_tar_members(tmp_path):
    source = tmp_path / "unsafe.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        content = b"escape"
        member = tarfile.TarInfo("../escape.txt")
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    catalog = {
        "datasets": {
            "unsafe": {
                "title": "Unsafe",
                "source_url": "https://example.test/data",
                "citation_url": "https://example.test/citation",
                "sources": [
                    {
                        "path": source.name,
                        "md5": _md5_bytes(source.read_bytes()),
                        "format": "tar-xz",
                    }
                ],
            }
        }
    }
    with pytest.raises(ValueError, match="Unsafe dataset archive path"):
        exporter.package_datasets(["unsafe"], catalog, tmp_path)
