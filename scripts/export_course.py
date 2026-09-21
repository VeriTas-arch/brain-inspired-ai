"""Export a lesson ZIP and its expanded contents into a per-lesson directory."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import lzma
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"(?<=\]\()([^\s)]+)(?=\))")
DEPENDENCIES = re.compile(
    r"<!-- course-dependencies -->.*?<!-- /course-dependencies -->", re.DOTALL
)


def _archive_path(name: str) -> Path:
    """Return a safe relative path from an archive member name."""
    path = Path(name)
    if "\\" in name or path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe dataset archive path: {name}")
    return path


def _add_file(files: dict[Path, bytes], destination: Path, content: bytes) -> None:
    """Add one packaged file while rejecting conflicting archive members."""
    if destination in files and files[destination] != content:
        raise ValueError(f"Conflicting dataset file: {destination}")
    files[destination] = content


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_datasets(
    dataset_names: list[str], catalog: dict, data_root: Path = ROOT / "data"
) -> dict[Path, bytes]:
    """Verify and stage only the datasets declared by one lesson."""
    if not dataset_names:
        return {}
    data_root = Path(data_root).resolve()
    files: dict[Path, bytes] = {}
    manifest = {"datasets": []}
    for name in dataset_names:
        dataset = catalog["datasets"][name]
        packaged_before = set(files)
        source_records = []
        for source in dataset["sources"]:
            relative = _archive_path(source["path"])
            candidate = data_root / relative
            path = candidate.resolve()
            if (
                not path.is_relative_to(data_root)
                or not candidate.is_file()
                or candidate.is_symlink()
            ):
                raise FileNotFoundError(f"Missing regular dataset source: {path}")
            actual_md5 = _md5(path)
            if actual_md5 != source["md5"]:
                raise ValueError(
                    f"Dataset checksum mismatch for {path}: {actual_md5} != {source['md5']}"
                )
            source_records.append(
                {
                    "path": relative.as_posix(),
                    "md5": actual_md5,
                    "bytes": path.stat().st_size,
                    "format": source["format"],
                }
            )
            archive_format = source["format"]
            if archive_format == "gzip":
                with gzip.open(path, "rb") as stream:
                    _add_file(files, Path("data") / relative.with_suffix(""), stream.read())
            elif archive_format == "tar-xz":
                with tarfile.open(path, "r:gz") as archive:
                    for member in archive.getmembers():
                        _archive_path(member.name)
                        if not (member.isdir() or member.isfile()):
                            raise ValueError(f"Unsupported dataset archive member: {member.name}")
                compressed = io.BytesIO()
                with (
                    gzip.open(path, "rb") as source_stream,
                    lzma.LZMAFile(
                        compressed, "wb", preset=9 | lzma.PRESET_EXTREME
                    ) as destination_stream,
                ):
                    shutil.copyfileobj(source_stream, destination_stream)
                destination = Path(
                    "data/" + relative.as_posix().removesuffix(".tar.gz") + ".tar.xz"
                )
                _add_file(files, destination, compressed.getvalue())
            elif archive_format == "zip":
                _add_file(files, Path("data") / relative, path.read_bytes())
                try:
                    with ZipFile(path) as archive:
                        for member in archive.infolist():
                            if member.is_dir():
                                continue
                            destination = (
                                Path("data") / relative.parent / _archive_path(member.filename)
                            )
                            _add_file(files, destination, archive.read(member))
                except BadZipFile as error:
                    raise ValueError(f"Invalid dataset ZIP: {path}") from error
            else:
                raise ValueError(f"Unknown dataset archive format: {archive_format}")
        packaged = set(files) - packaged_before
        manifest["datasets"].append(
            {
                "id": name,
                "title": dataset["title"],
                "source_url": dataset["source_url"],
                "citation_url": dataset["citation_url"],
                "sources": source_records,
                "packaged_files": len(packaged),
                "packaged_bytes": sum(len(files[path]) for path in packaged),
            }
        )
    files[Path("data/DATASETS.json")] = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    return files


def rewrite_links(text: str, source: Path, destination: Path, mapping: dict[Path, Path]) -> str:
    """Relocate local Markdown links and reject files absent from the export."""

    def replace(match):
        target = match.group(0)
        if target.startswith(("https://", "http://", "#", "mailto:")):
            return target
        filename, separator, fragment = target.partition("#")
        linked_file = (source.parent / filename).resolve()
        if linked_file not in mapping:
            raise ValueError(f"Unpublished link in {source}: {target}")
        relative = Path(os.path.relpath(mapping[linked_file], destination.parent)).as_posix()
        return relative + separator + fragment

    return LINK.sub(replace, text)


def export_course(topic: str, output_dir: Path, data_root: Path = ROOT / "data") -> Path:
    """Build matching ZIP and directory outputs without overwriting either one."""
    catalog = tomllib.loads((ROOT / "courses.toml").read_text())
    lesson = catalog["lessons"][topic]
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    version = project["version"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", version):
        raise ValueError("Version must contain only letters, digits, dots, underscores or hyphens")
    name = f"biai-{topic}-{version}"
    output = Path(output_dir) / topic / f"{name}.zip"
    expanded = output.with_suffix("")
    for destination in (output, expanded):
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)

    mapping = {}
    for pattern in catalog["common"]["files"] + lesson["files"]:
        matches = sorted(ROOT.glob(pattern))
        if not matches:
            raise ValueError(f"No files matched: {pattern}")
        for source in matches:
            if (
                not source.is_file()
                or source.is_symlink()
                or not source.resolve().is_relative_to(ROOT)
            ):
                raise ValueError(f"Expected a regular source file: {source}")
            mapping[source.resolve()] = source.relative_to(ROOT)
    readme = ROOT / "biai" / topic / "README.md"
    notebook = ROOT / "biai" / topic / f"{topic}.ipynb"
    mapping[readme] = Path("README.md")
    mapping[notebook] = Path(notebook.name)
    # Existing reference documentation links back to the repository's root README.
    links = {**mapping, ROOT / "README.md": Path("README.md")}
    declared = {
        re.split(r"[<>=!~\[; ]", item, maxsplit=1)[0]: item for item in project["dependencies"]
    }
    requirements = [
        declared[key] for key in catalog["common"]["dependencies"] + lesson["dependencies"]
    ]
    command = "python -m pip install " + " ".join(f'"{item}"' for item in requirements)
    dependency_block = (
        f"<!-- course-dependencies -->\n```bash\n{command}\n```\n<!-- /course-dependencies -->"
    )

    files = package_datasets(lesson["datasets"], catalog, data_root)
    for source, destination in mapping.items():
        if source.suffix == ".md":
            text = rewrite_links(source.read_text(encoding="utf-8"), source, destination, links)
            if source == readme:
                text, count = DEPENDENCIES.subn(lambda _: dependency_block, text)
                if count != 1:
                    raise ValueError(f"Expected one dependency block in {readme}")
            files[destination] = text.encode("utf-8")
        elif source.suffix == ".ipynb":
            document = json.loads(source.read_text(encoding="utf-8"))
            for cell in document["cells"]:
                text = "".join(cell["source"])
                if cell["cell_type"] == "markdown":
                    cell["source"] = rewrite_links(text, source, destination, links).splitlines(
                        True
                    )
                elif cell["cell_type"] == "code":
                    compile(text, str(source), "exec")
                    cell["outputs"] = []
                    cell["execution_count"] = None
            files[destination] = (
                json.dumps(document, ensure_ascii=False, indent=1) + "\n"
            ).encode()
        else:
            files[destination] = source.read_bytes()

    # Keep the exact dependency constraints available without installing the course itself.
    metadata = {
        "name": project["name"],
        "version": version,
        "description": lesson["title"],
        "requires-python": project["requires-python"],
        "dependencies": requirements,
    }
    files[Path("pyproject.toml")] = (
        "[project]\n"
        + "\n".join(
            f"{key} = {json.dumps(value, ensure_ascii=False)}" for key, value in metadata.items()
        )
        + "\n"
    ).encode()
    provenance = {
        "lesson": topic,
        "version": version,
        "commit": None,
        "dirty": None,
    }
    if (ROOT / ".git").exists():
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
            )
            changes = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            provenance.update(commit=commit.stdout.strip(), dirty=bool(changes.stdout))
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".course-", dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        with ZipFile(staged, "w", compression=ZIP_DEFLATED) as archive:
            for path, content in sorted(files.items()):
                compression = ZIP_STORED if path.suffix == ".xz" else ZIP_DEFLATED
                archive.writestr(f"{name}/{path.as_posix()}", content, compress_type=compression)
            archive.comment = json.dumps(provenance).encode()
        with ZipFile(staged) as archive:
            if archive.testzip() is not None:
                raise ValueError("ZIP integrity check failed")
            archive.extractall(temporary)
        # Reserve the directory exclusively; publish the ZIP after its contents are ready.
        expanded.mkdir()
        try:
            shutil.copytree(Path(temporary) / name, expanded, dirs_exist_ok=True)
            os.link(staged, output)
        except BaseException:
            shutil.rmtree(expanded)
            raise
    return output


def main():
    catalog = tomllib.loads((ROOT / "courses.toml").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lesson", choices=catalog["lessons"])
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "dist", help="Parent of the lesson directories"
    )
    args = parser.parse_args()
    print(export_course(args.lesson, args.output_dir))


if __name__ == "__main__":
    main()
