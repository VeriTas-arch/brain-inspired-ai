"""Export one self-contained lesson ZIP from the current source tree."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import tomllib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"(?<=\]\()([^\s)]+)(?=\))")
DEPENDENCIES = re.compile(
    r"<!-- course-dependencies -->.*?<!-- /course-dependencies -->", re.DOTALL
)


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


def export_course(
    topic: str, output_dir: Path, *, version: str | None = None, number: int | None = None
) -> Path:
    """Build and check the selected lesson without overwriting an existing ZIP."""
    catalog = tomllib.loads((ROOT / "courses.toml").read_text())
    lesson = catalog["lessons"][topic]
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    version = version or project["version"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", version):
        raise ValueError("Version must contain only letters, digits, dots, underscores or hyphens")
    if number is not None and number < 0:
        raise ValueError("Lesson number must be nonnegative")
    prefix = "biai" if number is None else f"biai-{number:02d}"
    name = f"{prefix}-{topic}-{version}"
    output = Path(output_dir) / f"{name}.zip"
    if output.exists():
        raise FileExistsError(output)

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

    files = {}
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
        "version": project["version"],
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
        "number": number,
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
                archive.writestr(f"{name}/{path.as_posix()}", content)
            archive.comment = json.dumps(provenance).encode()
        with ZipFile(staged) as archive:
            if archive.testzip() is not None:
                raise ValueError("ZIP integrity check failed")
        # Linking publishes the completed file atomically and rejects a concurrent overwrite.
        os.link(staged, output)
    return output


def main():
    catalog = tomllib.loads((ROOT / "courses.toml").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lesson", choices=catalog["lessons"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--version", help="Release label; defaults to the project version")
    parser.add_argument("--number", type=int, help="Optional lesson number for this release")
    args = parser.parse_args()
    print(export_course(args.lesson, args.output_dir, version=args.version, number=args.number))


if __name__ == "__main__":
    main()
