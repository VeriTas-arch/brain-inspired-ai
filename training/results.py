"""Small output conventions shared by the teaching entry points."""

import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path


def check_output(output_dir: Path, *, force: bool = False) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    if output_dir.exists() and not force:
        raise FileExistsError(f"Results already exist: {output_dir}; use --force to replace them")


def publish_output(staging: Path, output_dir: Path, *, force: bool = False) -> None:
    """Replace one completed result, restoring the previous directory if the move fails."""
    check_output(output_dir, force=force)
    previous = staging.with_name(staging.name + "-previous")
    if output_dir.exists():
        output_dir.rename(previous)
    try:
        staging.rename(output_dir)
    except BaseException:
        if previous.exists():
            previous.rename(output_dir)
        raise
    if previous.exists():
        shutil.rmtree(previous)


@contextmanager
def result_directory(output_dir: Path, *, force: bool = False):
    """Publish successful output only; keep failed work available for inspection."""
    output_dir = Path(output_dir).resolve()
    check_output(output_dir, force=force)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".pending-{output_dir.name}-", dir=output_dir.parent))
    try:
        yield staging
        publish_output(staging, output_dir, force=force)
    except BaseException:
        print(f"Unpublished results retained at {staging}", file=sys.stderr)
        raise


def case_name(
    protocol: str, algorithm: str, *, game: str = "", method: str = "", seed: int = 0
) -> str:
    suffix = game.removesuffix("-v5").replace("/", "_").lower() if game else method
    name = "-".join(part for part in (protocol, algorithm, suffix) if part)
    return f"{name}-seed{seed}" if seed else name


def run_metadata(project: Path) -> dict:
    """Record versions without copying source or requiring a clean worktree."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project, capture_output=True, text=True
    )
    changes = subprocess.run(
        ["git", "status", "--porcelain"], cwd=project, capture_output=True, text=True
    )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(changes.stdout) if changes.returncode == 0 else None,
        "versions": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in ("torch", "torchrl", "tensordict", "gymnasium", "ale-py")
            },
        },
    }


def prepare_case(output_dir: Path | None, **config) -> Path:
    """Create one case; an existing configuration must never be overwritten."""
    name = case_name(
        config["protocol"],
        config["algorithm"],
        game=config["games"][0] if config["protocol"] == "single" else "",
        method=config.get("method", ""),
        seed=config["seed"],
    )
    if output_dir is None:
        output_dir = Path("results") / name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("x") as output:
        output.write(json.dumps(config, indent=2) + "\n")
    metadata = {**run_metadata(Path(__file__).resolve().parents[1]), "case": name}
    (output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for name in ("checkpoints", "figures", "videos"):
        (output_dir / name).mkdir(exist_ok=True)
    return output_dir


def parameter_digest(module) -> str:
    """Compare initialization without retaining a full checkpoint."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()
