"""Small output conventions shared by the teaching entry points."""

import hashlib
import json
import platform
import shutil
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from biai.paths import ASSETS_DIR, PROJECT_ROOT, RESULTS_DIR


def check_result_path(output_dir: Path) -> None:
    """Keep training and evaluation output away from source and reference assets."""
    output = Path(output_dir).resolve()
    protected = (ASSETS_DIR, *(PROJECT_ROOT / name for name in ("biai", "tests", ".git")))
    if PROJECT_ROOT.resolve().is_relative_to(output) or any(
        output.is_relative_to(path.resolve()) for path in protected
    ):
        raise ValueError(f"Protected output directory: {output}")


def check_output(output_dir: Path, *, force: bool = False) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    if output_dir.exists() and not force:
        raise FileExistsError(f"Results already exist: {output_dir}; use --force to replace them")


def publish_outputs(outputs: dict[Path, Path], *, force: bool = False) -> list[Path]:
    """Publish with rollback on failure; return old directories for caller cleanup."""
    for output_dir in outputs.values():
        check_output(output_dir, force=force)
    previous = []
    with ExitStack() as rollback:
        for staging, output_dir in outputs.items():
            if output_dir.exists():
                backup = staging.with_name(staging.name + "-previous")
                output_dir.rename(backup)
                rollback.callback(backup.rename, output_dir)
                previous.append(backup)
            staging.rename(output_dir)
            rollback.callback(output_dir.rename, staging)
        rollback.pop_all()
    return previous


@contextmanager
def result_directory(output_dir: Path, *, force: bool = False):
    """Publish successful output only; keep failed work available for inspection."""
    output_dir = Path(output_dir).resolve()
    check_output(output_dir, force=force)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".pending-{output_dir.name}-", dir=output_dir.parent))
    try:
        yield staging
        previous = publish_outputs({staging: output_dir}, force=force)
    except BaseException:
        print(f"Unpublished results retained at {staging}", file=sys.stderr)
        raise
    for backup in previous:
        shutil.rmtree(backup)


def case_name(
    protocol: str, algorithm: str, *, game: str = "", method: str = "", seed: int = 0
) -> str:
    suffix = game.removesuffix("-v5").replace("/", "_").lower() if game else method
    name = "-".join(part for part in (protocol, algorithm, suffix) if part)
    return f"{name}-seed{seed}" if seed else name


def run_metadata() -> dict:
    """Record runtime versions without inspecting Git or the source tree."""
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("torch", "gymnasium", "ale-py")},
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
        output_dir = RESULTS_DIR / name
    output_dir = Path(output_dir)
    check_result_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("x") as output:
        output.write(json.dumps(config, indent=2) + "\n")
    metadata = {**run_metadata(), "case": name}
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
