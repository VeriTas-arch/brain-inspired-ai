"""Build notebook figures and fixed-window GIFs from the current teaching cases."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import matplotlib.pyplot as plt

from scripts.tutorial_examples import TEACHING_RESULTS, load_teaching_results

GAMES = ("Pong-v5", "Breakout-v5", "SpaceInvaders-v5")
METHODS = (("finetune", "Sequential"), ("ewc", "EWC"), ("gpm", "GPM"))


def build_figures(results_dir: Path = TEACHING_RESULTS):
    """Plot actual evaluation means; leave unobserved tasks blank."""
    results_dir = Path(results_dir)
    data = load_teaching_results(results_dir)
    output = results_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for axis, game in zip(axes, GAMES[:2], strict=True):
        for algorithm in ("ppo", "dqn"):
            key = f"single-{algorithm}-{game.removesuffix('-v5').lower()}/learning.json"
            evaluations = data["learning_curves"][key]["evaluations"]
            axis.plot(
                [row["step"] / 1e6 for row in evaluations],
                [row["mean_raw_reward"] for row in evaluations],
                marker="o",
                label=algorithm.upper(),
            )
        axis.set(title=game.removesuffix("-v5"), xlabel="Environment transitions (millions)")
        axis.set_ylabel("Mean raw reward (10 episodes)")
        if game == "Pong-v5":
            axis.set_ylim(-22, 22)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "single_learning.png", dpi=160)
    plt.close(figure)
    for algorithm in ("ppo", "dqn"):
        figure, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        for axis, game in zip(axes, GAMES, strict=True):
            for suffix, label in METHODS:
                rows = data["continual"][f"continual-{algorithm}-{suffix}"]["score_matrix"]
                rows = [row for row in rows if row["scores"][game] is not None]
                axis.plot(
                    [row["stage"] for row in rows],
                    [row["scores"][game] for row in rows],
                    marker="o",
                    label=label,
                )
            axis.set(
                title=game.removesuffix("-v5"), xlabel="Completed training stage", xlim=(0.8, 3.2)
            )
            axis.set_xticks([1, 2, 3], ["1: Pong", "2: Breakout", "3: SpaceInvaders"], fontsize=8)
            axis.set_ylabel("Mean raw reward (10 episodes)")
            if game == "Pong-v5":
                axis.set_ylim(-22, 22)
            axis.grid(alpha=0.25)
            axis.legend()
        figure.tight_layout()
        figure.savefig(output / f"{algorithm}_continual.png", dpi=160)
        plt.close(figure)
    (output / "figures.json").write_text(
        json.dumps(
            {
                "inputs": {
                    str(path.relative_to(results_dir)): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for case in sorted(data["runs"])
                    for name in ("learning.json", "training_summary.json")
                    if (path := results_dir / case / name).is_file()
                },
                "figures": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(output.glob("*.png"))
                },
            },
            indent=2,
        )
        + "\n"
    )
    return output


def build_videos(run: Path = TEACHING_RESULTS):
    """Reuse final videos and evaluate the shared first-stage checkpoint once."""
    from algorithms import MultiHeadPPOAgent
    from scripts.evaluate import _record_example_video, _set_agent_eval
    from training import configure_ppo_runtime, seed_everything

    run = Path(run)
    data = load_teaching_results(run)
    media = run / "continual-ppo-finetune" / "evaluation" / "videos"
    media.mkdir(parents=True, exist_ok=True)
    stage = run / "continual-ppo-finetune/checkpoints/stage-01.pt"
    stage_video = media / "pong-stage1.mp4"
    configure_ppo_runtime("async")
    seed_everything(0, deterministic=True)
    agent = MultiHeadPPOAgent(4)
    agent.load(str(stage))
    agent.configure_runtime(compile_enabled=True)
    _set_agent_eval(agent)
    _record_example_video(agent, "Pong-v5", stage_video, seed=0)
    sources = [(media / "pong-stage1.gif", stage_video, stage)]
    for model in data["models"]:
        metrics = model["metrics"]
        checkpoint = run / model["checkpoint"]
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != model["sha256"]:
            raise ValueError(f"Checkpoint changed: {checkpoint}")
        directory = (run / model["metrics_path"]).parent / "videos"
        if metrics["mode"] == "single" and metrics["game"] == "Pong-v5":
            name = directory / "pong.gif"
            sources.append(
                (name, directory / f"Pong-v5_{metrics['algorithm']}_eval_gameplay.mp4", checkpoint)
            )
        if metrics["mode"] == "continual" and metrics["algorithm"] == "ppo":
            method = directory / "pong.gif"
            sources.append(
                (method, directory / "continual_ppo_Pong-v5_eval_gameplay.mp4", checkpoint)
            )
    records = []
    for target, video, checkpoint in sources:
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(video),
                "-filter_complex_threads",
                "1",
                "-filter_complex",
                "[0:v]trim=start=0:end=10,setpts=PTS-STARTPTS,fps=12,scale=160:-1:flags=lanczos,split[a][b];"
                "[a]palettegen[p];[b][p]paletteuse=dither=bayer",
                "-loop",
                "0",
                str(target),
            ],
            check=True,
        )
        records.append(
            {
                "gif": str(target.relative_to(run)),
                "gif_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "video": str(video.relative_to(run)),
                "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                "checkpoint": str(checkpoint.relative_to(run)),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
        )
    (run / "figures" / "media.json").write_text(
        json.dumps(
            {
                "training_commits": {
                    case: metadata["commit"] for case, metadata in data["runs"].items()
                },
                "seed": 0,
                "start_seconds": 0,
                "duration_seconds": 10,
                "gif_fps": 12,
                "source_fps": 30,
                "playback": "same timeline as evaluation MP4, not simulator wall time",
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", action="store_true", help="Requires local checkpoints and MP4s")
    parser.add_argument("--results-dir", type=Path, default=TEACHING_RESULTS)
    args = parser.parse_args()
    output = build_figures(args.results_dir)
    if args.videos:
        build_videos(args.results_dir)
    print(output)


if __name__ == "__main__":
    main()
