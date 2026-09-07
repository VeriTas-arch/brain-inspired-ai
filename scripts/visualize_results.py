"""Visualize the JSON artifacts produced by training and evaluation scripts."""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def _make_axes(panel_count: int):
    columns = min(2, panel_count)
    rows = math.ceil(panel_count / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(7 * columns, 4 * rows),
        squeeze=False,
    )
    return figure, axes.flat


def _finish_figure(figure, axes, used_panels: int, output_path: Path) -> Path:
    for axis in axes[used_panels:]:
        figure.delaxes(axis)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def _evaluation_series(results_dir: Path) -> list[tuple[str, list[float]]]:
    """Read reward series from the current evaluation JSON schema."""
    series = []
    for metrics_path in sorted(results_dir.glob("**/metrics.json")):
        data = _load_json(metrics_path)
        relative_run = metrics_path.parent.relative_to(results_dir)
        mode = data.get("mode")
        algorithm = data.get("algorithm", "unknown")

        if mode == "single" and isinstance(data.get("rewards"), list):
            label = f"{relative_run}: {data.get('game', 'unknown')} ({algorithm})"
            series.append((label, data["rewards"]))
        elif mode in {"continual", "multitask"} and isinstance(data.get("games"), dict):
            for game, game_data in sorted(data["games"].items()):
                rewards = game_data.get("rewards")
                if isinstance(rewards, list):
                    label = f"{relative_run}: {game} ({mode}, {algorithm})"
                    series.append((label, rewards))
    return series


def plot_evaluation_results(
    results_dir: str | Path,
    output_path: str | Path = "outputs/evaluation_rewards.png",
) -> Path:
    """Plot every evaluation series on its own raw-reward axis."""
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    series = _evaluation_series(results_dir)
    if not series:
        raise ValueError(f"No compatible metrics.json files found under {results_dir}")

    figure, axes_iterator = _make_axes(len(series))
    axes = list(axes_iterator)
    for axis, (label, rewards) in zip(axes[: len(series)], series, strict=True):
        axis.plot(range(1, len(rewards) + 1), rewards, marker="o")
        axis.set_title(label)
        axis.set_xlabel("Evaluation Episode")
        axis.set_ylabel("Raw Environment Reward")
        axis.grid(True, alpha=0.3)

    output_path = _finish_figure(figure, axes, len(series), Path(output_path))
    print(f"Evaluation plot saved to {output_path}")
    return output_path


def _continual_series(results_dir: Path) -> list[tuple[str, list[int], list[float], float | None]]:
    """Read per-task trajectories from current continual evaluation reports."""
    series = []
    for report_path in sorted(results_dir.glob("**/continual_evaluation.json")):
        data = _load_json(report_path)
        relative_run = report_path.parent.relative_to(results_dir)
        score_matrix = data.get("score_matrix")
        if not isinstance(score_matrix, list):
            continue

        task_names = list(data.get("per_task", {}))
        if not task_names and score_matrix:
            task_names = list(score_matrix[0].get("scores", {}))
        for task_name in task_names:
            stages = []
            scores = []
            for row in score_matrix:
                score = row.get("scores", {}).get(task_name)
                if score is not None:
                    stages.append(int(row["stage"]))
                    scores.append(float(score))
            if not scores:
                continue
            forgetting = data.get("per_task", {}).get(task_name, {}).get("forgetting")
            series.append((f"{relative_run}: {task_name}", stages, scores, forgetting))
    return series


def plot_continual_results(
    results_dir: str | Path,
    output_path: str | Path = "outputs/continual_scores.png",
) -> Path:
    """Plot each task's raw score trajectory without averaging across games."""
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    series = _continual_series(results_dir)
    if not series:
        raise ValueError(f"No compatible continual_evaluation.json files found under {results_dir}")

    figure, axes_iterator = _make_axes(len(series))
    axes = list(axes_iterator)
    for axis, (label, stages, scores, forgetting) in zip(axes[: len(series)], series, strict=True):
        axis.plot(stages, scores, marker="o")
        title = label
        if forgetting is not None:
            title += f" (forgetting={float(forgetting):.2f})"
        axis.set_title(title)
        axis.set_xlabel("Training Stage")
        axis.set_ylabel("Raw Environment Reward")
        axis.set_xticks(stages)
        axis.grid(True, alpha=0.3)

    output_path = _finish_figure(figure, axes, len(series), Path(output_path))
    print(f"Continual-learning plot saved to {output_path}")
    return output_path


def main() -> None:
    """Parse command-line arguments and render current result artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="outputs", help="Experiment output directory")
    parser.add_argument(
        "--type",
        choices=["evaluation", "continual"],
        default="evaluation",
        help="JSON artifact type to visualize",
    )
    parser.add_argument("--output", help="Output path for the plot")
    args = parser.parse_args()

    if args.type == "evaluation":
        plot_evaluation_results(
            args.results_dir,
            args.output or "outputs/evaluation_rewards.png",
        )
    else:
        plot_continual_results(
            args.results_dir,
            args.output or "outputs/continual_scores.png",
        )


if __name__ == "__main__":
    main()
