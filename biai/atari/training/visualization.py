"""Training plots and streaming MP4 recording."""

import subprocess
from contextlib import ExitStack, suppress
from pathlib import Path
from tempfile import TemporaryFile

import cv2
import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import torch


class VideoRecorder:
    """Write uint8 grayscale or RGB observations without retaining a trajectory."""

    def __init__(self, output_path: str, fps: int = 30):
        self.output_path = Path(output_path)
        self.fps = fps
        self.writer = None
        self.frame_count = 0
        self.errors = None

    def add_frame(self, frame):
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            raise ValueError("Video frames must be uint8")
        if frame.ndim == 2:
            frame = np.repeat(frame[:, :, None], 3, axis=2)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected grayscale or RGB frame, received {frame.shape}")
        if min(frame.shape[:2]) < 84:
            frame = cv2.resize(frame, (max(84, frame.shape[1]), max(84, frame.shape[0])))
        frame = np.ascontiguousarray(frame)
        if self.writer is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.errors = TemporaryFile()
            self.writer = subprocess.Popen(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{frame.shape[1]}x{frame.shape[0]}",
                    "-r",
                    str(self.fps),
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-pix_fmt",
                    "yuv420p",
                    "-b:v",
                    "2M",
                    "-threads",
                    "2",
                    str(self.output_path),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self.errors,
            )
        self.writer.stdin.write(frame)
        self.frame_count += 1

    def close(self):
        """Finalize the stream and check FFmpeg's exit status, including late failures."""
        process = self.writer
        errors, self.errors = self.errors, None

        def stop_encoder():
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    process.kill()
                process.wait()
            self.writer = None

        with ExitStack() as resources:
            if errors is not None:
                resources.callback(errors.close)
            if process is not None:
                resources.callback(stop_encoder)
                write_error = None
                try:
                    process.stdin.close()
                except BrokenPipeError as error:
                    write_error = error
                code = process.wait()
                errors.seek(0)
                message = errors.read().decode("utf-8", errors="replace")
                if code or write_error:
                    raise RuntimeError(f"Video encoder failed ({code}): {message}") from write_error
                if self.frame_count and (
                    not self.output_path.is_file() or self.output_path.stat().st_size == 0
                ):
                    raise RuntimeError(f"Video encoder produced no output: {self.output_path}")

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


class MetricsPlotter:
    """Plot training metrics."""

    def __init__(self):
        """Initialize plotter."""
        self.metrics = {}

    def add_metric(self, name: str, value: float):
        """Add a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def plot(self, output_path: str, max_cols: int = 2):
        """Plot metrics in a compact grid layout.

        Args:
            output_path: Path to save the figure
            max_cols: Maximum number of subplots per row (default: 2)
        """
        if not self.metrics:
            print("No metrics to plot.")
            return

        num_metrics = len(self.metrics)
        ncols = min(max_cols, num_metrics)
        nrows = int(np.ceil(num_metrics / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5 * ncols, 3.5 * nrows),
            squeeze=False,
        )

        # Plot each metric into its own subplot
        for idx, (name, values) in enumerate(sorted(self.metrics.items())):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row][col]

            if name == "episode_reward" and len(values) > 10:
                ax.plot(values, alpha=0.3, color="blue", label="Raw")
                window = min(10, len(values) // 5)
                if window > 1:
                    moving_avg = []
                    for i in range(len(values)):
                        start = max(0, i - window + 1)
                        moving_avg.append(sum(values[start : i + 1]) / (i - start + 1))
                    ax.plot(moving_avg, color="red", linewidth=2, label=f"MA({window})")
                    ax.legend()
            else:
                ax.plot(values)

            ax.set_title(name.replace("_", " ").title())
            xlabel = "Episode" if name == "episode_reward" else "Step"
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)

        for idx in range(num_metrics, nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            fig.delaxes(axes[row][col])

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()
