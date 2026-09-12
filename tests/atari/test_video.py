"""Exercise the actual streaming encoder and its failure boundary."""

import sys

import imageio_ffmpeg
import numpy as np
import pytest

from biai.atari.training import VideoRecorder, run_evaluation_episodes


@pytest.mark.parametrize("shape", ((84, 84), (210, 160, 3)))
def test_streamed_video_decodes_every_frame_without_retaining_arrays(tmp_path, shape):
    path = tmp_path / "episode.mp4"
    with VideoRecorder(str(path), fps=30) as recorder:
        frame = np.zeros(shape, dtype=np.uint8)
        for value in (0, 50, 100, 150, 200, 250):
            frame.fill(value)
            recorder.add_frame(frame)
        assert recorder.frame_count == 6
        assert not any(isinstance(value, (list, np.ndarray)) for value in vars(recorder).values())
    recorder.close()
    reader = imageio_ffmpeg.read_frames(str(path))
    metadata = next(reader)
    frames = list(reader)
    assert metadata["size"] == (shape[1], shape[0])
    assert len(frames) == 6
    means = [np.frombuffer(frame, dtype=np.uint8).mean() for frame in frames]
    np.testing.assert_allclose(means, (0, 50, 100, 150, 200, 250), atol=3)


def test_late_encoder_failure_reaches_caller_even_with_nonempty_output(monkeypatch, tmp_path):
    encoder = tmp_path / "failing_encoder"
    encoder.write_text(
        f"#!{sys.executable}\n"
        "import sys\nfrom pathlib import Path\n"
        "sys.stdin.buffer.read()\n"
        "Path(sys.argv[-1]).write_bytes(b'partial video')\n"
        "sys.stderr.write('late encoding failure')\n"
        "sys.exit(1)\n"
    )
    encoder.chmod(0o700)
    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: str(encoder))
    path = tmp_path / "failed.mp4"
    with pytest.raises(RuntimeError, match="late encoding failure"):
        with VideoRecorder(str(path)) as recorder:
            recorder.add_frame(np.zeros((84, 84), dtype=np.uint8))
    assert path.stat().st_size > 0
    assert recorder.writer is None
    assert recorder.errors is None


def test_recording_callback_preserves_complete_episode_evaluation():
    class Environment:
        def reset(self):
            self.steps = 0
            return 0

        def step(self, action):
            self.steps += 1
            return self.steps, 2.0, self.steps == 3, False

    class Agent:
        def select_action(self, state, deterministic):
            assert deterministic
            return 0

    env = Environment()
    observations = []
    rewards = run_evaluation_episodes(
        Agent(), env, 2, 3, frame_callback=lambda: observations.append(env.steps)
    )
    assert rewards == [6.0, 6.0]
    assert observations == [0, 1, 2, 0, 1, 2]
    with pytest.raises(RuntimeError, match="did not finish"):
        run_evaluation_episodes(Agent(), env, 1, 2, frame_callback=lambda: None)
