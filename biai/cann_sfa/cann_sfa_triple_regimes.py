"""
Simulation and plots for three representative 1D CANN+SFA regimes:

- Smooth Tracking
- Oscillatory Tracking
- Traveling Wave

References
----------
- <https://papers.nips.cc/paper_files/paper/2022/file/d6797a91df19b768409b5178642dcb26-Paper-Conference.pdf>
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


@dataclass(frozen=True)
class RegimeConfig:
    """Configuration of one CANN+SFA simulation regime.

    Attributes
    ----------
    name : str
        Regime name used in plot titles.
    num : int
        Number of neurons (spatial grid points on the ring).
    tau : float
        Time constant of neural activity variable u.
    tau_v : float
        Time constant of adaptation variable v.
    g : float
        Gain of the firing-rate nonlinearity.
    k : float
        Strength of global divisive inhibition.
    a : float
        Spatial width of recurrent kernel and external drive.
    m : float
        Coupling strength from u to adaptation v.
    dt : float
        Simulation time step in ms.
    alpha : float
        Amplitude of external Gaussian stimulus.
    v_ext : float
        External target velocity along the ring (rad/ms).
    target_pos0 : float
        Initial target position on the ring (rad).
    stim_duration : float | None
        Stimulus duration in ms. None means full trial duration.
    total_duration : float
        Total simulation duration in ms.
    """

    name: str
    num: int = 256
    tau: float = 1.0
    tau_v: float = 50.0
    g: float = 1.0
    k: float = 1.0
    a: float = 0.3
    m: float = 0.05
    dt: float = 0.1
    alpha: float = 0.2
    v_ext: float = 0.004
    target_pos0: float = 0.0
    stim_duration: float | None = None
    total_duration: float = 1400.0


# A brief moving cue selects the traveling wave direction before stimulus removal.
REGIMES = {
    "smooth": RegimeConfig(name="Smooth Tracking", m=0.05, alpha=0.2, v_ext=0.004),
    "oscillatory": RegimeConfig(
        name="Oscillatory Tracking", m=16.7 / 50.0, alpha=0.2, v_ext=0.20 * 0.3 / 50.0
    ),
    "travelling_wave": RegimeConfig(
        name="Traveling Wave", m=0.90, alpha=0.03, v_ext=0.004, stim_duration=30.0
    ),
}


class CANN1DSFA:
    def __init__(self, num, m=0.1, tau=1.0, tau_v=10.0, g=4.0, k=0.1, a=0.5):
        self.num = num
        self.tau = tau
        self.tau_v = tau_v
        self.g = g
        self.k = k
        self.a = a
        self.m = m
        self.z_range = 2.0 * np.pi
        self.x = np.arange(num) * self.z_range / num - np.pi

        self.u = np.zeros(num)
        self.v = np.zeros(num)
        self.input = np.zeros(num)
        self.center = 0.0

        # The recurrent matrix is circulant; store its first column in Fourier space.
        distance = wrapped_difference(self.x - self.x[0], z_range=self.z_range)
        kernel = np.exp(-0.5 * np.square(distance / self.a)) / (np.sqrt(2.0 * np.pi) * self.a)
        self.kernel_fft = np.fft.rfft(kernel)
        self.phase_kernel = np.exp(1j * self.x)

    def update(self, dt):
        u2 = np.square(self.u)
        r = self.g * u2 / (1.0 + self.k * np.sum(u2))
        irec = np.fft.irfft(self.kernel_fft * np.fft.rfft(r), n=self.num)
        # Both right-hand sides use the old state (exponential Euler).
        next_u = self.u + (-np.expm1(-dt / self.tau)) * (irec + self.input - self.v - self.u)
        self.v += (-np.expm1(-dt / self.tau_v)) * (self.m * self.u - self.v)
        self.u[:] = np.maximum(next_u, 0.0)
        self.input[:] = 0.0
        population_vector = np.sum(self.u * self.phase_kernel)
        self.center = (
            float(np.angle(population_vector))
            if abs(population_vector) > 1e-12 * max(1.0, self.u.sum())
            else np.nan
        )


def wrapped_difference(x, z_range=2.0 * np.pi):
    return ((x + 0.5 * z_range) % z_range) - 0.5 * z_range


def gaussian_drive(x, pos, amplitude, a, z_range=2.0 * np.pi):
    delta = wrapped_difference(x - pos, z_range=z_range)
    return amplitude * np.exp(-(delta**2) / (4.0 * a**2))


def wrap_trajectory_for_display(values, z_range=2.0 * np.pi):
    wrapped = np.asarray(wrapped_difference(np.asarray(values), z_range=z_range))
    jumps = np.abs(np.diff(wrapped)) > (0.5 * z_range)
    wrapped = wrapped.copy()
    wrapped[1:][jumps] = np.nan
    return wrapped


def regime_title(config: RegimeConfig) -> str:
    return f"{config.name} | m={config.m:.3f}, alpha={config.alpha:.3f}, v_ext={config.v_ext:.4f}"


def setup_position_axis(ax):
    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks([-np.pi, 0.0, np.pi])
    ax.set_yticklabels(["-π", "0", "π"])


def build_inputs(config: RegimeConfig, x):
    if (
        config.num < 3
        or min(config.dt, config.total_duration, config.tau, config.tau_v, config.a) <= 0
    ):
        raise ValueError("num must be at least 3; time scales, duration and width must be positive")
    total_steps = int(round(config.total_duration / config.dt))
    if total_steps < 1 or not np.isclose(total_steps * config.dt, config.total_duration):
        raise ValueError("total_duration must be a positive integer multiple of dt")
    ts = np.arange(total_steps) * config.dt
    x = np.asarray(x)
    stim_duration = config.total_duration if config.stim_duration is None else config.stim_duration
    stim_steps = max(0, min(total_steps, int(round(stim_duration / config.dt))))

    inputs = np.zeros((total_steps, config.num))
    target_visible = np.full(total_steps, np.nan)
    if stim_steps > 0:
        target_visible[:stim_steps] = wrapped_difference(
            config.target_pos0 + config.v_ext * ts[:stim_steps]
        )
        inputs[:stim_steps] = np.asarray(
            gaussian_drive(
                x[None, :],
                target_visible[:stim_steps, None],
                amplitude=config.alpha,
                a=config.a,
                z_range=2.0 * np.pi,
            )
        )
    return inputs, target_visible


def simulate_regime(config: RegimeConfig):
    """Run from rest; pair post-update states with the input held over each step.

    ``ts`` contains state times and ``input_ts`` contains input onset times.
    ``relative`` uses unwrapped coordinates and is undefined after cue removal.
    """
    model = CANN1DSFA(
        num=config.num,
        tau=config.tau,
        tau_v=config.tau_v,
        g=config.g,
        k=config.k,
        a=config.a,
        m=config.m,
    )
    inputs, target_visible = build_inputs(config, model.x)
    input_ts = np.arange(inputs.shape[0]) * config.dt
    ts = input_ts + config.dt
    u = np.zeros((inputs.shape[0], config.num))
    center = np.zeros(inputs.shape[0])
    for i, current_input in enumerate(inputs):
        model.input[:] = current_input
        model.update(config.dt)
        u[i] = model.u
        center[i] = model.center
    # Unwrap each defined segment without propagating undefined centers forever.
    valid_indices = np.flatnonzero(np.isfinite(center))
    for segment in np.split(valid_indices, np.flatnonzero(np.diff(valid_indices) > 1) + 1):
        center[segment] = np.unwrap(center[segment])
    target_unwrapped = np.where(
        np.isfinite(target_visible), config.target_pos0 + config.v_ext * input_ts, np.nan
    )
    if valid_indices.size:
        first = valid_indices[0]
        if np.isfinite(target_unwrapped[first]):
            center += 2 * np.pi * np.round((target_unwrapped[first] - center[first]) / (2 * np.pi))
    return {
        "config": config,
        "ts": ts,
        "input_ts": input_ts,
        "x": np.asarray(model.x),
        "u": u,
        "input": inputs,
        "center": center,
        "target_visible": target_visible,
        "target_unwrapped": target_unwrapped,
        "relative": center - target_unwrapped,
    }


def plot_regime(result):
    config = result["config"]
    ts = result["ts"]
    center_display = wrap_trajectory_for_display(result["center"])
    target_display = wrap_trajectory_for_display(result["target_visible"])

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, layout="constrained")

    axes[0].plot(ts, center_display, linewidth=1.8, label="bump")
    axes[0].plot(ts, target_display, linewidth=1.6, label="target")
    axes[0].set_ylabel("position (rad)")
    setup_position_axis(axes[0])
    axes[0].legend(loc="upper right")
    axes[0].set_title(regime_title(config))

    if np.all(np.isfinite(result["target_visible"])):
        axes[1].plot(ts, result["relative"], color="tab:red", linewidth=1.2)
        axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        axes[1].set_ylabel("bump - applied target (rad)")
        axes[1].set_title("Unwrapped tracking error")
    else:
        axes[1].plot(ts, result["center"], color="tab:red", linewidth=1.2)
        axes[1].set_ylabel("unwrapped position (rad)")
        axes[1].set_title("Bump trajectory after cue removal")
        if config.stim_duration is not None:
            axes[1].axvline(config.stim_duration, color="black", linestyle="--", label="cue off")
            axes[1].legend()

    extent = [ts[0], ts[-1], result["x"][0], result["x"][-1]]
    heatmap = axes[2].imshow(
        result["u"].T, aspect="auto", origin="lower", extent=extent, cmap="magma"
    )
    fig.colorbar(heatmap, ax=axes, label="activity u", shrink=0.3, anchor=(0, 0))
    axes[2].plot(ts, target_display, color="cyan", linewidth=1.0)
    axes[2].set_xlabel("time (ms)")
    axes[2].set_ylabel("network position")
    setup_position_axis(axes[2])
    axes[2].set_title("activity heatmap")

    plt.show()
    return fig


def animate_regime(
    result,
    frame_stride: int = 60,
    fps: int = 12,
):
    config = result["config"]
    ts = result["ts"]
    x = result["x"]
    u = result["u"]
    inputs = result["input"]
    center = wrap_trajectory_for_display(result["center"])
    target = wrap_trajectory_for_display(result["target_visible"])

    frames = list(range(0, len(ts), frame_stride))
    if frames[-1] != len(ts) - 1:
        frames.append(len(ts) - 1)

    max_y = float(max(u.max(), inputs.max()) * 1.15)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    profile_ax, center_ax = axes

    profile_ax.set_xlim(float(x[0]), float(x[-1]))
    profile_ax.set_ylim(float(min(inputs.min(), 0.0) * 1.1), max_y)
    profile_ax.set_ylabel("activity")
    profile_ax.set_title(regime_title(config))
    (input_line,) = profile_ax.plot([], [], linewidth=1.8, label="external input")
    (bump_line,) = profile_ax.plot([], [], linewidth=2.0, linestyle="--", label="bump")
    profile_ax.legend(loc="upper right")

    center_ax.set_xlim(float(ts[0]), float(ts[-1]))
    center_ax.set_xlabel("time (ms)")
    center_ax.set_ylabel("position")
    setup_position_axis(center_ax)
    (target_line,) = center_ax.plot([], [], linewidth=1.8, label="target")
    (center_line,) = center_ax.plot([], [], linewidth=1.8, label="bump")
    cursor = center_ax.axvline(float(ts[0]), color="black", linestyle=":", linewidth=1.0)
    status = center_ax.text(
        0.02,
        0.96,
        "",
        transform=center_ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "black"},
    )
    center_ax.legend(loc="upper right")

    def init():
        input_line.set_data([], [])
        bump_line.set_data([], [])
        target_line.set_data([], [])
        center_line.set_data([], [])
        cursor.set_xdata([float(ts[0]), float(ts[0])])
        status.set_text("")
        return (input_line, bump_line, target_line, center_line, cursor, status)

    def update(frame_id):
        idx = frames[frame_id]
        input_line.set_data(x, inputs[idx])
        bump_line.set_data(x, u[idx])
        target_line.set_data(ts[: idx + 1], target[: idx + 1])
        center_line.set_data(ts[: idx + 1], center[: idx + 1])
        cursor.set_xdata([float(ts[idx]), float(ts[idx])])
        status.set_text(
            f"t={ts[idx]:.1f} ms\n"
            f"bump={center[idx]:.3f}\n"
            f"target={target[idx]:.3f}\n"
            f"delta={wrapped_difference(center[idx] - target[idx]):.3f}"
        )
        return (input_line, bump_line, target_line, center_line, cursor, status)

    animation = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(frames),
        interval=1000 / fps,
        blit=False,
        repeat=True,
    )
    fig.tight_layout()

    plt.close(fig)
    return animation
