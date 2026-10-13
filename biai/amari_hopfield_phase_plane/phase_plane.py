"""Numerical helpers for the two-neuron phase-plane lesson."""

from dataclasses import dataclass

import matplotlib.colors as mcolors
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.special import expit


@dataclass(frozen=True)
class TwoNeuronModel:
    """Symmetric two-neuron rate model with cross coupling."""

    tau: float = 1.0
    coupling: float = 1.0
    activation: str = "atan"
    gain: float = 2.0
    atan_scale: float = 3.14
    lam: float = 1.4
    current: float = 0.0

    def response(self, state):
        """Evaluate the selected activation function."""
        state = np.asarray(state, dtype=float)
        if self.activation == "atan":
            argument = self.atan_scale * self.gain * self.lam * state / 2
            return 2 * np.arctan(argument) / self.atan_scale
        if self.activation == "sigmoid":
            return expit(self.gain * state)
        if self.activation == "tanh":
            return np.tanh(self.gain * state)
        raise ValueError(f"Unknown activation: {self.activation}")

    def response_derivative(self, state):
        """Evaluate the derivative of the selected activation."""
        state = np.asarray(state, dtype=float)
        if self.activation == "atan":
            argument = self.atan_scale * self.gain * self.lam * state / 2
            return self.gain * self.lam / (1 + argument**2)
        if self.activation == "sigmoid":
            value = self.response(state)
            return self.gain * value * (1 - value)
        if self.activation == "tanh":
            return self.gain * (1 - self.response(state) ** 2)
        raise ValueError(f"Unknown activation: {self.activation}")

    def derivative(self, _time, state):
        """Return the state derivative for one point or a grid of points."""
        state = np.asarray(state, dtype=float)
        return -state / self.tau + self.coupling * self.response(state[::-1]) + self.current

    def jacobian(self, state):
        """Return the analytic two-by-two Jacobian."""
        response_derivative = self.response_derivative(state)
        return np.array(
            [
                [-1 / self.tau, self.coupling * response_derivative[1]],
                [self.coupling * response_derivative[0], -1 / self.tau],
            ]
        )


def integrate(model, initial_state, duration, *, samples=1001, rtol=1e-9, atol=1e-11):
    """Integrate a trajectory and sample it at evenly spaced display times."""
    times = np.linspace(0.0, duration, samples)
    solution = solve_ivp(
        model.derivative,
        (0.0, duration),
        initial_state,
        t_eval=times,
        rtol=rtol,
        atol=atol,
    )
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError(solution.message)
    return solution


def find_equilibria(model, *, limit=2.65):
    """Find distinct equilibria reached from a regular grid of initial guesses."""
    points = []
    guesses = np.linspace(-limit, limit, 11)
    for first in guesses:
        for second in guesses:
            solution = root(
                lambda state: model.derivative(0, state),
                [first, second],
                jac=model.jacobian,
            )
            point = solution.x
            residual = np.linalg.norm(model.derivative(0, point))
            if (
                residual < 1e-9
                and np.max(np.abs(point)) <= limit
                and not any(np.linalg.norm(point - known) < 1e-5 for known in points)
            ):
                points.append(point)
    return sorted(points, key=lambda point: point[0])


def classify_equilibrium(model, point):
    """Classify a two-dimensional equilibrium from its Jacobian eigenvalues."""
    eigenvalues = np.linalg.eigvals(model.jacobian(point)).real
    if np.any(np.abs(eigenvalues) < 1e-8):
        return "nonhyperbolic"
    if np.all(eigenvalues < 0):
        return "stable node"
    if np.all(eigenvalues > 0):
        return "unstable node"
    return "saddle"


def draw_phase_plane(axis, model, *, names=("x1", "x2"), title):
    """Draw nullclines, flow, equilibria, and one trajectory on an axis."""
    grid = np.linspace(-2.5, 2.5, 81)
    first, second = np.meshgrid(grid, grid)
    first_rate, second_rate = model.derivative(0, np.array([first, second]))
    speed = np.hypot(first_rate, second_rate)
    linewidth = 0.4 + 2.5 * speed / speed.max()
    axis.streamplot(
        first,
        second,
        first_rate,
        second_rate,
        color="thistle",
        density=0.9,
        linewidth=linewidth,
        arrowsize=0.9,
    )

    parameter = np.unique(np.r_[np.linspace(-2.5, 2.5, 1201), np.linspace(-0.05, 0.05, 1201)])
    response = model.tau * (model.coupling * model.response(parameter) + model.current)
    axis.plot(response, parameter, color="C0", lw=1.4, label=f"{names[0]} nullcline")
    axis.plot(parameter, response, "--", color="C1", lw=1.4, label=f"{names[1]} nullcline")

    colors = {"stable node": "C2", "saddle": "C3", "unstable node": "C4"}
    labels = set()
    for point in find_equilibria(model):
        kind = classify_equilibrium(model, point)
        axis.plot(
            *point,
            "o",
            color=colors.get(kind, "C5"),
            ms=5,
            zorder=5,
            label=kind if kind not in labels else None,
        )
        labels.add(kind)

    trajectory = integrate(model, [0.1, -0.1], 50)
    axis.plot(*trajectory.y, color="C2", lw=1.4, label="trajectory")
    axis.set(
        xlim=(-2.65, 2.65),
        ylim=(-2.65, 2.65),
        xlabel=names[0],
        ylabel=names[1],
        title=title,
    )
    axis.legend(loc="best", fontsize=7)


def energy(output_1, output_2, *, lam=1.4, coupling=1.0):
    """Evaluate the two-neuron Hopfield energy in output coordinates."""
    potential_1 = -4 * np.log(np.cos(np.pi * output_1 / 2)) / (np.pi**2 * lam)
    potential_2 = -4 * np.log(np.cos(np.pi * output_2 / 2)) / (np.pi**2 * lam)
    return -coupling * output_1 * output_2 + potential_1 + potential_2


def draw_energy_landscape(axis):
    """Draw the symmetric two-well energy landscape on an axis."""
    values = np.linspace(-0.995, 0.995, 401)
    output_1, output_2 = np.meshgrid(values, values)
    landscape = energy(output_1, output_2)
    lower, upper = np.percentile(landscape, [2, 85])
    image = axis.imshow(
        landscape,
        origin="lower",
        extent=[values[0], values[-1], values[0], values[-1]],
        cmap="turbo",
        norm=mcolors.Normalize(vmin=lower, vmax=upper, clip=True),
        aspect="equal",
    )
    levels = np.linspace(lower, upper, 28)
    axis.contour(output_1, output_2, landscape, levels=levels, colors="white", linewidths=0.5)
    axis.set(
        xlabel="neuron 1 output",
        ylabel="neuron 2 output",
        title="Energy landscape",
    )
    return image
