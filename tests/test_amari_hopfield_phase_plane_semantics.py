"""Numerical checks for the two-neuron phase-plane lesson."""

import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import minimize_scalar

from biai.amari_hopfield_phase_plane.phase_plane import (
    TwoNeuronModel,
    classify_equilibrium,
    energy,
    find_equilibria,
    integrate,
)


def test_arctangent_models_have_two_stable_nodes_and_one_saddle():
    for gain, expected_stable_state in ((2.0, 1.238764073), (100.0, 1.397807737)):
        model = TwoNeuronModel(tau=1.4, gain=gain)
        points = find_equilibria(model)

        assert [classify_equilibrium(model, point) for point in points] == [
            "stable node",
            "saddle",
            "stable node",
        ]
        assert_allclose(points[1], 0, atol=1e-8)
        assert_allclose(points[0], -points[2], atol=1e-8)
        assert_allclose(points[2][0], expected_stable_state, atol=1e-8)


def test_analytic_jacobian_matches_finite_difference():
    models = (
        TwoNeuronModel(),
        TwoNeuronModel(tau=1.4, gain=100.0),
        TwoNeuronModel(coupling=2.0, activation="sigmoid", gain=0.8),
        TwoNeuronModel(coupling=-5.0, activation="tanh", gain=0.4),
    )
    state = np.array([0.12, -0.45])
    step = 1e-6

    for model in models:
        numerical = np.column_stack(
            [
                (
                    model.derivative(0, state + step * direction)
                    - model.derivative(0, state - step * direction)
                )
                / (2 * step)
                for direction in np.eye(2)
            ]
        )
        assert_allclose(model.jacobian(state), numerical, rtol=1e-6, atol=1e-8)


def test_time_series_reach_the_reported_long_time_states():
    sigmoid = TwoNeuronModel(coupling=2.0, activation="sigmoid", gain=4.0)
    critical_tanh = TwoNeuronModel(activation="tanh", gain=1.0)

    sigmoid_solution = integrate(sigmoid, [10.0, -1.0], 2000.0, samples=1001)
    critical_solution = integrate(critical_tanh, [10.0, -1.0], 2000.0, samples=1001)

    assert_allclose(sigmoid_solution.y[:, -1], 2.0, atol=0.001)
    assert_allclose(
        critical_solution.y[:, -1],
        np.sqrt(3 / (2 * 2000)),
        rtol=0.01,
    )


def test_energy_is_symmetric_with_two_off_center_minima():
    values = np.linspace(-0.95, 0.95, 100)
    assert_allclose(energy(values, values), energy(-values, -values))

    minimum = minimize_scalar(
        lambda value: energy(value, value),
        bounds=(0.01, 0.95),
        method="bounded",
    )
    assert 0.5 < minimum.x < 0.65
    assert minimum.fun < energy(0, 0)
