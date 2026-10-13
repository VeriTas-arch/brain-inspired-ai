"""Check CANN dynamics against a dense reference and ring-coordinate semantics."""

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biai.cann_sfa.cann_sfa_triple_regimes import (
    CANN1DSFA,
    REGIMES,
    RegimeConfig,
    animate_regime,
    plot_regime,
    simulate_regime,
)


@pytest.mark.parametrize("num", [63, 64])
def test_fft_updates_match_independent_dense_exponential_euler(num):
    model = CANN1DSFA(num, m=0.3, tau=1.0, tau_v=50.0, g=1.0, k=1.0, a=0.3)
    rng = np.random.default_rng(0)
    u = rng.uniform(0.0, 0.3, num)
    v = rng.uniform(0.0, 0.1, num)
    model.u[:] = u
    model.v[:] = v
    distance = np.abs(model.x[:, None] - model.x[None, :])
    distance = np.minimum(distance, 2 * np.pi - distance)
    weights = np.exp(-(distance**2) / (2 * model.a**2)) / (np.sqrt(2 * np.pi) * model.a)
    dt = 0.1
    for _ in range(40):
        drive = rng.uniform(0.0, 0.2, num)
        rates = model.g * u**2 / (1 + model.k * (u**2).sum())
        next_u = np.maximum(
            np.exp(-dt / model.tau) * u
            + (1 - np.exp(-dt / model.tau)) * (weights @ rates + drive - v),
            0,
        )
        v = np.exp(-dt / model.tau_v) * v + (1 - np.exp(-dt / model.tau_v)) * model.m * u
        u = next_u
        model.input[:] = drive
        model.update(dt)
        np.testing.assert_allclose(model.u, u, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(model.v, v, rtol=1e-12, atol=1e-14)


def test_tracking_error_is_continuous_across_the_ring_boundary():
    config = replace(REGIMES["smooth"], target_pos0=np.pi - 0.02, total_duration=40.0)
    result = simulate_regime(config)
    assert np.ptp(result["target_visible"]) > 6
    assert np.max(np.abs(np.diff(result["relative"]))) < 0.01
    shifted = simulate_regime(replace(config, target_pos0=config.target_pos0 + 2 * np.pi))
    np.testing.assert_allclose(shifted["relative"], result["relative"], atol=1e-12)


def test_state_timestamps_and_cue_removal():
    config = RegimeConfig(name="short cue", num=64, total_duration=2.0, dt=0.1, stim_duration=0.5)
    result = simulate_regime(config)
    np.testing.assert_allclose(result["ts"], np.arange(1, 21) * config.dt)
    np.testing.assert_allclose(result["input_ts"], np.arange(20) * config.dt)
    assert np.all(result["input"][:5].max(axis=1) > 0)
    assert np.all(result["input"][5:] == 0)
    assert np.all(np.isnan(result["target_visible"][5:]))
    assert np.all(np.isnan(result["relative"][5:]))
    # The first state is the exponential response to the first input from rest.
    np.testing.assert_allclose(
        result["u"][0], (1 - np.exp(-config.dt / config.tau)) * result["input"][0]
    )


def test_no_activity_has_no_decoded_position():
    result = simulate_regime(RegimeConfig(name="silent", alpha=0, total_duration=1.0))
    assert not result["u"].any()
    assert np.isnan(result["center"]).all()
    assert np.isnan(result["relative"]).all()


@pytest.mark.parametrize("direction", [-1, 1])
def test_moving_cue_selects_autonomous_wave_direction(direction):
    result = simulate_regime(replace(REGIMES["travelling_wave"], v_ext=direction * 0.004))
    late = result["ts"] > 700
    speed = np.polyfit(result["ts"][late], result["center"][late], 1)[0]
    assert direction * speed > 0.04
    assert result["u"][-1].max() > 0.5
    assert np.all(result["input"][result["input_ts"] >= 30] == 0)


@pytest.mark.integration
def test_plot_and_inline_animation_do_not_write_output(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    result = simulate_regime(RegimeConfig(name="plot", num=64, total_duration=2.0))
    figure = plot_regime(result)
    assert len(figure.axes) == 4
    animation = animate_regime(result, frame_stride=10)
    assert "data:image/png;base64" in animation.to_jshtml()
    assert not list(tmp_path.iterdir())
