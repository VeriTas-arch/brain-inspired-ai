"""Regression checks for the Amari-Hopfield associative-memory lesson."""

import numpy as np
import pytest

from biai.amari_hopfield_model.hopfield_model import (
    AmariHopfieldNetwork,
    add_bit_flip_noise,
    add_mask_noise,
    load_patterns,
)


def test_hebbian_weights_are_symmetric_with_zero_diagonal():
    patterns = load_patterns()
    network = AmariHopfieldNetwork(patterns.shape[1])
    network.store_patterns(patterns)

    expected = patterns.T @ patterns / patterns.shape[1]
    np.fill_diagonal(expected, 0)
    np.testing.assert_allclose(network.weight, expected)
    np.testing.assert_allclose(network.weight, network.weight.T)
    np.testing.assert_array_equal(np.diag(network.weight), 0)


def test_asynchronous_recovery_lowers_energy_and_recovers_stored_pattern():
    patterns = load_patterns()
    network = AmariHopfieldNetwork(patterns.shape[1])
    network.store_patterns(patterns)
    rng = np.random.default_rng(0)
    corrupted = add_bit_flip_noise(patterns[1], probability=0.4, rng=rng)

    recovered, energy = network.recover_async(
        corrupted, n_updates=10_000, rng=rng, track_energy=True
    )

    np.testing.assert_array_equal(recovered, patterns[1])
    assert np.all(np.diff(energy) <= 1e-5)


def test_mask_noise_rejects_bounds_outside_the_image():
    image = np.ones((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="row_bounds"):
        add_mask_noise(image, row_bounds=(0, 5), column_bounds=(1, 3))
