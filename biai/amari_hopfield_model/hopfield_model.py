"""Amari-Hopfield associative memory with asynchronous and synchronous updates."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_PATTERN_PATH = Path(__file__).with_name("test_data.npy")
OTHER_PATTERN_PATH = Path(__file__).with_name("other_images.npy")


def load_patterns(path: str | Path = DEFAULT_PATTERN_PATH) -> np.ndarray:
    """Load bipolar patterns stored as rows of a NumPy array."""
    patterns = np.asarray(np.load(path), dtype=np.float32)
    if patterns.ndim != 2:
        raise ValueError("patterns must have shape (n_patterns, n_units)")
    if not np.all(np.isin(patterns, (-1, 1))):
        raise ValueError("patterns must contain only -1 and 1")
    return patterns


def add_bit_flip_noise(
    image: np.ndarray,
    probability: float,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Flip each bipolar pixel independently with the given probability."""
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    noisy = np.asarray(image).copy()
    flips = rng.random(noisy.shape) < probability
    noisy[flips] *= -1
    return noisy


def add_mask_noise(
    image: np.ndarray,
    row_bounds: tuple[int, int],
    column_bounds: tuple[int, int],
    *,
    mask_value: float = 1,
) -> np.ndarray:
    """Replace one rectangular region of a two-dimensional image."""
    noisy = np.asarray(image).copy()
    if noisy.ndim != 2:
        raise ValueError("image must be two-dimensional")
    row_start, row_stop = row_bounds
    column_start, column_stop = column_bounds
    if not (0 <= row_start < row_stop <= noisy.shape[0]):
        raise ValueError("row_bounds are outside the image")
    if not (0 <= column_start < column_stop <= noisy.shape[1]):
        raise ValueError("column_bounds are outside the image")
    noisy[row_start:row_stop, column_start:column_stop] = mask_value
    return noisy


class AmariHopfieldNetwork:
    """Fully connected associative memory for bipolar patterns."""

    def __init__(self, num_units: int, threshold: float = 0.0) -> None:
        if num_units <= 0:
            raise ValueError("num_units must be positive")
        self.num_units = num_units
        self.threshold = threshold
        self.weight = np.zeros((num_units, num_units), dtype=np.float32)

    def store_patterns(self, patterns: np.ndarray) -> None:
        """Set the weights with the Hebbian outer-product rule."""
        patterns = np.asarray(patterns, dtype=np.float32)
        if patterns.ndim != 2 or patterns.shape[1] != self.num_units:
            raise ValueError(f"patterns must have shape (n_patterns, {self.num_units})")
        if len(patterns) == 0:
            raise ValueError("at least one pattern is required")
        if not np.all(np.isin(patterns, (-1, 1))):
            raise ValueError("patterns must contain only -1 and 1")

        self.weight = patterns.T @ patterns / self.num_units
        np.fill_diagonal(self.weight, 0)

    def recover_async(
        self,
        initial_state: np.ndarray,
        n_updates: int,
        *,
        rng: np.random.Generator,
        track_energy: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Recover a pattern by updating one randomly selected unit at a time."""
        state = np.asarray(initial_state, dtype=np.float32).copy()
        if state.shape != (self.num_units,):
            raise ValueError(f"initial_state must have shape ({self.num_units},)")
        if not np.all(np.isin(state, (-1, 1))):
            raise ValueError("initial_state must contain only -1 and 1")
        if n_updates < 0:
            raise ValueError("n_updates must be non-negative")

        energy_history = np.empty(n_updates, dtype=np.float32) if track_energy else None
        current_energy = self.energy(state) if track_energy else None
        for step, unit in enumerate(rng.integers(0, self.num_units, size=n_updates)):
            local_field = self.weight[unit] @ state - self.threshold
            old_value = state[unit]
            new_value = 1 if local_field >= 0 else -1
            state[unit] = new_value
            if energy_history is not None:
                current_energy -= (new_value - old_value) * local_field
                energy_history[step] = current_energy

        if energy_history is not None:
            return state, energy_history
        return state

    def recover_sync(
        self,
        initial_state: np.ndarray,
        n_steps: int,
    ) -> np.ndarray:
        """Recover a pattern by updating every unit from the previous state."""
        state = np.asarray(initial_state, dtype=np.float32).copy()
        if state.shape != (self.num_units,):
            raise ValueError(f"initial_state must have shape ({self.num_units},)")
        if not np.all(np.isin(state, (-1, 1))):
            raise ValueError("initial_state must contain only -1 and 1")
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")

        for _ in range(n_steps):
            local_field = self.weight @ state - self.threshold
            state = np.where(local_field >= 0, 1, -1).astype(np.float32)
        return state

    def energy(self, state: np.ndarray) -> float:
        """Return the Hopfield energy of one network state."""
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (self.num_units,):
            raise ValueError(f"state must have shape ({self.num_units},)")
        return float(-0.5 * state @ self.weight @ state + self.threshold * state.sum())


def _plot_patterns(patterns: np.ndarray, overlap: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, image, title in zip(
        axes,
        (*patterns[:2], overlap),
        ("Pattern 1", "Pattern 2", "Shared foreground"),
        strict=True,
    ):
        axis.imshow(image.reshape(28, 28), cmap="gray", vmin=-1, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    plt.show()


def _plot_recovery(
    original: np.ndarray,
    corrupted: np.ndarray,
    recovered: np.ndarray,
    energy_history: np.ndarray,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, image, label in zip(
        axes,
        (original, corrupted, recovered),
        ("Original", "Corrupted", "Recovered"),
        strict=True,
    ):
        axis.imshow(image.reshape(28, 28), cmap="gray", vmin=-1, vmax=1)
        axis.set_title(label)
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(energy_history)
    plt.xlabel("Asynchronous update")
    plt.ylabel("Energy")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Run the three recovery examples used in the course."""
    patterns = load_patterns()
    image1, image2 = patterns[:2]
    overlap = np.where((image1 == 1) & (image2 == 1), 1, -1).astype(np.float32)
    _plot_patterns(patterns, overlap)

    network = AmariHopfieldNetwork(num_units=patterns.shape[1])
    network.store_patterns(patterns)
    rng = np.random.default_rng(0)
    n_updates = 10_000

    bit_flip = add_bit_flip_noise(image2, 0.4, rng=rng)
    recovered, energy = network.recover_async(bit_flip, n_updates, rng=rng, track_energy=True)
    _plot_recovery(image2, bit_flip, recovered, energy, "Bit-flip noise")

    mask = add_mask_noise(
        image1.reshape(28, 28),
        row_bounds=(7, 20),
        column_bounds=(8, 25),
    ).reshape(-1)
    recovered, energy = network.recover_async(mask, n_updates, rng=rng, track_energy=True)
    _plot_recovery(image1, mask, recovered, energy, "Occlusion")

    corrupted_overlap = add_bit_flip_noise(overlap, 0.5, rng=rng)
    recovered, energy = network.recover_async(
        corrupted_overlap, n_updates, rng=rng, track_energy=True
    )
    _plot_recovery(overlap, corrupted_overlap, recovered, energy, "Ambiguous cue")


if __name__ == "__main__":
    main()
