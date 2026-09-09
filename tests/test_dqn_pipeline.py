"""Ordered deferred DQN diagnostics."""

import copy

import pytest
import torch

from algorithms import DQNAgent, EWCWrapper, MultiHeadDQNAgent
from training import DQNMetrics, seed_everything


@pytest.mark.parametrize("wrapped", (False, True))
def test_deferred_diagnostics_preserve_every_metric_and_q_history(wrapped):
    seed_everything(17)
    if wrapped:
        base = MultiHeadDQNAgent(4, device="cpu")
        base.register_task("task", 2)
        base.set_task("task")
        eager = EWCWrapper(base)
        eager.configure_runtime()
    else:
        eager = DQNAgent(4, 2, device="cpu")
    actual = copy.deepcopy(eager)
    if wrapped:
        actual.configure_runtime()
    else:
        actual.configure_runtime(compile_enabled=False)
    batches = [
        dict(
            states=torch.randint(256, (2, 4, 84, 84), dtype=torch.uint8),
            next_states=torch.randint(256, (2, 4, 84, 84), dtype=torch.uint8),
            actions=torch.arange(2),
            rewards=torch.tensor([0.0, 1.0]),
            dones=torch.tensor([0.0, 1.0]),
        )
        for _ in range(7)
    ]
    expected = [eager.update(batch) for batch in batches]
    observed = []
    metrics = DQNMetrics(actual, observed.append, capacity=3)
    for batch in batches:
        actual.update(batch, metrics_sink=metrics.record)
    assert len(observed) == 6
    metrics.flush()
    assert observed == expected
    if not wrapped:
        assert actual.q_value_history == eager.q_value_history
