"""Regression checks for the continual_mnist lesson."""

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tests.notebook_helpers import definitions


@pytest.mark.parametrize("scenario", ["task", "class"])
def test_fisher_uses_training_label_map_and_per_sample_gradients(scenario):
    ns = definitions("continual_mnist")
    torch.manual_seed(0)
    model = ns["EWCModel"](input_size=3, hidden_units=4)
    x = torch.tensor([[0.2, 0.5, 0.1], [1.0, 0.3, 0.7]])
    raw_labels = torch.tensor([2, 8])
    loader = DataLoader(TensorDataset(x, raw_labels), batch_size=1)
    actual = model.compute_fisher(loader, task_id=1, classes=[2, 8], scenario=scenario)
    expected = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    for image, output_slot in zip(x, (2, 3)):
        logits = model(image[None])
        if scenario == "task":
            loss = F.cross_entropy(logits[:, 2:4], torch.tensor([output_slot - 2]))
        else:
            loss = F.cross_entropy(logits[:, :4], torch.tensor([output_slot]))
        grads = torch.autograd.grad(loss, model.parameters())
        for (name, _), grad in zip(model.named_parameters(), grads):
            expected[name] += grad.square() / 2
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name])
    with pytest.raises(ValueError, match="batch_size=1"):
        model.compute_fisher(DataLoader(loader.dataset, batch_size=2), 1, [2, 8])
    assert not list(model.named_buffers())


def test_ewc_retains_each_old_task_anchor():
    ns = definitions("continual_mnist")
    model = ns["EWCModel"](input_size=3, hidden_units=4)
    fisher = {name: torch.ones_like(p) for name, p in model.named_parameters()}
    model.consolidate(fisher)
    first_anchor = {name: p.detach().clone() for name, p in model.named_parameters()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.25)
    model.consolidate(fisher)
    expected = (
        sum((p - first_anchor[name]).square().sum() for name, p in model.named_parameters()) / 2
    )
    torch.testing.assert_close(model.ewc_loss(), expected)
    assert model.ewc_loss() > 0
    assert len(model.fisher_matrices) == 2
    for value in fisher.values():
        value.zero_()
    torch.testing.assert_close(model.ewc_loss(), expected)


def test_continual_protocols_exclude_future_classes_from_loss_and_prediction():
    ns = definitions("continual_mnist")
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 100.0, 100.0]], requires_grad=True)
    local = torch.tensor([0])
    assert ns["evaluate_scenario"](None, logits, local, 1, 1, "task") == 1
    assert ns["evaluate_scenario"](None, logits, local, 1, 1, "class") == 0
    for scenario, zero_columns in (("task", [0, 1, 4, 5]), ("class", [4, 5])):
        loss = ns["scenario_loss"](logits, local, 1, scenario)
        (gradient,) = torch.autograd.grad(loss, logits)
        assert gradient[:, zero_columns].count_nonzero() == 0
        assert gradient[:, 2:4].count_nonzero() > 0


def test_replay_reservoir_is_bounded_repeatable_and_preserves_label_mapping():
    ns = definitions("continual_mnist")
    images = torch.arange(200, dtype=torch.float32).reshape(200, 1)
    labels = torch.tensor([8, 2] * 50 + [9, 4] * 50)
    first = TensorDataset(images[:100], labels[:100])
    second = TensorDataset(images[100:], labels[100:])
    rng_state = torch.get_rng_state().clone()
    buffers = [ns["ReplayBuffer"](40, seed=12) for _ in range(2)]
    for buffer in buffers:
        buffer.add_task(first, [8, 2], 0)
        assert buffer.size == 40
        assert buffer.images.device.type == buffer.labels.device.type == "cpu"
        assert buffer.labels.max() < 2
        buffer.add_task(second, [9, 4], 1)
        assert buffer.size == 40 and buffer.seen == 200
        sample_x, sample_y = buffer.sample(40)
        ids = sample_x[:, 0].long()
        assert ids.unique().numel() == 40
        torch.testing.assert_close(sample_y, ids % 2 + 2 * (ids >= 100))
        assert set(sample_y.tolist()) == {0, 1, 2, 3}
        assert buffer.storage_bytes() == 40 * (4 + 8)
    torch.testing.assert_close(buffers[0].images, buffers[1].images)
    torch.testing.assert_close(buffers[0].labels, buffers[1].labels)
    assert torch.equal(torch.get_rng_state(), rng_state)


def test_replay_mixed_loss_uses_old_global_labels_and_fixed_batch(monkeypatch):
    ns = definitions("continual_mnist")
    ns["CONFIG"].update(learning_rate=0.001, epochs_per_task=1)
    ns["evaluate_all_tasks"] = lambda *args: ([0.5, 0.5], 0.5)
    model = ns["EWCModel"](input_size=3, hidden_units=4)
    buffer = ns["ReplayBuffer"](4, seed=10)
    old_x = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    buffer.add_task(TensorDataset(old_x, torch.tensor([8, 2, 8, 2])), [8, 2], 0)
    current_x = torch.arange(12, 30, dtype=torch.float32).reshape(6, 3)
    loader = DataLoader(TensorDataset(current_x, torch.tensor([9, 4, 9, 4, 9, 4])), batch_size=4)
    observed = []
    inputs = []
    model.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach().clone()))
    real_ce = F.cross_entropy

    def capture(logits, targets):
        observed.append((logits.shape, targets.detach().clone()))
        return real_ce(logits, targets)

    monkeypatch.setattr(F, "cross_entropy", capture)
    history = ns["train_task"](model, loader, [], 1, [9, 4], 0, "class", buffer)
    assert [shape for shape, _ in observed] == [torch.Size([4, 4]), torch.Size([2, 4])]
    for batch, (_, targets) in zip(inputs, observed):
        ids = (batch[:, 0] / 3).long()
        torch.testing.assert_close(targets, ids % 2 + 2 * (ids >= 4))
    assert history[0]["updates"] == 2
    assert history[0]["current_examples"] == history[0]["replay_examples"] == 3
    assert model.fc3.weight.grad[4:].count_nonzero() == 0
    assert buffer.seen == 4
