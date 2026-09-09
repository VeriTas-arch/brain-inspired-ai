"""Check functional preservation and Adam-aware projection, not just tensor shapes."""

import copy

import pytest
import torch
import torch.nn as nn

from algorithms.ppo import MultiHeadPPOAgent
from algorithms.subspace_projection import (
    AdamSubspaceProjection,
    affine_matrix,
    build_input_subspaces,
    sample_conv_inputs,
)
from training.ppo_runtime import CollectedRollout, PPOLearner


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize(
    "kernel,stride,padding,dilation", ((8, 4, 0, 1), (4, 2, 0, 1), (3, 1, 2, 2))
)
def test_sampled_conv_inputs_equal_selected_unfold_patches(
    device, kernel, stride, padding, dilation
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    layer = nn.Conv2d(4, 8, kernel, stride=stride, padding=padding, dilation=dilation)
    x = torch.arange(2 * 4 * 32 * 32, device=device, dtype=torch.float32).reshape(2, 4, 32, 32)
    full = torch.nn.functional.unfold(x, kernel, dilation, padding, stride).transpose(1, 2)
    indices = torch.randint(full.shape[1], (2, 16), generator=torch.Generator().manual_seed(7)).to(
        device
    )
    expected = full.gather(1, indices[..., None].expand(-1, -1, full.shape[-1]))
    actual = sample_conv_inputs(x, layer, 16, torch.Generator().manual_seed(7))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def tiny_agent(device="cpu"):
    agent = MultiHeadPPOAgent(4, device=device)
    agent.backbone = nn.Sequential(nn.Linear(4, 512), nn.Tanh()).to(device)
    agent.network = agent.backbone
    agent.register_task("old", 3)
    agent.register_task("new", 2)
    agent.set_task("new")
    return agent


def test_boundary_basis_matches_direct_uncentered_svd_with_bias():
    net = nn.Sequential(nn.Linear(3, 2))
    states = torch.tensor([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 128, 0]], dtype=torch.uint8)
    rng = torch.get_rng_state().clone()
    subspace = build_input_subspaces(net, states, threshold=0.95, seed=9)["0"]
    torch.testing.assert_close(rng, torch.get_rng_state())
    x = torch.cat((states.double() / 255, torch.ones(4, 1)), dim=1)
    _, singular, vh = torch.linalg.svd(x)
    rank = int(torch.searchsorted(singular.square().cumsum(0), 0.95 * singular.square().sum())) + 1
    assert subspace["rank"] == rank
    basis = subspace["basis"].double()
    torch.testing.assert_close(basis @ basis.T, vh[:rank].T @ vh[:rank], atol=2e-7, rtol=2e-7)


def test_convolution_uses_receptive_fields_and_one_bias_coordinate():
    net = nn.Sequential(nn.Conv2d(1, 2, 2, stride=2))
    # Every patch is identical, so the rank-one span is known independently.
    states = torch.full((3, 1, 4, 4), 255, dtype=torch.uint8)
    entry = build_input_subspaces(net, states, threshold=0.995, seed=3)["0"]
    assert entry["sample_count"] == 3 * 16
    assert entry["dimension"] == 5
    assert entry["rank"] == 1
    torch.testing.assert_close(entry["basis"] @ entry["basis"].T, torch.full((5, 5), 0.2))


def test_projected_adam_matches_displacement_formula_with_inherited_moments():
    torch.manual_seed(17)
    net = nn.Sequential(nn.Linear(2, 2, dtype=torch.float64))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    for _ in range(3):
        optimizer.zero_grad()
        net(torch.tensor([[0.5, -1.0]], dtype=torch.float64)).square().sum().backward()
        optimizer.step()
    reference = copy.deepcopy(net)
    raw_optimizer = torch.optim.Adam(reference.parameters(), lr=0.01)
    raw_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    basis = torch.linalg.qr(
        torch.tensor([[1.0, 2.0], [2.0, -1.0], [1.0, 1.0]], dtype=torch.float64)
    ).Q
    importance = torch.tensor([1.0, 0.3], dtype=torch.float64)
    subspaces = {"0": {"basis": basis, "importance": importance}}
    before = affine_matrix(net[0]).detach().clone()
    projection = AdamSubspaceProjection(optimizer, net, subspaces)
    try:
        for model, opt in ((net, optimizer), (reference, raw_optimizer)):
            opt.zero_grad()
            model(torch.tensor([[2.0, 1.0]], dtype=torch.float64)).sum().backward()
            opt.step()
        raw = affine_matrix(reference[0]).detach() - before
        expected = raw - (raw @ basis) @ basis.T
        actual = affine_matrix(net[0]).detach() - before
        torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-12)
        # The first protected affine input [1, 2, 1] has unchanged output under GPM.
        torch.testing.assert_close(
            actual @ basis[:, 0], torch.zeros(2, dtype=torch.float64), atol=1e-14, rtol=0
        )
        assert actual.norm() > 0
        assert projection.metrics()["optimizer_steps"] == 1
        assert projection.metrics()["layers"]["0"]["projection_relative_error"] < 1e-12
        for p, q in zip(net.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(
                optimizer.state[p]["exp_avg"], raw_optimizer.state[q]["exp_avg"]
            )
    finally:
        projection.close()


def test_projecting_raw_gradient_before_adam_does_not_preserve_orthogonality():
    p = nn.Parameter(torch.zeros(2, dtype=torch.float64))
    basis = torch.tensor([1.0, 2.0], dtype=torch.float64)
    basis /= basis.norm()
    gradient = torch.tensor([2.0, -1.0], dtype=torch.float64)
    assert abs(float(gradient @ basis)) < 1e-12
    p.grad = gradient
    torch.optim.Adam([p], lr=0.01).step()
    assert abs(float(p.detach() @ basis)) > 0.001


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_projection_runs_once_per_ppo_minibatch_and_leaves_old_heads_unchanged(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    agent = tiny_agent(device)
    states = torch.randint(256, (8, 4), dtype=torch.uint8, device=device)
    with torch.no_grad():
        actions, logs, values = agent.sample_action_and_value(states)
    basis = torch.linalg.qr(torch.randn(5, 2)).Q
    projection = AdamSubspaceProjection(
        agent.optimizer,
        agent.backbone,
        {"0": {"basis": basis, "importance": torch.ones(2)}},
    )
    old = {k: v.clone() for k, v in agent.actors["old"].state_dict().items()}
    critic = {k: v.clone() for k, v in agent.critics["old"].state_dict().items()}
    new = agent.actors["new"].weight.detach().clone()
    data = dict(
        states=states,
        actions=actions,
        log_probs=logs,
        values=values.flatten(),
        rewards=torch.arange(8).float(),
        dones=torch.ones(8),
    )
    try:
        PPOLearner(agent, update_epochs=2, minibatch_size=4).update(
            CollectedRollout(data, torch.zeros(1), 8, ())
        )
        assert projection.steps == 4
        for k, v in old.items():
            torch.testing.assert_close(v, agent.actors["old"].state_dict()[k], rtol=0, atol=0)
        for k, v in critic.items():
            torch.testing.assert_close(v, agent.critics["old"].state_dict()[k], rtol=0, atol=0)
        assert not torch.equal(new, agent.actors["new"].weight)
    finally:
        projection.close()


def test_accumulation_preserves_old_span_and_uses_total_task_energy():
    net = nn.Sequential(nn.Linear(3, 2, bias=False))
    old = {"0": {"basis": torch.eye(3)[:, :1]}}
    # Old basis covers 80%; a 90% total threshold needs only one of two residual axes.
    states = torch.tensor([[255, 0, 0]] * 8 + [[0, 255, 0], [0, 0, 255]], dtype=torch.uint8)
    entry = build_input_subspaces(net, states, threshold=0.9, seed=3, previous=old)["0"]
    basis = entry["basis"]
    assert entry["rank"] == 2
    assert entry["added_rank"] == 1
    torch.testing.assert_close(basis.T @ basis, torch.eye(2))
    torch.testing.assert_close(basis @ basis.T @ old["0"]["basis"], old["0"]["basis"])
    assert entry["captured_energy_fraction"] >= 0.9 - 1e-7
    covered = build_input_subspaces(net, states[:8], threshold=0.9, seed=3, previous={"0": entry})
    assert covered["0"]["added_rank"] == 0
    full = {"0": {"basis": torch.eye(3)}}
    saturated = build_input_subspaces(net, states, threshold=1, seed=3, previous=full)["0"]
    assert saturated["rank"] == 3
    assert saturated["added_rank"] == 0
