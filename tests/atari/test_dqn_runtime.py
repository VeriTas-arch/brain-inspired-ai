"""DQN tensor computation, compiled task changes, and optimizer/projection parity."""

import copy

import pytest
import torch
import torch.nn.functional as F

from biai.atari.algorithms import DQNAgent, EWCWrapper, MultiHeadDQNAgent
from biai.atari.algorithms.subspace_projection import AdamSubspaceProjection, affine_layers
from biai.atari.training import seed_everything


def replay_batch(size=4):
    return dict(
        states=torch.randint(256, (size, 4, 84, 84), dtype=torch.uint8),
        next_states=torch.randint(256, (size, 4, 84, 84), dtype=torch.uint8),
        actions=torch.arange(size) % 2,
        rewards=torch.linspace(-1, 1, size),
        dones=(torch.arange(size) % 2).float(),
    )


def test_eager_dqn_matches_explicit_td_update():
    seed_everything(0)
    agent = DQNAgent(4, 2, device="cpu", target_update_freq=2, tau=0.5)
    reference = DQNAgent(4, 2, device="cpu", target_update_freq=2, tau=0.5)
    reference.load_checkpoint_state(copy.deepcopy(agent.checkpoint_state()))
    means = []
    for step in range(4):
        batch = replay_batch()
        values = reference.network(batch["states"])
        selected = values.gather(1, batch["actions"][:, None]).flatten()
        with torch.no_grad():
            expected = batch["rewards"] + reference.gamma * reference.target_network(
                batch["next_states"]
            ).max(1).values * (1 - batch["dones"])
        loss = F.mse_loss(expected, selected)
        means.append(values.detach().mean().item())
        reference.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.network.parameters(), 10.0)
        reference.optimizer.step()
        if (step + 1) % 2 == 0:
            with torch.no_grad():
                for target, online in zip(
                    reference.target_network.parameters(), reference.network.parameters()
                ):
                    target.lerp_(online, 0.5)
        result = agent.update(batch)
        assert result["loss"] == pytest.approx(loss.item())
        assert result["q_value"] == pytest.approx(sum(means) / len(means))
        for key in ("network", "target_network"):
            torch.testing.assert_close(
                getattr(agent, key).state_dict(),
                getattr(reference, key).state_dict(),
                rtol=0,
                atol=0,
            )


@pytest.mark.filterwarnings(r"default:^The CUDA Graph is empty\.:UserWarning:torch\.cuda\.graphs$")
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("variant", ("single", "multi", "ewc", "gpm"))
def test_compiled_dqn_updates_task_switches_and_checkpoints(
    variant, tmp_path, fresh_compiler_state
):
    seed_everything(0, deterministic=True)
    batch = replay_batch()
    agents = []
    for _ in range(2):
        seed_everything(5, deterministic=True)
        base = DQNAgent(4, 2) if variant == "single" else MultiHeadDQNAgent(4)
        if variant != "single":
            base.register_task("old", 2)
            base.set_task("old")
        # Initialize Adam moments eagerly, then compile the continuing learner.
        base.update(batch)
        agent = EWCWrapper(base, ewc_lambda=4.0) if variant == "ewc" else base
        if variant == "ewc":
            agent.consolidate_weights(batch)
        if variant != "single":
            agent.register_task("new", 3)
            agent.set_task("new")
        base.target_update_freq = 2
        agents.append(agent)
    eager, compiled = agents
    eager.configure_runtime(compile_enabled=False)
    compiled.configure_runtime(compile_enabled=True)
    bases = [getattr(agent, "agent", agent) for agent in agents]
    projections = []
    if variant == "gpm":
        subspaces = {}
        for name, layer in affine_layers(bases[0].backbone.network).items():
            width = layer.weight[0].numel() + int(layer.bias is not None)
            subspaces[name] = {"basis": torch.linalg.qr(torch.randn(width, 2)).Q}
        projections = [
            AdamSubspaceProjection(
                base.optimizer, base.backbone.network, subspaces, compile_projection=index == 1
            )
            for index, base in enumerate(bases)
        ]
    try:
        for step in range(8):
            if variant == "multi":
                for agent in agents:
                    agent.set_task("old" if step % 2 else "new")
            actions = [
                agent.select_action(batch["states"][0], deterministic=True) for agent in agents
            ]
            assert actions[0] == actions[1]
            metrics = [agent.update(batch) for agent in agents]
            for key in metrics[0]:
                assert metrics[1][key] == pytest.approx(metrics[0][key], rel=2e-4, abs=2e-6)
            for left, right in zip(bases[0].network.parameters(), bases[1].network.parameters()):
                torch.testing.assert_close(left, right, rtol=2e-4, atol=2e-5)
                torch.testing.assert_close(left.grad, right.grad, rtol=3e-3, atol=3e-5)
            target = "target_network" if variant == "single" else "target_backbone"
            torch.testing.assert_close(
                getattr(bases[0], target).state_dict(),
                getattr(bases[1], target).state_dict(),
                rtol=2e-4,
                atol=2e-5,
            )
            assert all(p.grad is None for p in getattr(bases[1], target).parameters())
            computation = bases[1]._compute if variant == "single" else bases[1]._computation()
            assert computation._update_graph is not None
            torch.testing.assert_close(
                bases[1].optimizer.state_dict(),
                bases[0].optimizer.state_dict(),
                rtol=3e-3,
                atol=3e-5,
            )
            if variant != "single":
                torch.testing.assert_close(
                    bases[1].heads.state_dict(),
                    bases[0].heads.state_dict(),
                    rtol=2e-4,
                    atol=2e-5,
                )
        if variant == "ewc":
            assert metrics[1]["ewc_loss"] > 0
        if projections:
            for projection in projections:
                assert projection.metrics()["optimizer_steps"] == 8
                assert all(
                    row["projection_relative_error"] < 0.01
                    for row in projection.metrics()["layers"].values()
                )
        checkpoint = tmp_path / "dqn.pt"
        compiled.save(str(checkpoint))
        saved = torch.load(checkpoint, weights_only=True)
        assert "_orig_mod" not in str(saved.keys())
        # Loading recreates multi-head modules: cached callables must not retain old heads.
        bases[1].load_checkpoint_state(saved.get("agent", saved))
        assert bases[1].select_action(batch["states"][0], deterministic=True) == actions[1]
        bases[1].update(batch)
    finally:
        for projection in projections:
            projection.close()
