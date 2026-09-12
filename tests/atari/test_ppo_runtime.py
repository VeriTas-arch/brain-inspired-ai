"""Tests for the PPO runtime shared by training protocols."""

import copy

import pytest
import torch

from biai.atari.environments import VectorStep
from biai.atari.training import CollectedRollout, PPOCollector, PPOLearner, flatten_rollout_data


class _FakeAgent:
    device = torch.device("cpu")
    gamma = 0.5
    clip_coef = 0.1
    ent_coef = 0.01
    vf_coef = 0.5

    @staticmethod
    def _values(states: torch.Tensor) -> torch.Tensor:
        return states[:, 0, 0, 0].to(torch.float32).unsqueeze(1)

    def sample_action_and_value(self, states: torch.Tensor):
        batch_size = len(states)
        return (
            torch.zeros(batch_size, dtype=torch.long),
            torch.zeros(batch_size),
            self._values(states),
        )

    def get_action_and_value(self, states: torch.Tensor, actions: torch.Tensor | None = None):
        if actions is None:
            actions = torch.zeros(len(states), dtype=torch.long)
        zeros = torch.zeros(len(states))
        return actions, zeros, zeros, self._values(states)

    def get_value(self, states: torch.Tensor) -> torch.Tensor:
        return self._values(states)


class _FakeVectorEnvironment:
    num_envs = 2

    def __init__(self) -> None:
        self.step_index = 0

    @staticmethod
    def _observations(first: int, second: int) -> torch.Tensor:
        return torch.stack(
            (
                torch.full((4, 84, 84), first, dtype=torch.uint8),
                torch.full((4, 84, 84), second, dtype=torch.uint8),
            )
        )

    def reset(self) -> torch.Tensor:
        return self._observations(1, 2)

    def step_and_reset(self, actions: torch.Tensor) -> VectorStep:
        torch.testing.assert_close(actions, torch.zeros(2, dtype=torch.long))
        if self.step_index == 0:
            result = VectorStep(
                observations=self._observations(100, 3),
                transition_observations=self._observations(9, 3),
                rewards=torch.tensor([1.0, 2.0]),
                terminated=torch.tensor([False, False]),
                truncated=torch.tensor([True, False]),
            )
        else:
            result = VectorStep(
                observations=self._observations(4, 200),
                transition_observations=self._observations(4, 8),
                rewards=torch.tensor([3.0, 4.0]),
                terminated=torch.tensor([False, True]),
                truncated=torch.tensor([False, True]),
            )
        self.step_index += 1
        return result


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_collector_preserves_vector_trajectories_and_time_limit_bootstrap(device) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    agent = _FakeAgent()
    agent.device = torch.device(device)
    learner = PPOLearner(agent, update_epochs=3, minibatch_size=32)
    collector = PPOCollector(_FakeVectorEnvironment(), learner, rollout_length=8)

    rollout = collector.collect(transition_budget=4)

    assert rollout.transition_count == 4
    assert rollout.data["states"].device.type == device
    assert rollout.episode_returns == (1.0, 6.0)
    assert collector.episode_count == 2
    torch.testing.assert_close(
        rollout.data["rewards"],
        torch.tensor([[5.5, 2.0], [3.0, 4.0]]),
    )
    torch.testing.assert_close(
        rollout.data["dones"],
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )
    assert rollout.data["states"][0, 0, 0, 0, 0].item() == 1
    assert rollout.data["states"][1, 0, 0, 0, 0].item() == 100
    torch.testing.assert_close(rollout.next_value.cpu(), torch.tensor([4.0, 200.0]))


def test_collector_requires_exact_vector_transition_budget() -> None:
    learner = PPOLearner(_FakeAgent())
    collector = PPOCollector(_FakeVectorEnvironment(), learner)

    with pytest.raises(ValueError, match="divisible"):
        collector.collect(transition_budget=3)


def test_compiled_learner_compiles_only_policy_hot_paths(monkeypatch) -> None:
    compiled_functions = []

    def fake_compile(function, **kwargs):
        compiled_functions.append((function, kwargs))
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    learner = PPOLearner(_FakeAgent(), compile_policy=True)

    assert learner.compiled
    assert len(compiled_functions) == 3
    assert all(
        options["mode"] == "reduce-overhead" and options["fullgraph"]
        for _, options in compiled_functions
    )
    assert compiled_functions[-1][1]["dynamic"] is False


def test_learner_configures_optional_regularizer_for_same_runtime(monkeypatch) -> None:
    from biai.atari.algorithms import EWCWrapper

    agent = EWCWrapper(_FakeAgent())
    calls = []
    monkeypatch.setattr(agent, "configure_regularizer", lambda: calls.append(True))
    PPOLearner(agent)
    assert calls == [True]
    with pytest.raises(ValueError, match="only one PPO regularizer"):
        PPOLearner(agent, regularizer=lambda: torch.zeros(()))


def test_flatten_rollout_data_keeps_time_before_environment_order() -> None:
    rollout_data = {
        "states": torch.arange(4).reshape(2, 2, 1, 1, 1),
        "actions": torch.tensor([[0, 1], [2, 3]]),
    }

    flattened = flatten_rollout_data(rollout_data)

    torch.testing.assert_close(flattened["states"].flatten(), torch.arange(4))
    torch.testing.assert_close(flattened["actions"], torch.arange(4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("multi_head", (False, True))
def test_fullgraph_cuda_policy_loss_and_task_switching(
    multi_head, monkeypatch, fresh_compiler_state
):
    from biai.atari.algorithms.ppo import MultiHeadPPOAgent, PPOAgent

    # Compare graph semantics at IEEE precision; production keeps backend defaults.
    monkeypatch.setattr(torch.backends.cuda.matmul, "fp32_precision", "ieee")
    monkeypatch.setattr(torch.backends.cudnn.conv, "fp32_precision", "ieee")
    torch.manual_seed(41)
    if multi_head:
        agent = MultiHeadPPOAgent(4, device="cuda")
        agent.register_task("pong", 3)
        agent.register_task("breakout", 4)
        tasks = ("pong", "breakout", "pong")
    else:
        agent = PPOAgent(4, 3, device="cuda")
        tasks = (None,)
    eager = PPOLearner(agent, minibatch_size=4)
    compiled = PPOLearner(agent, minibatch_size=4, compile_policy=True)
    states = torch.randint(256, (4, 4, 84, 84), dtype=torch.uint8, device="cuda")
    for task in tasks:
        if task is not None:
            agent.set_task(task)
        with torch.no_grad():
            # Match RolloutBuffer ownership: outputs must survive the next CUDA graph.
            actions, log_probs, values = (
                tensor.clone() for tensor in compiled.sample_action_and_value(states)
            )
            _, expected_log_probs, _, expected_values = agent.get_action_and_value(states, actions)
        torch.testing.assert_close(log_probs, expected_log_probs, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(values, expected_values, atol=1e-5, rtol=1e-4)
        inputs = (
            states,
            actions,
            log_probs,
            torch.randn(4, device="cuda"),
            values.flatten() + 0.2,
            values.flatten(),
            torch.arange(4, device="cuda"),
            None,
        )
        expected = eager._minibatch_loss(*inputs)
        actual = compiled._minibatch_loss(*inputs)
        for left, right in zip(actual, expected, strict=True):
            torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-4)
        parameters = [p for group in agent.optimizer.param_groups for p in group["params"]]
        expected_gradients = torch.autograd.grad(expected[0], parameters, allow_unused=True)
        actual_gradients = torch.autograd.grad(actual[0], parameters, allow_unused=True)
        for left, right in zip(actual_gradients, expected_gradients, strict=True):
            if right is None:
                assert left is None
                continue
            torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("variant", ("single", "multi", "ewc", "gpm"))
def test_compiled_ppo_optimizer_matches_eager_with_regularization_and_projection(
    variant, monkeypatch, fresh_compiler_state
):
    from biai.atari.algorithms import EWCWrapper, MultiHeadPPOAgent, PPOAgent
    from biai.atari.algorithms.subspace_projection import AdamSubspaceProjection, affine_layers

    monkeypatch.setattr(torch.backends.cuda.matmul, "fp32_precision", "ieee")
    monkeypatch.setattr(torch.backends.cudnn.conv, "fp32_precision", "ieee")
    torch.manual_seed(83)
    if variant == "single":
        base = PPOAgent(4, 3, device="cuda")
    else:
        base = MultiHeadPPOAgent(4, device="cuda")
        base.register_task("old", 2)
        base.register_task("new", 3)
        base.set_task("new")
    reference = EWCWrapper(base, ewc_lambda=0.2) if variant == "ewc" else base
    if variant == "ewc":
        for name, parameter in reference._collect_regularized_params().items():
            reference.aggregated_fisher[name] = torch.full_like(parameter, 0.01)
            reference.aggregated_mean[name] = parameter.detach().clone() + 0.1
            reference.aggregated_correction[name] = torch.zeros((), device="cuda")
    actual = copy.deepcopy(reference)
    actual_base = actual.agent if variant == "ewc" else actual
    learners = [
        PPOLearner(reference, update_epochs=2, minibatch_size=4),
        PPOLearner(actual, update_epochs=2, minibatch_size=4, compile_policy=True),
    ]
    projections = []
    if variant == "gpm":
        for agent, compiled in ((base, False), (actual_base, True)):
            subspaces = {}
            for name, layer in affine_layers(agent.backbone).items():
                basis = torch.zeros(layer.weight[0].numel() + 1, 1)
                basis[0, 0] = 1
                subspaces[name] = {"basis": basis}
            projections.append(
                AdamSubspaceProjection(
                    agent.optimizer, agent.backbone, subspaces, compile_projection=compiled
                )
            )
    states = torch.randint(256, (8, 4, 84, 84), device="cuda", dtype=torch.uint8)
    with torch.no_grad():
        actions, logs, values = base.sample_action_and_value(states)
    rollout = CollectedRollout(
        dict(
            states=states,
            actions=actions,
            log_probs=logs,
            values=values.flatten(),
            rewards=torch.arange(8, dtype=torch.float32) / 8,
            dones=torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ),
        torch.zeros(1, device="cuda"),
        8,
        (),
    )
    try:
        for update in range(2):
            metrics = []
            for learner in learners:
                torch.manual_seed(100 + update)
                metrics.append(learner.update(rollout))
            assert metrics[1] == pytest.approx(metrics[0], rel=1e-3, abs=3e-5)
            if variant == "ewc":
                assert metrics[1]["ewc_loss"] > 0
            assert learners[1]._update_graphs
            torch.testing.assert_close(
                actual_base.optimizer.state_dict(),
                base.optimizer.state_dict(),
                rtol=1e-3,
                atol=3e-6,
            )
            for expected_group, actual_group in zip(
                base.optimizer.param_groups, actual_base.optimizer.param_groups, strict=True
            ):
                for expected, observed in zip(
                    expected_group["params"], actual_group["params"], strict=True
                ):
                    torch.testing.assert_close(observed, expected, rtol=1e-3, atol=3e-6)
    finally:
        for projection in projections:
            projection.close()


def test_ppo_benchmark_honors_explicit_torch_thread_count():
    from biai.atari.scripts.benchmark_ppo_runtime import benchmark_configuration

    previous = torch.get_num_threads()
    try:
        result = benchmark_configuration(
            game="Pong-v5",
            backend="async",
            compile_policy=False,
            transitions=2,
            warmup_transitions=1,
            num_envs=1,
            batch_size=32,
            seed=0,
            device="cpu",
            environment_only=True,
            torch_threads=2,
        )
        assert torch.get_num_threads() == 2
        assert result.transitions == 2
    finally:
        torch.set_num_threads(previous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("size", (8, 10))
def test_ppo_update_graphs_follow_task_switch_and_loaded_optimizer(fresh_compiler_state, size):
    from biai.atari.algorithms import MultiHeadPPOAgent

    torch.manual_seed(51)
    eager = MultiHeadPPOAgent(4, device="cuda")
    eager.register_task("a", 2)
    eager.register_task("b", 3)
    eager.set_task("a")
    actual = copy.deepcopy(eager)
    learners = (
        PPOLearner(eager, update_epochs=1, minibatch_size=4),
        PPOLearner(actual, update_epochs=1, minibatch_size=4, compile_policy=True),
    )
    for index, task in enumerate(("a", "b", "a", "b")):
        for agent in (eager, actual):
            agent.set_task(task)
        states = torch.randint(256, (size, 4, 84, 84), dtype=torch.uint8, device="cuda")
        with torch.no_grad():
            actions, logs, values = eager.sample_action_and_value(states)
        data = dict(
            states=states,
            actions=actions,
            log_probs=logs,
            values=values.flatten(),
            rewards=torch.ones(size),
            dones=torch.zeros(size),
        )
        rollout = CollectedRollout(data, torch.zeros(1, device="cuda"), size, ())
        for learner in learners:
            torch.manual_seed(index + 72)
            learner.update(rollout)
        torch.testing.assert_close(
            actual.actors.state_dict(), eager.actors.state_dict(), rtol=1e-3, atol=1e-5
        )
        torch.testing.assert_close(
            actual.critics.state_dict(), eager.critics.state_dict(), rtol=1e-3, atol=1e-5
        )
        torch.testing.assert_close(
            actual.optimizer.state_dict(), eager.optimizer.state_dict(), rtol=2e-3, atol=2e-5
        )
        if index == 2:
            checkpoint = copy.deepcopy(eager.checkpoint_state())
            actual.load_checkpoint_state(checkpoint)
