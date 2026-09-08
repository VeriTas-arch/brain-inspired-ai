"""Exercise the complete teaching protocol with real PPO/GPM and tiny environments."""

import copy
import json

import pytest
import torch
from torch import nn

from algorithms import MultiHeadPPOAgent
from environments import VectorStep
from scripts import train_continual as training

TASKS = ("pong", "breakout", "space_invaders")


class TinyPPO(MultiHeadPPOAgent):
    def __init__(self, state_dim=4, **kwargs):
        super().__init__(state_dim, device="cpu", **kwargs)
        self.backbone = nn.Sequential(nn.Flatten(), nn.Linear(4, 512), nn.Tanh())
        self.network = self.backbone


class TinyEnvironment:
    def __init__(self, game, num_envs=1, *, training=True, **kwargs):
        self.game = game
        self.num_envs = num_envs
        self.training = training
        self.action_space = 2
        self.transitions = 0
        self.closed = False

    def observations(self):
        states = torch.zeros(self.num_envs, 4, 1, 1, dtype=torch.uint8)
        states[:, TASKS.index(self.game)] = 255
        return states

    def reset(self):
        return self.observations()

    def step_and_reset(self, actions):
        self.transitions += self.num_envs
        return VectorStep(
            observations=self.observations(),
            transition_observations=self.observations(),
            rewards=torch.ones(self.num_envs),
            terminated=torch.ones(self.num_envs, dtype=torch.bool),
            truncated=torch.zeros(self.num_envs, dtype=torch.bool),
        )

    def close(self):
        self.closed = True


class TinyEvaluationEnvironment(TinyEnvironment):
    def reset(self):
        return self.observations()[0]

    def step(self, action):
        return self.reset(), float(TASKS.index(self.game) + 1), True, False


def test_three_task_gpm_starts_fresh_accumulates_and_saves_an_evaluable_checkpoint(
    monkeypatch, tmp_path
):
    environments, agents, boundaries, projections = [], [], [], []
    build_subspaces = training.build_input_subspaces
    projection_type = training.AdamSubspaceProjection

    class RecordedPPO(TinyPPO):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            agents.append(self)

    class RecordedProjection(projection_type):
        def __init__(self, *args):
            super().__init__(*args)
            projections.append(self)

    def make_environment(game, num_envs, **kwargs):
        environment = TinyEnvironment(game, num_envs, **kwargs)
        environments.append(environment)
        return environment

    def record_boundary(*args, **kwargs):
        result = build_subspaces(*args, **kwargs)
        agent = agents[0]
        boundaries.append(
            (
                result,
                copy.deepcopy(agent.actors.state_dict()),
                copy.deepcopy(agent.critics.state_dict()),
            )
        )
        return result

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(training, "MultiHeadPPOAgent", RecordedPPO)
    monkeypatch.setattr(training, "make_vector_atari_env", make_environment)
    monkeypatch.setattr(training, "AtariEnv", TinyEvaluationEnvironment)
    monkeypatch.setattr(training, "build_input_subspaces", record_boundary)
    monkeypatch.setattr(training, "AdamSubspaceProjection", RecordedProjection)

    training.train_continual(
        games=list(TASKS),
        algorithm="ppo",
        method="gpm",
        task_steps=[4, 6, 4],
        num_envs=2,
        batch_size=2,
        eval_episodes=1,
        gpm_collection_steps=8,
        gpm_samples=4,
        seed=7,
    )

    assert len(agents) == 1
    assert [env.transitions for env in environments if env.training] == [4, 6, 4]
    assert [env.transitions for env in environments if not env.training] == [8, 8, 8]
    assert all(env.closed for env in environments)
    # Stage one has no old subspace. Later hooks run once per actual PPO minibatch.
    assert [projection.steps for projection in projections] == [12, 8]
    assert not agents[0].optimizer._optimizer_step_pre_hooks
    assert not agents[0].optimizer._optimizer_step_post_hooks
    for index, (subspaces, actors, critics) in enumerate(boundaries):
        basis = subspaces["1"]["basis"]
        assert basis.shape[1] == index + 1
        torch.testing.assert_close(basis.T @ basis, torch.eye(index + 1), atol=1e-6, rtol=0)
        if index:
            previous = boundaries[index - 1][0]["1"]["basis"]
            torch.testing.assert_close(basis @ (basis.T @ previous), previous, atol=1e-6, rtol=0)
        for current, final in (
            (actors, agents[0].actors.state_dict()),
            (critics, agents[0].critics.state_dict()),
        ):
            for name, expected in current.items():
                torch.testing.assert_close(final[name], expected, rtol=0, atol=0)

    report = json.loads(
        (tmp_path / "outputs/continual/ppo_gpm/seed-7/continual_evaluation.json").read_text()
    )
    assert report["method"] == "gpm" and report["initialization"] == "random"
    assert report["task_steps"] == dict(zip(TASKS, (4, 6, 4), strict=True))
    assert report["score_matrix"][0]["scores"] == {
        "pong": 1.0,
        "breakout": None,
        "space_invaders": None,
    }
    assert report["score_matrix"][-1]["scores"] == dict(zip(TASKS, (1.0, 2.0, 3.0), strict=True))
    assert report["gpm_diagnostics"]["pong"]["projection"] is None
    assert report["gpm_diagnostics"]["breakout"]["projection"]["optimizer_steps"] == 12

    path = tmp_path / "checkpoints/continual/ppo_gpm/seed-7.pt"
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["gpm"]["completed_tasks"] == list(TASKS)
    assert checkpoint["gpm"]["subspaces"]["1"]["rank"] == 3
    restored = TinyPPO()
    restored.load(str(path))
    for task in TASKS:
        agents[0].set_task(task)
        restored.set_task(task)
        observation = TinyEvaluationEnvironment(task).reset()
        assert restored.select_action(observation, deterministic=True) == agents[0].select_action(
            observation, deterministic=True
        )


def test_boundary_sampling_preserves_weights_rng_and_closes_environment(monkeypatch):
    agent = TinyPPO()
    agent.register_task("pong", 2)
    agent.set_task("pong")
    environment = TinyEnvironment("pong", 2, training=False)
    monkeypatch.setattr(training, "make_vector_atari_env", lambda *args, **kwargs: environment)
    weights = copy.deepcopy(agent.checkpoint_state())
    rng = torch.get_rng_state().clone()
    states = training.collect_gpm_states(
        agent, num_envs=2, env_backend="sync", seed=23, collection_steps=8, samples=3
    )
    assert states.shape == (3, 4, 1, 1)
    assert states.dtype == torch.uint8
    assert environment.closed and environment.transitions == 8
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert not agent.optimizer.state
    for name, value in agent.backbone.state_dict().items():
        torch.testing.assert_close(value, weights["backbone"][name], rtol=0, atol=0)


def test_failed_new_task_update_removes_projection_hooks_and_closes_environment(
    monkeypatch, tmp_path
):
    environments, failed_agents = [], []
    update = training.PPOLearner.update

    def fail_on_new_task(learner, rollout):
        if learner.agent.current_task == "breakout":
            failed_agents.append(learner.agent)
            raise RuntimeError("interrupted update")
        return update(learner, rollout)

    def make_environment(game, num_envs, **kwargs):
        environment = TinyEnvironment(game, num_envs, **kwargs)
        environments.append(environment)
        return environment

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(training, "MultiHeadPPOAgent", TinyPPO)
    monkeypatch.setattr(training, "AtariEnv", TinyEvaluationEnvironment)
    monkeypatch.setattr(training, "make_vector_atari_env", make_environment)
    monkeypatch.setattr(training.PPOLearner, "update", fail_on_new_task)
    with pytest.raises(RuntimeError, match="interrupted update"):
        training.train_continual(
            games=list(TASKS),
            algorithm="ppo",
            method="gpm",
            steps_per_game=4,
            num_envs=2,
            batch_size=2,
            eval_episodes=1,
            gpm_collection_steps=4,
            gpm_samples=2,
        )
    assert len(failed_agents) == 1
    assert not failed_agents[0].optimizer._optimizer_step_pre_hooks
    assert not failed_agents[0].optimizer._optimizer_step_post_hooks
    assert all(environment.closed for environment in environments)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"algorithm": "dqn"}, "only for PPO"),
        ({"use_ewc": True}, "another method"),
        ({"task_steps": [2]}, "one positive"),
        ({"task_steps": [2, 0, 2]}, "one positive"),
        ({"task_steps": [2, 3, 2], "num_envs": 2}, "divisible"),
        ({"gpm_threshold": 0}, "gpm_threshold"),
        ({"gpm_samples": 9, "gpm_collection_steps": 8}, "gpm_samples"),
        ({"gpm_samples": 1, "gpm_collection_steps": 3, "num_envs": 2}, "divisible"),
    ],
)
def test_invalid_gpm_options_fail_before_environment_creation(options, message):
    options = {"algorithm": "ppo", "method": "gpm", **options}
    with pytest.raises(ValueError, match=message):
        training.train_continual(**options)
