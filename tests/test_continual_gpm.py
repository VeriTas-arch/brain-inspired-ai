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
    def __init__(self, state_dim=4, device="cpu", **kwargs):
        super().__init__(state_dim, device=device, **kwargs)
        self.backbone = nn.Sequential(nn.Flatten(), nn.Linear(4, 512), nn.Tanh()).to(device)
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


@pytest.mark.parametrize("device,compile_ppo", (("cpu", False), ("cuda", False), ("cuda", True)))
def test_three_task_gpm_starts_fresh_accumulates_and_saves_an_evaluable_checkpoint(
    monkeypatch, tmp_path, device, compile_ppo
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    environments, agents, boundaries, projections = [], [], [], []
    build_subspaces = training.build_input_subspaces
    projection_type = training.AdamSubspaceProjection

    class RecordedPPO(TinyPPO):
        def __init__(self, **kwargs):
            super().__init__(device=device, **kwargs)
            agents.append(self)

    class RecordedProjection(projection_type):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
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
        compile_ppo=compile_ppo,
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
    for index, (spaces, actors, critics) in enumerate(boundaries, start=1):
        boundary_path = tmp_path / report["boundary_checkpoints"][index - 1]
        boundary = torch.load(boundary_path, weights_only=True)
        assert boundary["gpm"]["completed_tasks"] == list(TASKS[:index])
        assert boundary["training_stage"]["stage"] == index
        for name, expected in actors.items():
            torch.testing.assert_close(
                boundary["actors"][name.rsplit(".", 1)[0]][name.rsplit(".", 1)[1]],
                expected,
                rtol=0,
                atol=0,
            )
        for name, expected in critics.items():
            torch.testing.assert_close(
                boundary["critics"][name.rsplit(".", 1)[0]][name.rsplit(".", 1)[1]],
                expected,
                rtol=0,
                atol=0,
            )
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


@pytest.mark.parametrize("algorithm", ("ppo", "dqn"))
def test_selective_boundary_retention_matches_full_trajectory_selection(monkeypatch, algorithm):
    class IndexedEnvironment(TinyEnvironment):
        def observations(self):
            values = torch.arange(self.transitions, self.transitions + self.num_envs) % 256
            return values.to(torch.uint8)[:, None, None, None].expand(-1, 4, 1, 1).clone()

    if algorithm == "ppo":
        agent = TinyPPO()
    else:
        from algorithms import MultiHeadDQNAgent

        agent = MultiHeadDQNAgent(4, device="cpu")
        agent.backbone.network = nn.Sequential(nn.Flatten(), nn.Linear(4, 512))
        agent.target_backbone = copy.deepcopy(agent.backbone)
    agent.register_task("pong", 2)
    agent.set_task("pong")
    environment = IndexedEnvironment("pong", 2)
    monkeypatch.setattr(training, "make_vector_atari_env", lambda *args, **kwargs: environment)
    states = training.collect_gpm_states(
        agent, num_envs=2, env_backend="sync", seed=23, collection_steps=300, samples=17
    )
    indices = torch.randperm(300, generator=torch.Generator().manual_seed(23))[:17]
    torch.testing.assert_close(states[:, 0, 0, 0], (indices % 256).to(torch.uint8), rtol=0, atol=0)
    assert environment.transitions == 300


def test_stage_evaluation_reuses_only_the_current_final_periodic_result(monkeypatch, tmp_path):
    calls = []

    class Evaluation(TinyEvaluationEnvironment):
        def step(self, action):
            calls.append(self.game)
            return super().step(action)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(training, "MultiHeadPPOAgent", TinyPPO)
    monkeypatch.setattr(training, "AtariEnv", Evaluation)
    monkeypatch.setattr(
        training,
        "make_vector_atari_env",
        lambda game, num_envs, **kwargs: TinyEnvironment(game, num_envs),
    )
    training.train_continual(
        games=list(TASKS),
        algorithm="ppo",
        steps_per_game=4,
        batch_size=2,
        eval_interval=4,
        eval_episodes=2,
    )
    # Earlier tasks are reevaluated at every later boundary; the current task is evaluated once.
    assert [calls.count(game) for game in TASKS] == [6, 4, 2]
    report = json.loads(
        (tmp_path / "outputs/continual/ppo_ewcFalse/seed-0/continual_evaluation.json").read_text()
    )
    assert report["score_matrix"][-1]["scores"] == dict(zip(TASKS, (1.0, 2.0, 3.0), strict=True))


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
        ({"algorithm": "dqn", "num_envs": 0}, "num_envs must be positive"),
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


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_dqn_gpm_projects_before_target_sync_and_preserves_old_heads(monkeypatch, tmp_path, device):
    from algorithms import MultiHeadDQNAgent

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    agents, environments, boundaries, projections = [], [], [], []
    build_subspaces = training.build_input_subspaces
    projection_type = training.AdamSubspaceProjection
    collect_states = training.collect_gpm_states

    class TinyDQN(MultiHeadDQNAgent):
        def __init__(self, state_dim=4, **kwargs):
            super().__init__(state_dim, device=device, **kwargs)
            self.backbone.network = nn.Sequential(nn.Flatten(), nn.Linear(4, 512)).to(device)
            self.target_backbone = copy.deepcopy(self.backbone)
            self.target_update_freq = 1
            agents.append(self)

        def update(self, batch, **kwargs):
            # Each new task uses a fresh replay buffer.
            assert torch.all(batch["states"][:, TASKS.index(self.current_task)] == 255)
            metrics = super().update(batch, **kwargs)
            gradients = torch.cat(
                [parameter.grad.flatten() for parameter in self.backbone.parameters()]
            )
            assert gradients.norm() <= 1.000001
            for name, value in self.backbone.state_dict().items():
                torch.testing.assert_close(
                    self.target_backbone.state_dict()[name], value, rtol=0, atol=0
                )
            return metrics

    class Environment(TinyEvaluationEnvironment):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            environments.append(self)

        def step(self, action):
            self.transitions += 1
            return self.reset(), 1.0, False, True

    class Projection(projection_type):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            projections.append(self)

    def boundary(backbone, states, **kwargs):
        # Input to the first affine layer must be normalized exactly once.
        inputs = []
        handle = backbone[1].register_forward_pre_hook(
            lambda module, args: inputs.append(args[0].clone())
        )
        try:
            result = build_subspaces(backbone, states, **kwargs)
        finally:
            handle.remove()
        assert all(float(x.max()) == 1.0 for x in inputs)
        boundaries.append((result, copy.deepcopy(agents[0].heads.state_dict())))
        return result

    def collect(agent, **kwargs):
        rng = torch.get_rng_state().clone()
        epsilons = dict(agent.task_epsilons)
        updates = agent.update_count
        states = collect_states(agent, **kwargs)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        assert agent.task_epsilons == epsilons and agent.update_count == updates
        return states

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(training, "MultiHeadDQNAgent", TinyDQN)
    monkeypatch.setattr(training, "AtariEnv", Environment)
    monkeypatch.setattr(training, "DEFAULT_DQN_LEARNING_STARTS", 0)
    monkeypatch.setattr(training, "AdamSubspaceProjection", Projection)
    monkeypatch.setattr(training, "build_input_subspaces", boundary)
    monkeypatch.setattr(training, "collect_gpm_states", collect)
    training.train_continual(
        games=list(TASKS),
        algorithm="dqn",
        method="gpm",
        task_steps=[12, 12, 12],
        batch_size=2,
        eval_episodes=1,
        gpm_collection_steps=8,
        gpm_samples=4,
        seed=7,
    )
    assert len(agents) == 1
    assert agents[0].update_count == 6
    assert [p.steps for p in projections] == [2, 2]
    assert all(env.closed for env in environments)
    assert [env.transitions for env in environments if env.training] == [12, 12, 12]
    assert not agents[0].optimizer._optimizer_step_post_hooks
    for index, (subspaces, heads) in enumerate(boundaries):
        basis = subspaces["1"]["basis"]
        assert basis.shape[1] == index + 1
        if index:
            previous = boundaries[index - 1][0]["1"]["basis"]
            torch.testing.assert_close(basis @ (basis.T @ previous), previous, atol=1e-6, rtol=0)
        for name, expected in heads.items():
            torch.testing.assert_close(agents[0].heads.state_dict()[name], expected, rtol=0, atol=0)
    for projection in projections:
        metrics = projection.metrics()["layers"]["1"]
        assert metrics["raw_update_energy"] > 0
        assert metrics["projection_relative_error"] < 1e-3
    path = tmp_path / "checkpoints/continual/dqn_gpm/seed-7.pt"
    restored = TinyDQN()
    restored.load(str(path))
    assert restored.update_count == 6
    assert torch.load(path, weights_only=True)["gpm"]["completed_tasks"] == list(TASKS)
    for task in TASKS:
        agents[0].set_task(task)
        restored.set_task(task)
        state = TinyEvaluationEnvironment(task).reset()
        assert restored.select_action(state, deterministic=True) == agents[0].select_action(
            state, deterministic=True
        )
    report = json.loads(
        (tmp_path / "outputs/continual/dqn_gpm/seed-7/continual_evaluation.json").read_text()
    )
    assert report["gpm_diagnostics"]["pong"]["boundary_policy"] == "greedy"
    assert len(report["score_matrix"]) == 3


@pytest.mark.parametrize("method", ("finetune", "ewc", "gpm"))
def test_periodic_task_evaluation_saves_loadable_models_without_changing_budget(
    monkeypatch, tmp_path, method
):
    environments = []

    def make_env(game, num_envs, **kwargs):
        env = TinyEnvironment(game, num_envs, **kwargs)
        environments.append(env)
        return env

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(training, "MultiHeadPPOAgent", TinyPPO)
    monkeypatch.setattr(training, "make_vector_atari_env", make_env)
    monkeypatch.setattr(training, "AtariEnv", TinyEvaluationEnvironment)
    training.train_continual(
        games=["pong"],
        algorithm="ppo",
        method=method,
        steps_per_game=8,
        num_envs=2,
        batch_size=2,
        eval_interval=4,
        eval_episodes=2,
        gpm_samples=2,
        gpm_collection_steps=4,
    )
    name = "ppo_gpm" if method == "gpm" else f"ppo_ewc{method == 'ewc'}"
    data = json.loads(
        (tmp_path / f"outputs/continual/{name}/seed-0/pong/learning_evaluation.json").read_text()
    )
    # A periodic request does not split the eight-transition rollout.
    assert [e["step"] for e in data["evaluations"]] == [8]
    assert data["evaluations"][0]["rewards"] == [1.0, 1.0]
    checkpoint = torch.load(tmp_path / data["evaluations"][0]["checkpoint"], weights_only=True)
    restored = TinyPPO()
    restored.load_checkpoint_state(checkpoint.get("agent", checkpoint))
    restored.set_task("pong")
    assert restored.select_action(
        TinyEvaluationEnvironment("pong").reset(), deterministic=True
    ) in (0, 1)
    assert environments[0].transitions == 8 and environments[0].closed


def test_new_heads_are_matched_across_methods_and_stage_rewards_are_raw(monkeypatch, tmp_path):
    monkeypatch.setattr(training, "MultiHeadPPOAgent", TinyPPO)
    monkeypatch.setattr(training, "make_vector_atari_env", TinyEnvironment)
    monkeypatch.setattr(training, "AtariEnv", TinyEvaluationEnvironment)
    initial_heads = {}
    for method in ("finetune", "ewc", "gpm"):
        root = tmp_path / method
        root.mkdir()
        monkeypatch.chdir(root)
        training.train_continual(
            games=list(TASKS),
            algorithm="ppo",
            method=method,
            task_steps=[4, 4, 4],
            num_envs=2,
            batch_size=2,
            eval_episodes=2,
            gpm_samples=2,
            gpm_collection_steps=4,
        )
        slug = "ppo_gpm" if method == "gpm" else f"ppo_ewc{method == 'ewc'}"
        report = json.loads(
            (root / f"outputs/continual/{slug}/seed-0/continual_evaluation.json").read_text()
        )
        assert report["stage_episode_rewards"][-1]["rewards"] == {
            task: [float(i + 1)] * 2 for i, task in enumerate(TASKS)
        }
        initial_heads[method] = []
        for stage, task in enumerate(TASKS[1:], start=2):
            state = torch.load(
                root / f"checkpoints/continual/{slug}/seed-0/stage-{stage:02d}-initial.pt",
                weights_only=True,
            )
            state = state.get("agent", state)
            initial_heads[method].append((state["actors"][task], state["critics"][task]))
    for method in ("ewc", "gpm"):
        for reference, actual in zip(initial_heads["finetune"], initial_heads[method], strict=True):
            for ref_head, head in zip(reference, actual, strict=True):
                torch.testing.assert_close(ref_head, head, rtol=0, atol=0)
