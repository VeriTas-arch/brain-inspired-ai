"""Regression checks for the mathematical contracts taught in the notebooks."""

import ast
import json
import pickle
import random
from collections import OrderedDict

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from biai.paths import PROJECT_ROOT
from biai.reproducibility import parameter_digest


def definitions(topic):
    path = PROJECT_ROOT / "biai" / topic / f"{topic}.ipynb"
    namespace = {"SEED": 0, "device": torch.device("cpu"), "CONFIG": {"device": "cpu"}}
    for cell in json.loads(path.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        tree.body = [
            node
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))
        ]
        exec(compile(tree, str(path), "exec"), namespace)
    return namespace


def test_manual_update_matches_cross_entropy_autograd():
    ns = definitions("mlp")
    generator = torch.Generator().manual_seed(4)
    w1 = torch.randn(7, 9, generator=generator, dtype=torch.float64)
    w2 = torch.randn(3, 7, generator=generator, dtype=torch.float64)
    x = torch.randn(5, 9, generator=generator, dtype=torch.float64)
    target = torch.tensor([0, 1, 2, 1, 0])
    a, b = w1.clone().requires_grad_(), w2.clone().requires_grad_()
    h = F.relu(x @ a.T)
    logits = h @ b.T
    grad1, grad2 = torch.autograd.grad(F.cross_entropy(logits, target), (a, b))
    ns.update(W1=w1.clone(), W2=w2.clone(), eta=0.1, output_dim=3)
    ns["manual_backprop_update"](x, h.detach(), logits.detach(), target)
    torch.testing.assert_close(ns["W1"], w1 - 0.1 * grad1)
    torch.testing.assert_close(ns["W2"], w2 - 0.1 * grad2)


def test_manual_initialization_avoids_saturated_random_predictions():
    ns = definitions("mlp")
    ns.update(input_dim=784, hidden_dim=128, output_dim=10)
    path = PROJECT_ROOT / "biai/mlp/mlp.ipynb"
    cell = json.loads(path.read_text())["cells"][14]
    exec("".join(cell["source"]), ns)
    x = torch.rand(64, 784, generator=torch.Generator().manual_seed(1))
    _, logits = ns["forward"](x)
    assert F.cross_entropy(logits, torch.arange(64) % 10) < 5
    assert logits.softmax(1).max(1).values.mean() < 0.5


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


@pytest.mark.parametrize("first_order", [False, True])
@pytest.mark.parametrize("inner_steps", [1, 5])
def test_meta_gradient_and_functional_forward(first_order, inner_steps):
    ns = definitions("meta_learning")
    torch.manual_seed(0)
    # Use double precision for the independently accumulated Hessian products.
    model = ns["MAML_CNN"](n_way=2).double()
    x = torch.rand(4, 1, 28, 28, dtype=torch.float64)
    labels = torch.tensor([0, 1, 0, 1])
    params = OrderedDict(model.named_parameters())
    torch.testing.assert_close(model(x), model.functional_forward(x, params))
    fast, support_steps = params, []
    for _ in range(inner_steps):
        loss = F.cross_entropy(model.functional_forward(x[:2], fast), labels[:2])
        grads = torch.autograd.grad(loss, fast.values(), create_graph=not first_order)
        support_steps.append((fast, grads))
        fast = OrderedDict((name, p - 0.1 * g) for (name, p), g in zip(fast.items(), grads))
    query_loss = F.cross_entropy(model.functional_forward(x[2:], fast), labels[2:])
    query_grads = torch.autograd.grad(query_loss, fast.values(), retain_graph=True)
    expected = query_grads
    if not first_order:
        for step_params, step_grads in reversed(support_steps):
            hessian_vector = torch.autograd.grad(
                step_grads,
                step_params.values(),
                grad_outputs=tuple(g.detach() for g in expected),
                retain_graph=True,
            )
            expected = tuple(g - 0.1 * hv for g, hv in zip(expected, hessian_vector))
    query_loss.backward()
    for parameter, gradient in zip(model.parameters(), expected):
        torch.testing.assert_close(parameter.grad, gradient)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.conv1.weight.grad.norm() > 0


def test_adaptation_is_repeatable_and_does_not_modify_model_or_training_rng():
    ns = definitions("meta_learning")
    torch.manual_seed(0)
    model = ns["MAML_CNN"](n_way=2)
    model.train()
    digest = parameter_digest(model)
    training_rng = random.Random(7)
    state = training_rng.getstate()

    class Episodes:
        def sample_episode(self, n_way, k_shot, q_query, rng):
            generator = torch.Generator().manual_seed(rng.randrange(10000))
            images = torch.rand(4, 1, 28, 28, generator=generator)
            labels = torch.tensor([0, 1])
            return images[:2], labels, images[2:], labels

    kwargs = dict(n_way=2, k_shot=1, q_query=1, inner_lr=0.1, steps=(0, 1), n_episodes=2)
    first = ns["evaluate_adaptation"](model, Episodes(), **kwargs)
    second = ns["evaluate_adaptation"](model, Episodes(), **kwargs)
    assert first == second
    assert parameter_digest(model) == digest
    assert model.training
    assert training_rng.getstate() == state


def test_meta_normalization_is_per_image_and_uses_adapted_parameters():
    ns = definitions("meta_learning")
    torch.manual_seed(0)
    model = ns["MAML_CNN"](n_way=2)
    images = torch.rand(4, 1, 28, 28)
    fast = OrderedDict((name, p.detach().clone()) for name, p in model.named_parameters())
    for name, value in fast.items():
        if name.startswith("norm"):
            value.add_(torch.linspace(-0.2, 0.2, value.numel()).reshape_as(value))
    reference = ns["MAML_CNN"](n_way=2)
    reference.load_state_dict(fast)
    expected = reference(images)
    assert not torch.allclose(expected, model(images))
    for training in (True, False):
        model.train(training)
        actual = model.functional_forward(images, fast)
        torch.testing.assert_close(actual, expected)
        individual = torch.cat([model.functional_forward(image[None], fast) for image in images])
        torch.testing.assert_close(individual, actual, atol=1e-6, rtol=1e-5)
    assert not list(model.named_buffers())


def test_meta_methods_share_initialization_and_training_episodes(monkeypatch):
    ns = definitions("meta_learning")
    monkeypatch.setattr(ns["plt"], "show", lambda: ns["plt"].close("all"))
    initial_state = ns["MAML_CNN"](n_way=2).state_dict()

    class Episodes:
        input_channels = 1
        img_size = 28

        def __init__(self):
            self.draws = []

        def sample_episode(self, n_way, k_shot, q_query, rng):
            draw = rng.randrange(10000)
            self.draws.append(draw)
            images = torch.rand(4, 1, 28, 28, generator=torch.Generator().manual_seed(draw))
            labels = torch.tensor([0, 1])
            return images[:2], labels, images[2:], labels

    runs = []
    for first_order in (False, True):
        train, validation = Episodes(), Episodes()
        model = ns["train_maml"](
            first_order,
            train,
            validation,
            initial_state,
            2,
            1,
            1,
            n_meta_epochs=2,
            n_tasks_per_batch=1,
            validation_every=1,
            validation_steps=1,
            validation_episodes=2,
        )
        runs.append((model, train.draws, validation.draws))
    assert runs[0][0].initial_digest == runs[1][0].initial_digest
    assert runs[0][1] == runs[1][1]
    assert runs[0][2] == runs[1][2]
    assert runs[0][2][:2] == runs[0][2][2:]


def test_oja_update_uses_local_activity_rule():
    ns = definitions("mlp")
    layer = torch.nn.Linear(3, 2, bias=False).double()
    x = torch.tensor([[0.2, 0.4, 0.7], [0.8, 0.1, 0.5]], dtype=torch.float64)
    weights = layer.weight.detach().clone()
    updates = []
    for image in x:
        response = weights @ image
        updates.append(response[:, None] * image - response[:, None].square() * weights)
    ns["oja_update"](layer, x, 0.01)
    torch.testing.assert_close(layer.weight, weights + 0.01 * torch.stack(updates).mean(0))
    assert layer.weight.grad is None


def test_generalized_hebbian_update_removes_preceding_activity_directions():
    ns = definitions("mlp")
    layer = torch.nn.Linear(3, 2, bias=False).double()
    x = torch.tensor([[0.2, 0.4, 0.7], [0.8, 0.1, 0.5]], dtype=torch.float64)
    weights = layer.weight.detach().clone()
    expected = weights.clone()
    for image in x:
        response = weights @ image
        for unit in range(len(weights)):
            reconstruction = sum(response[k] * weights[k] for k in range(unit + 1))
            expected[unit] += 0.01 * response[unit] * (image - reconstruction) / len(x)
    ns["generalized_hebbian_update"](layer, x, 0.01)
    torch.testing.assert_close(layer.weight, expected)
    assert layer.weight.grad is None


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


@pytest.mark.parametrize("first_order", [False, True])
def test_rgb_meta_model_can_adapt_and_backpropagate(first_order):
    ns = definitions("meta_learning")
    model = ns["MAML_CNN"](2, input_channels=3, img_size=84)
    x = torch.rand(4, 3, 84, 84)
    labels = torch.tensor([0, 1])
    params = OrderedDict(model.named_parameters())
    torch.testing.assert_close(model(x), model.functional_forward(x, params))
    loss = F.cross_entropy(model.functional_forward(x[:2], params), labels)
    gradients = torch.autograd.grad(loss, params.values(), create_graph=not first_order)
    fast = OrderedDict((name, p - 0.1 * g) for (name, p), g in zip(params.items(), gradients))
    F.cross_entropy(model.functional_forward(x[2:], fast), labels).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.conv1.weight.grad.norm() > 0


def test_mini_imagenet_split_and_episode_loader(tmp_path, monkeypatch):
    ns = definitions("meta_learning")
    loader = ns["MiniImageNetFewShot"]
    for split, count in (("train", 64), ("validation", 16), ("test", 20)):
        path = tmp_path / f"mini-imagenet-cache-{split}.pkl"
        images = np.zeros((count * 20, 84, 84, 3), dtype=np.uint8)
        classes = {f"{split}-{i}": list(range(i * 20, (i + 1) * 20)) for i in range(count)}
        path.write_bytes(pickle.dumps(dict(image_data=images, class_dict=classes), protocol=4))
    monkeypatch.setitem(ns, "check_integrity", lambda path, checksum: path.is_file())
    monkeypatch.setitem(ns, "MiniImageNetFewShot", lambda split: loader(tmp_path, split))
    train, validation, test = ns["load_few_shot_data"]("mini-imagenet")
    assert [len(ds.classes) for ds in (train, validation, test)] == [64, 16, 20]
    sx, sy, qx, qy = train.sample_episode(5, 5, 15, random.Random(0))
    assert sx.shape == (25, 3, 84, 84) and qx.shape == (75, 3, 84, 84)
    assert sy.bincount().tolist() == [5] * 5 and qy.bincount().tolist() == [15] * 5
    train._load_image = lambda index: torch.tensor([index])
    sx, _, qx, _ = train.sample_episode(5, 5, 15, random.Random(0))
    assert set(sx.flatten().tolist()).isdisjoint(qx.flatten().tolist())
    with pytest.raises(ValueError, match="无法提供"):
        train.sample_episode(5, 5, 16)


def test_mini_imagenet_cache_rejects_python_objects(tmp_path):
    ns = definitions("meta_learning")
    path = tmp_path / "bad.pkl"
    path.write_bytes(pickle.dumps(eval))
    with path.open("rb") as stream, pytest.raises(pickle.UnpicklingError):
        ns["MiniImageNetUnpickler"](stream).load()


def test_task_size_matrix_runs_both_methods_with_matching_initialization(monkeypatch):
    ns = definitions("meta_learning")
    monkeypatch.setattr(ns["plt"], "show", lambda: ns["plt"].close("all"))

    class Episodes:
        def __init__(self, name):
            self.input_channels = 3 if name == "mini-imagenet" else 1
            self.img_size = 84 if self.input_channels == 3 else 28

        def sample_episode(self, n_way, k_shot, q_query, rng):
            generator = torch.Generator().manual_seed(rng.randrange(10000))
            shape = (self.input_channels, self.img_size, self.img_size)
            support = torch.rand(n_way * k_shot, *shape, generator=generator)
            query = torch.rand(n_way * q_query, *shape, generator=generator)
            return (
                support,
                torch.arange(n_way).repeat_interleave(k_shot),
                query,
                torch.arange(n_way).repeat_interleave(q_query),
            )

    monkeypatch.setitem(
        ns, "load_few_shot_data", lambda name: tuple(Episodes(name) for _ in range(3))
    )
    rows = ns["compare_task_sizes"](
        [("omniglot", 20, 5), ("mini-imagenet", 5, 5)],
        updates=1,
        tasks_per_batch=1,
        validation_episodes=1,
        test_episodes=1,
        q_query=1,
    )
    assert len(rows) == 4
    for first, second in zip(rows[::2], rows[1::2]):
        assert (first["method"], second["method"]) == ("MAML", "FO-MAML")
        assert first["initial_digest"] == second["initial_digest"]
        assert first["selected_update"] == second["selected_update"] == 1
        assert set(first["scores"]) == set(second["scores"]) == {0, 1, 5}


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
