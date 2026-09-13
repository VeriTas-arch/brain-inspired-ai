"""Regression checks for the meta_learning lesson."""

import pickle
import random
from collections import OrderedDict

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from biai.reproducibility import parameter_digest
from tests.notebook_helpers import definitions


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
            n_meta_updates=2,
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
