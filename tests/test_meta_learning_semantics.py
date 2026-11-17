"""Regression checks for the meta_learning lesson."""

import hashlib
import io
import pickle
import random
from collections import OrderedDict
from zipfile import ZipFile

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from biai.reproducibility import parameter_digest
from tests.notebook_helpers import definitions


def test_omniglot_cache_preserves_pixels_sampling_and_episode_independence(tmp_path, monkeypatch):
    ns = definitions("meta_learning")
    expected = {}
    path = tmp_path / "images_background.zip"
    # ZIP member order must not change the former sorted directory traversal.
    with ZipFile(path, "w") as archive:
        for character in reversed(range(4)):
            for sample in reversed(range(3)):
                pixels = np.random.default_rng(character * 3 + sample).integers(
                    0, 256, size=(105, 105), dtype=np.uint8
                )
                name = f"images_background/alphabet/character{character}/{sample}.png"
                image = Image.fromarray(pixels)
                content = io.BytesIO()
                image.save(content, format="PNG")
                archive.writestr(name, content.getvalue())
                resized = np.array(image.resize((28, 28), Image.Resampling.BILINEAR))
                expected[name] = 1.0 - torch.from_numpy(resized).float().unsqueeze(0) / 255
    monkeypatch.setitem(
        ns["datasets"].Omniglot.zips_md5,
        "images_background",
        hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest(),
    )

    dataset = ns["OmniglotFewShot"](tmp_path)
    assert dataset.classes == [("alphabet", f"character{i}") for i in range(4)]
    assert list(dataset.images) == sorted(expected)
    assert list(tmp_path.iterdir()) == [path]
    for path, image in expected.items():
        torch.testing.assert_close(dataset._load_image(path), image, rtol=0, atol=0)

    def unexpected_read(*args, **kwargs):
        raise AssertionError("Episode sampling must not reopen cached PNG files")

    monkeypatch.setattr(Image, "open", unexpected_read)
    subset = dataset.subset_classes(dataset.classes[:2])
    reference = dataset.subset_classes(dataset.classes[:2])
    reference._load_image = expected.__getitem__
    first = subset.sample_episode(2, 1, 2, random.Random(7))
    second = reference.sample_episode(2, 1, 2, random.Random(7))
    for actual, wanted in zip(first, second, strict=True):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    # Episode tensors may be modified without changing the shared image cache.
    first[0].zero_()
    first[2].zero_()
    repeated = subset.sample_episode(2, 1, 2, random.Random(7))
    for actual, wanted in zip(repeated, second, strict=True):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


@pytest.mark.parametrize("content", [None, b"corrupted archive"])
def test_omniglot_requires_verified_zip_without_implicit_download(tmp_path, monkeypatch, content):
    ns = definitions("meta_learning")
    if content is not None:
        (tmp_path / "images_background.zip").write_bytes(content)

    def unexpected_download(*args, **kwargs):
        raise AssertionError("Offline loading must not download missing or corrupt data")

    monkeypatch.setitem(ns, "download_url", unexpected_download)
    with pytest.raises(RuntimeError, match="ZIP 缺失或校验失败"):
        ns["OmniglotFewShot"](tmp_path)


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


@pytest.mark.cuda
@pytest.mark.parametrize("first_order", [False, True])
@pytest.mark.parametrize("input_channels,img_size", [(1, 28), (3, 84)])
def test_cuda_meta_graph_matches_serial_training(
    first_order, input_channels, img_size, monkeypatch
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    ns = definitions("meta_learning")
    ns["device"] = torch.device("cuda")
    monkeypatch.setitem(ns, "plot_meta_history", lambda *args: None)
    ns["seed_everything"](0, deterministic=True)
    reference = ns["MAML_CNN"](2, input_channels, img_size).cuda()
    initial = {name: value.clone() for name, value in reference.state_dict().items()}

    class Episodes:
        def __init__(self):
            self.input_channels, self.img_size = input_channels, img_size
            self.draws = []

        def sample_episode(self, n_way, k_shot, q_query, rng):
            draw = rng.randrange(10000)
            self.draws.append(draw)
            images = torch.rand(
                6, input_channels, img_size, img_size, generator=torch.Generator().manual_seed(draw)
            )
            return images[:2], torch.tensor([0, 1]), images[2:], torch.tensor([0, 0, 1, 1])

    reference_data, validation = Episodes(), Episodes()
    rng = random.Random(1)
    optimizer = torch.optim.Adam(reference.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3, eta_min=1e-4)
    history, validation_history = [], []
    best_accuracy = -1.0
    for update in range(3):
        optimizer.zero_grad()
        losses, accuracies = [], []
        for _ in range(2):
            sx, sy, qx, qy = [x.cuda() for x in reference_data.sample_episode(2, 1, 2, rng)]
            fast = OrderedDict(reference.named_parameters())
            for _ in range(5):
                loss = F.cross_entropy(reference.functional_forward(sx, fast), sy)
                grads = torch.autograd.grad(loss, fast.values(), create_graph=not first_order)
                fast = OrderedDict((n, p - 0.1 * g) for (n, p), g in zip(fast.items(), grads))
            logits = reference.functional_forward(qx, fast)
            loss = F.cross_entropy(logits, qy)
            (loss / 2).backward()
            losses.append(loss.item())
            accuracies.append((logits.argmax(1) == qy).float().mean().item())
        optimizer.step()
        scheduler.step()
        history.append(dict(update=update + 1, loss=sum(losses) / 2, accuracy=sum(accuracies) / 2))
        scores = ns["evaluate_adaptation"](
            reference, validation, 2, 1, 2, 0.1, n_episodes=2, seed=2
        )
        validation_history.append(dict(update=update + 1, scores=scores))
        if scores[5]["accuracy"] > best_accuracy:
            best_accuracy = scores[5]["accuracy"]
            selected_update = update + 1
            selected_state = {name: value.clone() for name, value in reference.state_dict().items()}

    train_data = Episodes()
    actual = ns["train_maml"](
        first_order,
        train_data,
        Episodes(),
        initial,
        2,
        1,
        2,
        n_meta_updates=3,
        n_tasks_per_batch=2,
        validation_every=1,
        validation_episodes=2,
    )
    assert train_data.draws == reference_data.draws
    assert actual.training_history == history
    assert actual.validation_history == validation_history
    assert actual.selected_update == selected_update
    for name, value in actual.state_dict().items():
        torch.testing.assert_close(value, selected_state[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in actual.parameters())


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
    monkeypatch.setitem(ns, "MiniImageNetFewShot", lambda root, split: loader(tmp_path, split))
    with pytest.raises(ValueError, match="不随课程包分发"):
        ns["load_few_shot_data"]("mini-imagenet")
    train, validation, test = ns["load_few_shot_data"](
        "mini-imagenet", use_network_data=True, data_dir=tmp_path
    )
    assert [len(ds.classes) for ds in (train, validation, test)] == [64, 16, 20]
    sx, sy, qx, qy = train.sample_episode(5, 5, 15, random.Random(0))
    assert sx.shape == (25, 3, 84, 84) and qx.shape == (75, 3, 84, 84)
    assert sy.bincount().tolist() == [5] * 5 and qy.bincount().tolist() == [15] * 5
    train._load_images = lambda indices: torch.tensor(indices)[:, None]
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


def test_mini_imagenet_batches_match_original_transforms_without_mutating_raw_cache():
    ns = definitions("meta_learning")
    dataset = ns["MiniImageNetFewShot"].__new__(ns["MiniImageNetFewShot"])
    dataset.images = np.random.default_rng(0).integers(0, 256, size=(40, 84, 84, 3), dtype=np.uint8)
    original = dataset.images.copy()
    dataset.classes = list(range(4))
    dataset.class_to_images = {i: list(range(i * 10, (i + 1) * 10)) for i in dataset.classes}
    rng = random.Random(7)
    support, query = [], []
    for key in rng.sample(dataset.classes, 3):
        selected = rng.sample(dataset.class_to_images[key], 5)
        support.extend(selected[:2])
        query.extend(selected[2:])
    transform = ns["transforms"].Compose(
        [ns["transforms"].ToTensor(), ns["transforms"].Normalize((0.5,) * 3, (0.5,) * 3)]
    )
    sx, sy, qx, qy = dataset.sample_episode(3, 2, 3, random.Random(7))
    for actual, indices in ((sx, support), (qx, query)):
        expected = torch.stack([transform(original[i]) for i in indices])
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert actual.is_contiguous()
    assert sy.tolist() == [0, 0, 1, 1, 2, 2]
    assert qy.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    sx.zero_()
    qx.zero_()
    np.testing.assert_array_equal(dataset.images, original)


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
        ns, "load_few_shot_data", lambda name, **kwargs: tuple(Episodes(name) for _ in range(3))
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
