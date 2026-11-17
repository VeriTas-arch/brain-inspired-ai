"""Check cached MNIST pixels and archive-backed CIFAR-10 dataset views."""

import json

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import datasets, transforms

from tests.notebook_helpers import definitions, notebook_path, prepare_offline_data


def data_cell(topic, text):
    cells = json.loads(notebook_path(topic).read_text())["cells"]
    return next(
        "".join(cell["source"])
        for cell in cells
        if cell["cell_type"] == "code" and text in "".join(cell["source"])
    )


@pytest.mark.parametrize("topic", ["mlp", "cnn", "continual_mnist"])
def test_mnist_cache_matches_per_image_preprocessing(topic, tmp_path, monkeypatch):
    root = prepare_offline_data(monkeypatch, tmp_path / "data")
    ns = definitions(topic)
    ns.update(DATASET_DIR=root, USE_NETWORK_DATA=False)
    source = datasets.MNIST(root, train=True)
    transform = transforms.ToTensor()
    if topic != "mlp":
        transform = transforms.Compose([transform, transforms.Normalize((0.1307,), (0.3081,))])
    expected = torch.stack([transform(Image.fromarray(image.numpy())) for image in source.data])
    state = torch.get_rng_state()
    if topic == "continual_mnist":
        train, _ = ns["load_split_mnist"]([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], root)
        cached = train[0][0].dataset
        assert all(task.dataset is cached for task, _ in train)
        for task, classes in train:
            assert task.indices == [i for i, label in enumerate(source.targets) if label in classes]
    else:
        exec(data_cell(topic, "train_dataset = datasets.MNIST("), ns)
        cached = ns["train_dataset"].dataset
        assert ns["val_dataset"].dataset is cached
    assert torch.equal(state, torch.get_rng_state())
    torch.testing.assert_close(cached.tensors[0], expected, atol=0, rtol=0)
    torch.testing.assert_close(cached.tensors[1], source.targets, atol=0, rtol=0)
    assert cached.tensors[0].device.type == "cpu"

    def unexpected_conversion(*args, **kwargs):
        raise AssertionError("Cached MNIST samples must not be converted through PIL again")

    monkeypatch.setattr(Image, "fromarray", unexpected_conversion)
    torch.testing.assert_close(cached[0][0], expected[0], atol=0, rtol=0)
    if topic == "mlp":
        train_indices = ns["train_dataset"].indices
        test_pixels = ns["test_dataset"].tensors[0]
        exec(data_cell(topic, "images, labels = train_dataset.dataset.tensors"), ns)
        flattened = ns["train_dataset"].dataset.tensors[0]
        assert flattened.data_ptr() == cached.tensors[0].data_ptr()
        assert ns["test_dataset"].tensors[0].data_ptr() == test_pixels.data_ptr()
        assert ns["train_dataset"].indices == train_indices
        torch.testing.assert_close(flattened, expected.flatten(1), atol=0, rtol=0)


@pytest.mark.parametrize("network", [False, True])
def test_cifar_archive_pixels_order_and_shared_views(tmp_path, monkeypatch, network):
    root = prepare_offline_data(monkeypatch, tmp_path / "data")
    data_dir = root / "network" if network else root
    ns = definitions("cnn")
    ns.update(DATASET_DIR=data_dir, USE_NETWORK_DATA=network)
    exec(data_cell("cnn", "def load_cifar10("), ns)
    train = ns["train_dataset_cifar"].dataset
    validation = ns["validation_source_cifar"]
    test = ns["test_dataset_cifar"]
    assert train.images is validation.images
    assert train.labels is validation.labels
    for dataset, seed in ((train, 0), (test, 1)):
        raw = np.random.default_rng(seed).integers(0, 256, size=(40, 3072), dtype=np.uint8)
        np.testing.assert_array_equal(
            dataset.images, raw.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        )
        assert dataset.labels == np.arange(10).repeat(4).tolist()
    assert not (data_dir / "cifar-10-batches-py").exists()
    assert not (data_dir / "cifar-10-python.tar.xz").exists()

    pixels = train.images.copy()
    torch.manual_seed(0)
    expected = ns["transform_cifar_train"](Image.fromarray(pixels[0]))
    torch.manual_seed(0)
    first = train[0][0]
    torch.testing.assert_close(first, expected, atol=0, rtol=0)
    torch.manual_seed(1)
    second = train[0][0]
    assert not torch.equal(first, second)
    state = torch.get_rng_state()
    torch.testing.assert_close(validation[0][0], validation[0][0], atol=0, rtol=0)
    assert torch.equal(state, torch.get_rng_state())
    np.testing.assert_array_equal(train.images, pixels)


@pytest.mark.parametrize("content", [None, b"corrupted archive"])
def test_cifar_rejects_invalid_archive_even_with_an_extracted_directory(
    tmp_path, monkeypatch, content
):
    ns = definitions("cnn")
    (tmp_path / "cifar-10-batches-py").mkdir()
    if content is not None:
        (tmp_path / "cifar-10-python.tar.gz").write_bytes(content)

    def unexpected_download(*args, **kwargs):
        raise AssertionError("Offline loading must not download data")

    monkeypatch.setitem(ns, "download_url", unexpected_download)
    with pytest.raises(RuntimeError, match="压缩包缺失或校验失败"):
        ns["load_cifar10"](tmp_path)
