"""Shared notebook loading, offline execution, and lesson assertions for tests."""

import ast
import hashlib
import io
import json
import pickle
import tarfile
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import datasets
from torchvision.datasets import utils as dataset_utils

from biai import paths
from biai.paths import PROJECT_ROOT


def notebook_path(topic):
    return paths.PROJECT_ROOT / "biai" / topic / f"{topic}.ipynb"


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


class SmallBudgets(ast.NodeTransformer):
    """Change only demonstration budgets in the in-memory smoke copy."""

    def __init__(self, use_network_data=None):
        self.use_network_data = use_network_data
        super().__init__()

    values = {
        "epochs": 1,
        "N_WAY": 2,
        "Q_QUERY": 1,
        "N_TASKS_PER_BATCH": 1,
        "N_TEST_EPISODES": 2,
        "N_META_UPDATES_MAML": 1,
        "N_META_UPDATES_FOMAML": 1,
        "CIFAR_EPOCHS": 1,
        "MNIST_EPOCHS": 1,
        "N_VALIDATION_EPISODES": 2,
    }

    def visit_Assign(self, node):
        self.generic_visit(node)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == "device":
                node.value = ast.Name(id="__smoke_device", ctx=ast.Load())
            elif name == "USE_NETWORK_DATA" and self.use_network_data is not None:
                node.value = ast.Constant(self.use_network_data)
            elif name in self.values:
                node.value = ast.Constant(self.values[name])
            elif name == "CONFIG":
                for index, key in enumerate(node.value.keys):
                    if key.value == "epochs_per_task":
                        node.value.values[index] = ast.Constant(1)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "range"
            and len(node.args) == 2
            and all(isinstance(arg, ast.Constant) for arg in node.args)
            and [arg.value for arg in node.args] == [1, 6]
        ):
            node.args[1] = ast.Constant(2)
        return node


def prepare_offline_data(monkeypatch, data_dir):
    """Install deterministic local datasets for a notebook smoke run."""
    monkeypatch.setattr(paths, "DATA_DIR", data_dir)
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    data_dir.mkdir(parents=True, exist_ok=True)
    cifar_path = data_dir / datasets.CIFAR10.filename
    with tarfile.open(cifar_path, "w:gz") as archive:
        for train, batch_list in (
            (True, datasets.CIFAR10.train_list),
            (False, datasets.CIFAR10.test_list),
        ):
            images = np.random.default_rng(0 if train else 1).integers(
                0, 256, size=(40, 3072), dtype=np.uint8
            )
            labels = np.arange(10).repeat(4)
            for index, (name, _) in reversed(list(enumerate(batch_list))):
                start, stop = index * 40 // len(batch_list), (index + 1) * 40 // len(batch_list)
                content = pickle.dumps(
                    {"data": images[start:stop], "labels": labels[start:stop].tolist()}
                )
                member = tarfile.TarInfo(f"{datasets.CIFAR10.base_folder}/{name}")
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
    monkeypatch.setattr(
        datasets.CIFAR10,
        "tgz_md5",
        hashlib.md5(cifar_path.read_bytes(), usedforsecurity=False).hexdigest(),
    )

    class TinyImages(torch.utils.data.Dataset):
        def __init__(self, root, train=True, download=False, transform=None):
            root = Path(root)
            assert root in (data_dir, data_dir / "network")
            assert download == (root == data_dir / "network")
            self.transform = transform
            self.targets = torch.arange(10).repeat_interleave(4)
            self.images = np.random.default_rng(0 if train else 1).integers(
                0, 256, size=(40, 28, 28), dtype=np.uint8
            )
            self.data = torch.from_numpy(self.images)

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            image = Image.fromarray(self.images[index])
            return self.transform(image), int(self.targets[index])

    omniglot_root = data_dir / "omniglot-py"
    omniglot_root.mkdir()
    for split in ("images_background", "images_evaluation"):
        path = omniglot_root / f"{split}.zip"
        with ZipFile(path, "w") as archive:
            for alphabet in (f"{split}_one", f"{split}_two"):
                for character in range(3):
                    for sample in range(20):
                        image = np.random.default_rng(character * 20 + sample).integers(
                            0, 256, size=(28, 28), dtype=np.uint8
                        )
                        content = io.BytesIO()
                        Image.fromarray(image).save(content, format="PNG")
                        archive.writestr(
                            f"{split}/{alphabet}/character{character}/{sample}.png",
                            content.getvalue(),
                        )
        checksum = hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()
        monkeypatch.setitem(datasets.Omniglot.zips_md5, split, checksum)

    def download_archive(url, root, filename, md5):
        root = Path(root)
        if filename == datasets.CIFAR10.filename:
            assert root == data_dir / "network"
            assert url == datasets.CIFAR10.url
            content = cifar_path.read_bytes()
        else:
            assert root == data_dir / "network/omniglot-py"
            assert url == f"{datasets.Omniglot.download_url_prefix}/{filename}"
            content = (omniglot_root / filename).read_bytes()
        assert hashlib.md5(content, usedforsecurity=False).hexdigest() == md5
        root.mkdir(parents=True, exist_ok=True)
        (root / filename).write_bytes(content)

    monkeypatch.setattr(datasets, "MNIST", TinyImages)
    monkeypatch.setattr(dataset_utils, "download_url", download_archive)
    return data_dir


def run_notebook_smoke(path, device, use_network_data=None):
    """Execute real notebook cells with reduced demonstration budgets."""
    namespace = {"__name__": "__main__", "__smoke_device": torch.device(device)}
    notebook = nbformat.read(path, as_version=4)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        tree = ast.fix_missing_locations(
            SmallBudgets(use_network_data=use_network_data).visit(ast.parse(cell.source))
        )
        exec(compile(tree, f"{path}:cell-{index}", "exec"), namespace)
    return namespace


def assert_lesson_smoke(topic, namespace, data_dir, use_network_data=False):
    """Check finite models and the lesson-specific comparison contracts."""
    assert namespace["DATA_DIR"] == data_dir
    assert namespace["USE_NETWORK_DATA"] is use_network_data
    expected_dir = data_dir / "network" if use_network_data else data_dir
    assert namespace["DATASET_DIR"] == expected_dir
    models = [value for value in namespace.values() if isinstance(value, torch.nn.Module)]
    assert models
    for model in models:
        assert all(torch.isfinite(p).all() for p in model.parameters())
    if topic == "cnn":
        assert not (expected_dir / "cifar-10-batches-py").exists()
        assert not (expected_dir / "cifar-10-python.tar.xz").exists()
        assert (
            namespace["train_dataset_cifar"].dataset.images
            is namespace["validation_source_cifar"].images
        )
    if topic == "continual_mnist":
        assert (
            len(
                {
                    run["initial_digest"]
                    for weights in namespace["results"].values()
                    for run in weights.values()
                }
            )
            == 1
        )
        assert len(namespace["results"]["task"]) == 3
        assert len(namespace["results"]["class"]) == 4
        baseline = namespace["results"]["class"][0.0]
        replay = namespace["results"]["class"]["replay"]
        for ordinary, mixed in zip(baseline["history"], replay["history"], strict=True):
            assert ordinary["updates"] == mixed["updates"]
            assert (
                ordinary["current_examples"] == mixed["current_examples"] + mixed["replay_examples"]
            )
            assert (mixed["replay_examples"] > 0) == (mixed["task"] > 0)
        assert (
            max(row["images"] for row in replay["memory"]) <= namespace["CONFIG"]["replay_capacity"]
        )
        for name in ("accuracy_matrix_task_il", "accuracy_matrix_class_il"):
            matrix = namespace[name]
            assert [len(row) for row in matrix] == [1, 2, 3, 4, 5]
            assert all(0 <= score <= 1 for row in matrix for score in row)
    if topic == "meta_learning":
        assert sorted(p.name for p in (expected_dir / "omniglot-py").iterdir()) == [
            "images_background.zip",
            "images_evaluation.zip",
        ]
        assert namespace["N_INNER_STEPS_TRAIN"] == namespace["N_INNER_STEPS_TEST"] == 5
        assert len(namespace["train_ds"].classes) == 4
        assert len(namespace["val_ds"].classes) == 2
        assert set(namespace["train_ds"].classes).isdisjoint(namespace["val_ds"].classes)
        train_paths = {p for files in namespace["train_ds"].class_to_images.values() for p in files}
        eval_paths = {p for files in namespace["val_ds"].class_to_images.values() for p in files}
        assert train_paths.isdisjoint(eval_paths)
        test_paths = {p for files in namespace["test_ds"].class_to_images.values() for p in files}
        assert test_paths.isdisjoint(train_paths | eval_paths)
        assert namespace["maml_model"].initial_digest == namespace["fomaml_model"].initial_digest
        image_ids = {path: index for index, path in enumerate(sorted(train_paths))}
        namespace["train_ds"]._load_image = lambda path: torch.tensor([image_ids[path]])
        support, support_labels, query, query_labels = namespace["train_ds"].sample_episode(2, 1, 2)
        assert set(support.flatten().tolist()).isdisjoint(query.flatten().tolist())
        assert sorted(support_labels.tolist()) == [0, 1]
        assert sorted(query_labels.tolist()) == [0, 0, 1, 1]
        with pytest.raises(ValueError, match="无法提供"):
            namespace["train_ds"].sample_episode(2, 10, 11)
