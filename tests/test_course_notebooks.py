"""Run the actual course cells with tiny offline datasets and short training budgets."""

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import datasets

from biai import paths

TOPICS = ("mlp", "cnn", "continual_mnist", "meta_learning", "atari")


def notebook_path(topic):
    return paths.PROJECT_ROOT / "biai" / topic / f"{topic}.ipynb"


@pytest.mark.parametrize("topic", TOPICS)
def test_notebook_format_code_and_local_links(topic):
    path = notebook_path(topic)
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, str(path), "exec")
            assert cell.execution_count is None
            assert cell.outputs == []
        else:
            for target in re.findall(r"\]\(([^)]+)\)", cell.source):
                if target.startswith(("https://", "http://", "#")):
                    continue
                assert (path.parent / target.split("#")[0]).exists(), target


class SmallBudgets(ast.NodeTransformer):
    """Change only demonstration budgets in the in-memory smoke copy."""

    values = {
        "epochs": 1,
        "N_WAY": 2,
        "Q_QUERY": 1,
        "N_TASKS_PER_BATCH": 1,
        "N_TEST_EPISODES": 2,
        "N_EPOCHS_DEMO_MAML": 1,
        "N_EPOCHS_DEMO_FOMAML": 1,
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


@pytest.fixture
def offline_data(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    monkeypatch.setattr(paths, "DATA_DIR", data_dir)
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))

    class TinyImages(torch.utils.data.Dataset):
        def __init__(self, root, train=True, download=False, transform=None, *, rgb=False):
            assert Path(root) == data_dir
            assert download
            self.transform = transform
            self.targets = torch.arange(10).repeat_interleave(4)
            shape = (40, 32, 32, 3) if rgb else (40, 28, 28)
            self.images = np.random.default_rng(0 if train else 1).integers(
                0, 256, size=shape, dtype=np.uint8
            )

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            image = Image.fromarray(self.images[index])
            return self.transform(image), int(self.targets[index])

    def prepare_omniglot(root, *, background, download):
        assert Path(root) == data_dir
        assert download
        split = "images_background" if background else "images_evaluation"
        for alphabet in (f"{split}_one", f"{split}_two"):
            for character in range(3):
                directory = data_dir / "omniglot-py" / split / alphabet / f"character{character}"
                directory.mkdir(parents=True)
                for sample in range(20):
                    image = np.random.default_rng(character * 20 + sample).integers(
                        0, 256, size=(28, 28), dtype=np.uint8
                    )
                    Image.fromarray(image).save(directory / f"{sample}.png")

    monkeypatch.setattr(datasets, "MNIST", TinyImages)
    monkeypatch.setattr(datasets, "CIFAR10", lambda *a, **kw: TinyImages(*a, **kw, rgb=True))
    monkeypatch.setattr(datasets, "Omniglot", prepare_omniglot)
    return data_dir


@pytest.mark.parametrize("topic", TOPICS[:-1])
@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.filterwarnings("error:__array__ implementation.*:DeprecationWarning")
def test_course_cells_train_evaluate_and_plot_offline(topic, device, offline_data, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    path = notebook_path(topic)
    monkeypatch.chdir(path.parent)
    namespace = {"__name__": "__main__", "__smoke_device": torch.device(device)}
    notebook = nbformat.read(path, as_version=4)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        tree = ast.fix_missing_locations(SmallBudgets().visit(ast.parse(cell.source)))
        exec(compile(tree, f"{path}:cell-{index}", "exec"), namespace)
    assert namespace["DATA_DIR"] == offline_data
    models = [value for value in namespace.values() if isinstance(value, torch.nn.Module)]
    assert models
    for model in models:
        assert all(torch.isfinite(p).all() for p in model.parameters())
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


def test_installed_package_and_atari_cli_work_outside_the_repository(tmp_path):
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "biai.atari.scripts.run_experiments",
            "train",
            "teaching",
            "--dry-run",
            "--results-dir",
            str(tmp_path / "results"),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.count("[24/24]") == 1
    assert "-m biai.atari.scripts.train_continual" in result.stdout
    assert "-m biai.atari.scripts.evaluate" in result.stdout
    assert not (tmp_path / "results").exists()
    metadata = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import json; from biai.paths import PROJECT_ROOT, DATA_DIR; "
                "print(json.dumps([str(PROJECT_ROOT), str(DATA_DIR)]))",
            ],
            cwd=tmp_path,
            env=environment,
            text=True,
        )
    )
    assert metadata == [str(paths.PROJECT_ROOT), str(paths.PROJECT_ROOT / "data")]
