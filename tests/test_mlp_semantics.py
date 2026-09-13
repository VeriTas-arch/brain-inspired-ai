"""Regression checks for the mlp lesson."""

import ast
import json

import torch
import torch.nn.functional as F

from biai.paths import PROJECT_ROOT
from tests.notebook_helpers import definitions


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
    cells = [
        "".join(cell["source"])
        for cell in json.loads(path.read_text())["cells"]
        if cell["cell_type"] == "code"
        and any(
            isinstance(node, ast.FunctionDef) and node.name == "manual_backprop_update"
            for node in ast.parse("".join(cell["source"])).body
        )
    ]
    assert len(cells) == 1, "Expected one manual backpropagation definition cell"
    exec(cells[0], ns)
    x = torch.rand(64, 784, generator=torch.Generator().manual_seed(1))
    _, logits = ns["forward"](x)
    assert F.cross_entropy(logits, torch.arange(64) % 10) < 5
    assert logits.softmax(1).max(1).values.mean() < 0.5


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
