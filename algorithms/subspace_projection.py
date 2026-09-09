"""Boundary input subspaces and projection of actual Adam steps for continual learning.

GPM projects onto the complement of the stored input bases. Biases are
included as an extra input coordinate, preserving the existing Atari network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def affine_layers(backbone: nn.Module) -> dict[str, nn.Module]:
    return {
        name: layer
        for name, layer in backbone.named_modules()
        if isinstance(layer, (nn.Conv2d, nn.Linear))
    }


@torch.no_grad()
def build_input_subspaces(
    backbone: nn.Module,
    observations: torch.Tensor,
    *,
    threshold: float,
    seed: int,
    batch_size: int = 32,
    patches_per_observation: int = 16,
    previous: dict | None = None,
) -> dict:
    """Compute uncentered SVD bases via float64 second moments at a task boundary.

    Convolution inputs are sampled receptive-field patches with a private RNG.
    No teacher logits, rewards, or held-out policy probes enter this calculation.
    Existing bases are retained; residual directions are added until the combined
    basis captures the threshold fraction of the current task input energy.
    See Saha et al. (2021), GPM memory update (equations 8-9).
    """
    if not 0 < threshold <= 1:
        raise ValueError("Require threshold in (0, 1]")
    if observations.dtype != torch.uint8 or not len(observations):
        raise ValueError("Require nonempty uint8 observations")
    layers = affine_layers(backbone)
    device = next(backbone.parameters()).device
    generator = torch.Generator().manual_seed(seed)
    moments, counts, handles = {}, {}, []

    def capture(name, layer, inputs):
        x = inputs[0]
        if isinstance(layer, nn.Conv2d):
            x = F.unfold(
                x, layer.kernel_size, layer.dilation, layer.padding, layer.stride
            ).transpose(1, 2)
            indices = torch.randint(
                x.shape[1], (len(x), patches_per_observation), generator=generator
            ).to(device)
            x = x.gather(1, indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        x = x.reshape(-1, x.shape[-1]).double()
        if layer.bias is not None:
            x = torch.cat((x, torch.ones(len(x), 1, device=device, dtype=x.dtype)), dim=1)
        if name not in moments:
            moments[name] = torch.zeros(x.shape[1], x.shape[1], dtype=x.dtype, device=device)
            counts[name] = 0
        moments[name].addmm_(x.T, x)
        counts[name] += len(x)

    try:
        for name, layer in layers.items():
            handles.append(
                layer.register_forward_pre_hook(
                    lambda module, inputs, name=name: capture(name, module, inputs)
                )
            )
        for batch in observations.split(batch_size):
            backbone(batch.to(device).float() / 255.0)
    finally:
        for handle in handles:
            handle.remove()

    result = {}
    for name, moment in moments.items():
        total = moment.trace().clamp_min(0)
        old = moment.new_empty((len(moment), 0))
        if previous is not None:
            old = previous[name]["basis"].to(moment)
            # Reorthogonalize rounded stored bases without changing their span.
            old = torch.linalg.qr(old, mode="reduced").Q
        covered = (old * (moment @ old)).sum().clamp(min=0, max=total)
        residual = moment - old @ (old.T @ moment)
        residual = residual - (residual @ old) @ old.T
        residual = (residual + residual.T) / 2
        eigenvalues, vectors = torch.linalg.eigh(residual)
        energy = eigenvalues.flip(0).clamp_min(0)
        required = (threshold * total - covered).clamp_min(0)
        available = len(moment) - old.shape[1]
        rank = 0
        if float(required) > float(total) * 1e-12 and available:
            rank = min(int(torch.searchsorted(energy.cumsum(0), required)) + 1, available)
        added = vectors.flip(1)[:, :rank]
        if old.shape[1] and rank:
            added = added - old @ (old.T @ added)
            added = torch.linalg.qr(added, mode="reduced").Q
        basis = torch.cat((old, added), dim=1)
        captured = (basis * (moment @ basis)).sum()
        result[name] = {
            "basis": basis.float().cpu().contiguous(),
            "singular_values": energy[:rank].sqrt().cpu(),
            "dimension": len(moment),
            "rank": basis.shape[1],
            "previous_rank": old.shape[1],
            "added_rank": rank,
            "captured_energy_fraction": float(captured / total) if float(total) else 1.0,
            "sample_count": counts[name],
        }
    return result


def affine_matrix(layer: nn.Module) -> torch.Tensor:
    weight = layer.weight.flatten(1)
    return weight if layer.bias is None else torch.cat((weight, layer.bias[:, None]), dim=1)


class AdamSubspaceProjection:
    """Project optimizer displacements after Adam's moment scaling, not raw grads.

    Hooks leave Adam's moments and the existing learner unchanged. Computing
    the displacement from pre/post weights introduces only floating-point rounding;
    the realized projection error is measured on every update. Only the shared
    affine layers are projected, including their biases.
    """

    def __init__(self, optimizer, backbone: nn.Module, subspaces: dict):
        self.layers = affine_layers(backbone)
        self.bases, self.before, self.sums = {}, {}, {}
        self.steps = 0
        for name, layer in self.layers.items():
            self.bases[name] = subspaces[name]["basis"].to(layer.weight)
            self.sums[name] = torch.zeros(4, device=layer.weight.device, dtype=torch.float64)
        self.handles = (
            optimizer.register_step_pre_hook(self._before_step),
            optimizer.register_step_post_hook(self._after_step),
        )

    @torch.no_grad()
    def _before_step(self, optimizer, args, kwargs):
        self.before = {name: affine_matrix(layer).clone() for name, layer in self.layers.items()}

    @torch.no_grad()
    def _after_step(self, optimizer, args, kwargs):
        for name, layer in self.layers.items():
            before, basis = self.before[name], self.bases[name]
            delta = affine_matrix(layer) - before
            coordinates = delta @ basis
            projected = delta - coordinates @ basis.T
            updated = before + projected
            width = layer.weight[0].numel()
            layer.weight.copy_(updated[:, :width].reshape_as(layer.weight))
            if layer.bias is not None:
                layer.bias.copy_(updated[:, -1])
            applied = affine_matrix(layer) - before
            protected = applied @ basis
            error = protected
            self.sums[name] += torch.stack(
                [x.square().sum().double() for x in (delta, applied, protected, error)]
            )
        self.steps += 1
        self.before.clear()

    def metrics(self) -> dict:
        result = {}
        for name, values in self.sums.items():
            raw, applied, protected, error = values.tolist()
            result[name] = {
                "raw_update_energy": raw,
                "retained_update_energy_fraction": applied / raw if raw else 0,
                "protected_update_energy_fraction": protected / raw if raw else 0,
                "projection_relative_error": (error / raw) ** 0.5 if raw else 0,
            }
        return {"optimizer_steps": self.steps, "layers": result}

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
