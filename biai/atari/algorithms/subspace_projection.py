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


def sample_conv_inputs(x, layer, count, generator):
    """Select receptive fields from a strided view before materializing patch data."""
    kh, kw = layer.kernel_size
    dh, dw = layer.dilation
    ph, pw = layer.padding
    if ph or pw:
        x = F.pad(x, (pw, pw, ph, ph))
    windows = x.unfold(2, dh * (kh - 1) + 1, layer.stride[0]).unfold(
        3, dw * (kw - 1) + 1, layer.stride[1]
    )
    height, width = windows.shape[2:4]
    indices = torch.randint(height * width, (len(x), count), generator=generator).to(x.device)
    batches = torch.arange(len(x), device=x.device)[:, None]
    patches = windows[batches, :, indices // width, indices % width]
    return patches[..., ::dh, ::dw].reshape(len(x), count, -1)


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
            x = sample_conv_inputs(x, layer, patches_per_observation, generator)
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
        required = (threshold * total - covered).clamp_min(0)
        available = len(moment) - old.shape[1]
        rank = 0
        added, energy = old[:, :0], moment.new_empty(0)
        if float(required) > float(total) * 1e-12 and available:
            residual = moment - old @ (old.T @ moment)
            residual = residual - (residual @ old) @ old.T
            residual = (residual + residual.T) / 2
            eigenvalues, vectors = torch.linalg.eigh(residual)
            energy = eigenvalues.flip(0).clamp_min(0)
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

    def __init__(
        self, optimizer, backbone: nn.Module, subspaces: dict, *, compile_projection: bool = False
    ):
        self.layers = affine_layers(backbone)
        self.bases, self.before, self.sums = {}, {}, {}
        self.steps = 0
        self.capturing_update = False
        optimizer._subspace_projection = self
        self.optimizer = optimizer
        for name, layer in self.layers.items():
            self.bases[name] = subspaces[name]["basis"].to(layer.weight)
            self.before[name] = torch.empty_like(affine_matrix(layer))
            self.sums[name] = torch.zeros(4, device=layer.weight.device, dtype=torch.float64)
        self._project = (
            torch.compile(self._project_step, mode="reduce-overhead", fullgraph=True)
            if compile_projection
            else self._project_step
        )
        self._project_for_capture = torch.compile(
            self._project_step, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}
        )
        self.handles = (
            optimizer.register_step_pre_hook(self._before_step),
            optimizer.register_step_post_hook(self._after_step),
        )

    @torch.no_grad()
    def _before_step(self, optimizer, args, kwargs):
        for name, layer in self.layers.items():
            self.before[name].copy_(affine_matrix(layer))

    @torch.no_grad()
    def _after_step(self, optimizer, args, kwargs):
        statistics = self._project_for_capture() if self.capturing_update else self._project()
        for name, values in zip(self.layers, statistics):
            self.sums[name].add_(values)
        if not self.capturing_update:
            self.steps += 1

    @torch.no_grad()
    def _project_step(self):
        statistics = []
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
            statistics.append(
                torch.stack([x.square().sum().double() for x in (delta, applied, protected, error)])
            )
        return tuple(statistics)

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
        self.optimizer._subspace_projection = None
        for handle in self.handles:
            handle.remove()
