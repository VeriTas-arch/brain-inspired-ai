"""Capture a fixed-shape learner update without consuming warmup training steps."""

import torch


class CudaUpdate:
    """Own stable inputs, gradients and a graph spanning loss, backward and Adam."""

    def __init__(self, optimizer, loss, clip_gradients, inputs):
        self.optimizer = optimizer
        self.inputs = tuple(value.detach().clone() for value in inputs)
        self.parameters = [p for group in optimizer.param_groups for p in group["params"]]
        self.projection = getattr(optimizer, "_subspace_projection", None)
        # The outer graph owns capture for the entire update. Disable only nested
        # Inductor capture, retaining full-graph loss/clip fusion inside that graph.
        options = {"triton.cudagraphs": False}
        self.loss = torch.compile(loss, fullgraph=True, dynamic=False, options=options)
        self.clip = torch.compile(clip_gradients, fullgraph=True, dynamic=False, options=options)
        weights = [p.detach().clone() for p in self.parameters]
        state = {
            p: {name: value.clone() for name, value in values.items()}
            for p, values in optimizer.state.items()
        }
        capturable = [group["capturable"] for group in optimizer.param_groups]
        projection_sums = (
            {name: value.clone() for name, value in self.projection.sums.items()}
            if self.projection is not None
            else {}
        )
        device = self.inputs[0].device
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        try:
            if self.projection is not None:
                self.projection.capturing_update = True
            optimizer.zero_grad(set_to_none=True)
            with torch.random.fork_rng(devices=[device]):
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self._step()
                torch.cuda.current_stream(device).wait_stream(stream)
                for group in optimizer.param_groups:
                    group["capturable"] = True
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    self.diagnostics = self._step()
            self.gradients = [p.grad for p in self.parameters]
        finally:
            # Copy into the captured allocations; load_state_dict would replace
            # optimizer tensors and leave the graph pointing at stale state.
            torch.cuda.current_stream(device).wait_stream(stream)
            with torch.no_grad():
                for parameter, saved in zip(self.parameters, weights, strict=True):
                    parameter.copy_(saved)
                for parameter, values in optimizer.state.items():
                    for name, value in values.items():
                        if parameter in state:
                            value.copy_(state[parameter][name])
                        else:
                            value.zero_()
                if self.projection is not None:
                    for name, value in self.projection.sums.items():
                        value.copy_(projection_sums[name])
                    self.projection.capturing_update = False
            for group, original in zip(optimizer.param_groups, capturable, strict=True):
                group["capturable"] = original

    def _step(self):
        self.optimizer.zero_grad(set_to_none=False)
        loss, diagnostics = self.loss(*self.inputs)
        loss.backward()
        self.clip()
        self.optimizer.step()
        return diagnostics

    def __call__(self, *inputs, refresh_prefix=True):
        for index, (destination, source) in enumerate(zip(self.inputs, inputs, strict=True)):
            if refresh_prefix or index == len(inputs) - 1:
                destination.copy_(source)
        # Task graphs have distinct gradient allocations, including for a shared
        # backbone. Restore the active graph's pointers and inactive heads' None.
        for parameter, gradient in zip(self.parameters, self.gradients, strict=True):
            parameter.grad = gradient
        self.graph.replay()
        if self.projection is not None:
            self.projection.steps += 1
        return self.diagnostics
