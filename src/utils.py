"""Exponential moving average of the model weights."""
import torch


class EMA:
    """Exponential moving average of the model weights: smoother, better-generalising model for evaluation."""
    def __init__(self, model, decay=0.999):
        import copy
        self.m = copy.deepcopy(model).eval(); self.d = decay
        for p in self.m.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.m.parameters(), model.parameters()): e.mul_(self.d).add_(p.detach(), alpha=1 - self.d)
        for e, b in zip(self.m.buffers(), model.buffers()): e.copy_(b)
