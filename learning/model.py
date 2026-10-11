"""Shared pair scorers. Local IDs select features; IDs never enter the network."""
import torch
from torch import nn


class FrozenEncoder(nn.Module):
    """Wrap a supplied local encoder; no download or architecture assumptions.

    Caller supplies only allowed visual/context tensors and pools its outputs to D.
    A real model-specific adapter is deliberately not implemented in v1.
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(False)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, allowed_tensor):
        self.encoder.eval()
        return self.encoder(allowed_tensor).detach()


class RelationHead(nn.Module):
    def __init__(self, dimension, hidden=64):
        super().__init__()
        self.dimension = dimension
        self.hidden = hidden
        self.relation = nn.Sequential(nn.Linear(3 * dimension, hidden), nn.ReLU(), nn.Linear(hidden, 3))
        self.sufficiency = nn.Sequential(nn.Linear(3 * dimension, hidden), nn.ReLU(), nn.Linear(hidden, 3))
        self.match = nn.Sequential(nn.Linear(2 * dimension, hidden), nn.ReLU(), nn.Linear(hidden, 3))

    def forward(self, door, region_a, region_b, view_a, view_b):
        tensors = (door, region_a, region_b, view_a, view_b)
        if any(x.ndim != 2 or x.shape[-1] != self.dimension for x in tensors):
            raise ValueError("expected [candidate_count, dimension]")
        if door.shape != region_a.shape or door.shape != region_b.shape or view_a.shape != view_b.shape:
            raise ValueError("pair shape mismatch")
        # Door-region connectivity and same-door identity are unordered relations.
        r = torch.cat((door, region_a + region_b, (region_a - region_b).abs()), -1)
        v = torch.cat((view_a + view_b, (view_a - view_b).abs()), -1)
        return {"relations": self.relation(r), "sufficiency": self.sufficiency(r), "matches": self.match(v)}
