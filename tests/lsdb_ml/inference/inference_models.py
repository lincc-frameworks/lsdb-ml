"""Tiny models and checks shared by the inference tests."""

import numpy as np
import torch
from torch import nn

# Counts factory calls; with processes=False every worker shares this module.
BUILDS = {"count": 0}


class AffineMag(nn.Module):
    """pred = 2 * mag + 1, exactly, so every row can be checked against its own input."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        with torch.no_grad():
            self.linear.weight.fill_(2.0)
            self.linear.bias.fill_(1.0)

    def forward(self, batch):
        """Map the (B,) "mag" tensor to (B,) predictions."""
        return self.linear(batch["mag"].unsqueeze(-1)).squeeze(-1)


def counting_factory():
    """Build AffineMag and count how often a model is built."""
    BUILDS["count"] += 1
    return AffineMag()


class MaskedMean(nn.Module):
    """Mean of the real (unpadded) light-curve values; 0 for empty light curves."""

    def forward(self, batch):
        """Average padded values using the mask."""
        flux, mask = batch["lc.flux"], batch["lc._mask"]
        return (flux * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


class Stacked(nn.Module):
    """Two outputs per row, as a (B, 2) tensor."""

    def forward(self, batch):
        """Return [mag, -mag]."""
        return torch.stack([batch["mag"], -batch["mag"]], dim=1)


class DropsRow(nn.Module):
    """A broken model that returns one prediction too few."""

    def forward(self, batch):
        """Drop the last row."""
        return batch["mag"][:-1]


MAG_INPUTS = "mag"


def check_alignment(result, source_frame):
    """Every prediction matches its own row's input, and rows match the source exactly once."""
    np.testing.assert_allclose(
        result["pred"].to_numpy(), 2 * result["mag"].to_numpy(np.float64) + 1, rtol=1e-6
    )
    # Also join on the object id against an eager computation on the raw frame.
    expected = 2 * source_frame.set_index("id")["mag"].astype(np.float64) + 1
    assert result["id"].is_unique
    joined = result.set_index("id")["pred"]
    np.testing.assert_allclose(joined.to_numpy(), expected.loc[joined.index].to_numpy(), rtol=1e-6)
