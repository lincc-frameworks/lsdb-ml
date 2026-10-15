"""Train a small model on an LSDB catalog with a nested light-curve column.

Run with ``python docs/examples/train_from_catalog.py``. It builds a synthetic catalog,
so no data download is needed. To use a real catalog, replace ``make_catalog()`` with
``lsdb.open_catalog(...)`` and adjust the spec's column names.
"""

import lsdb
import nested_pandas as npd
import numpy as np
import pandas as pd
import torch
from distributed import Client
from torch import nn
from torch.utils.data import DataLoader

from lsdb_ml import CatalogIterableDataset, ColumnSpec, Flat, Nested, SupervisedStep, TrainSession


def make_catalog(n_objects=2000, seed=0):
    """Objects whose label is the mean flux of their (ragged) light curve, plus noise."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, 30, n_objects)
    flux = rng.normal(0, 1, lengths.sum()).astype(np.float32)
    owner = np.repeat(np.arange(n_objects), lengths)
    mean_flux = np.bincount(owner, weights=flux) / lengths
    objects = pd.DataFrame(
        {
            "ra": rng.uniform(0, 360, n_objects),
            "dec": rng.uniform(-60, 60, n_objects),
            "label": (mean_flux + rng.normal(0, 0.01, n_objects)).astype(np.float32),
        }
    )
    light_curves = pd.DataFrame({"mjd": rng.uniform(0, 100, lengths.sum()), "flux": flux}, index=owner)
    frame = npd.NestedFrame(objects).join_nested(light_curves, "lc")
    return lsdb.from_dataframe(frame, partition_rows=200)


class MaskedMeanModel(nn.Module):
    """Predict the label from the masked mean of a learned per-observation embedding."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(1, 8)
        self.head = nn.Linear(8, 1)

    def forward(self, flux, mask):
        """Pool padded observations using the mask, so padding never contributes."""
        hidden = self.embed(flux.unsqueeze(-1)) * mask.unsqueeze(-1)
        pooled = hidden.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return self.head(pooled).squeeze(-1)


def main():
    """Stream the catalog for a few epochs and train until the loss is small."""
    catalog = make_catalog()
    spec = ColumnSpec({"flux": Nested("lc", "flux"), "label": Flat("label")})

    torch.manual_seed(0)
    model = MaskedMeanModel()
    step = SupervisedStep(
        loss_fn=nn.MSELoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
        inputs=["flux", "flux_mask"],
        target="label",
    )
    dataset = CatalogIterableDataset(catalog, spec, batch_size=64, seed=0, epochs=20, partitions_per_chunk=2)
    loader = DataLoader(dataset, batch_size=None, num_workers=0)

    recent = []
    with TrainSession(model, step, loader, checkpoint_dir="checkpoints", checkpoint_every=100) as session:
        for _, metrics in session:
            if metrics is None:  # skipped (empty) batch
                continue
            recent = (recent + [metrics["loss"]])[-10:]
            if session.batch_count % 10 == 0:
                print(f"batch {session.batch_count}: mean loss of last 10 batches {np.mean(recent):.5f}")
            if len(recent) == 10 and np.mean(recent) < 1e-3:
                print(f"converged after {session.batch_count} batches")
                session.stop()
    print("final checkpoint: checkpoints/checkpoint-final.pt")


if __name__ == "__main__":
    # A distributed client lets the stream compute the next chunk while the model trains.
    with Client(processes=False, n_workers=2, threads_per_worker=1, dashboard_address=None):
        main()
