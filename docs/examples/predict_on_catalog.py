"""Run a PyTorch model over every row of an LSDB catalog on Dask workers.

Run with ``python docs/examples/predict_on_catalog.py``. It uses a synthetic catalog;
for a real one, replace ``make_catalog()`` with ``lsdb.open_catalog(...)``.

The model is built once per Dask worker (by a worker plugin) and the result is a lazy
catalog with prediction columns, aligned row by row with the input.
"""

from pathlib import Path

import lsdb
import nested_pandas as npd
import numpy as np
import pandas as pd
import torch
from distributed import Client, LocalCluster
from torch import nn

from lsdb_ml import ModelDeployment, ModelSpec


def make_catalog(n_objects=5000, seed=0):
    """Objects with a magnitude and a ragged light curve."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(0, 40, n_objects)
    objects = pd.DataFrame(
        {
            "object_id": np.arange(n_objects),
            "ra": rng.uniform(0, 360, n_objects),
            "dec": rng.uniform(-60, 60, n_objects),
            "mag": rng.normal(20, 1, n_objects).astype(np.float32),
        }
    )
    light_curves = pd.DataFrame(
        {"mjd": rng.uniform(0, 100, lengths.sum()), "flux": rng.normal(1, 0.1, lengths.sum())},
        index=np.repeat(np.arange(n_objects), lengths),
    )
    return lsdb.from_dataframe(npd.NestedFrame(objects).join_nested(light_curves, "lc"), partition_rows=500)


class Scorer(nn.Module):
    """Toy model combining a flat column with a masked summary of a nested column."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(2, 2)

    def forward(self, batch):
        """Return two scores per object from the magnitude and the mean light-curve flux."""
        # Batches are keyed by column name; nested sub-columns share a "<column>._mask".
        mag, flux, mask = batch["mag"], batch["lc.flux"], batch["lc._mask"]
        mean_flux = (flux * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.head(torch.stack([mag, mean_flux], dim=1))


def main(weights_path):
    """Save some weights, deploy the model and predict on the catalog."""
    torch.manual_seed(0)
    torch.save(Scorer().state_dict(), weights_path)  # stands in for a trained model

    catalog = make_catalog().query("mag < 21")  # any lazy lsdb pipeline works here
    # Workers build Scorer() and load the weights themselves; only the factory and path are shipped.
    with ModelDeployment(ModelSpec(Scorer, weights=weights_path, device="cpu")) as deployed:
        scored = deployed.predict(
            catalog,
            inputs=["mag", "lc.flux"],  # flat vs nested is read from the catalog's meta
            outputs={"score_a": "float32", "score_b": "float32"},
            max_length=32,  # keep at most the last 32 observations per light curve
            truncate="tail",
            batch_size=1024,
        )
        print(scored.dtypes[["object_id", "mag", "score_a", "score_b"]])  # meta, nothing computed yet
        result = scored.compute()
    print(result[["object_id", "mag", "score_a", "score_b"]].head())
    print(f"{len(result)} rows scored")


if __name__ == "__main__":
    with LocalCluster(n_workers=2, threads_per_worker=2, dashboard_address=None) as cluster, Client(cluster):
        main(Path("scorer_weights.pt").resolve())
