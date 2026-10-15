import numpy as np
import pandas as pd
import pytest
import torch
from distributed import Client, LocalCluster
from inference_models import (
    BUILDS,
    MAG_SPEC,
    AffineMag,
    DropsRow,
    MaskedMean,
    Stacked,
    check_alignment,
    counting_factory,
)

from lsdb_ml import ColumnSpec, Flat, ModelDeployment, ModelSpec, Nested, OutputSpec, predict


@pytest.fixture(scope="module")
def client():
    """Two in-process workers with two threads each, so tasks run concurrently."""
    with (
        LocalCluster(n_workers=2, threads_per_worker=2, processes=False, dashboard_address=None) as cluster,
        Client(cluster) as dask_client,
    ):
        yield dask_client


@pytest.fixture
def scrambled_catalog(nested_catalog):
    """A non-trivial layout: a filter that empties some partitions, then a reordered subset."""
    filtered = nested_catalog.query("mag > 21")
    n = filtered.npartitions
    return filtered.partitions[[n - 1, 3, 0, 7, 5, 2, 9, 11, 1, 4, 6, 8, 10] + list(range(12, n - 1))]


def test_index_aligned_after_nontrivial_layout(client, scrambled_catalog, nested_frame):
    """Predictions line up with their input rows across empty, reordered partitions."""
    computed_input = scrambled_catalog.compute(progress_bar=False)
    assert 0 in [len(p.compute(progress_bar=False)) for p in scrambled_catalog.partitions]

    with ModelDeployment(ModelSpec(AffineMag)) as deployed:
        lazy = deployed.predict(scrambled_catalog, MAG_SPEC, {"pred": "float64"}, batch_size=3)
        result = lazy.compute(progress_bar=False)
    assert len(result) == len(computed_input)
    assert result.index.equals(computed_input.index)
    check_alignment(result, nested_frame)


def test_alignment_with_duplicate_index(client, nested_frame):
    """Rows sharing a _healpix_29 value still get their own predictions."""
    import lsdb

    frame = nested_frame.iloc[:60].copy()
    frame["ra"] = 10.0  # every object at the same position: one index value repeated
    frame["dec"] = 10.0
    catalog = lsdb.from_dataframe(frame, partition_rows=100)
    result = predict(catalog, AffineMag(), MAG_SPEC, {"pred": "float64"}, batch_size=7).compute(
        progress_bar=False
    )
    assert not result.index.is_unique
    check_alignment(result, nested_frame)


def test_model_loaded_once_per_worker(client, nested_catalog):
    """The worker plugin builds the model once per worker, not once per partition."""
    BUILDS["count"] = 0
    spec = ModelSpec(counting_factory)
    with ModelDeployment(spec) as deployed:
        deployed.predict(nested_catalog, MAG_SPEC, {"pred": "float64"}).compute(progress_bar=False)
        deployed.predict(nested_catalog, MAG_SPEC, {"pred": "float64"}).compute(progress_bar=False)
        # Deploying the same spec again reuses the installed plugin.
        predict(nested_catalog, spec, MAG_SPEC, {"pred": "float64"}).compute(progress_bar=False)
    assert nested_catalog.npartitions > len(client.scheduler_info()["workers"])
    assert BUILDS["count"] == len(client.scheduler_info()["workers"]) == 2


def test_meta_matches_computed(client, nested_catalog):
    """The lazy result's meta has the same columns and dtypes as the computed data."""
    for append in (True, False):
        lazy = predict(nested_catalog, AffineMag(), MAG_SPEC, {"pred": "float32"}, append=append)
        computed = lazy.compute(progress_bar=False)
        assert list(lazy.columns) == list(computed.columns)
        assert lazy.dtypes.to_dict() == computed.dtypes.to_dict()
        assert computed["pred"].dtype == np.float32
    assert list(lazy.columns) == ["pred"]
    assert len(computed) == 500


def test_nested_inputs(client, nested_catalog):
    """Nested columns are padded and masked, so a masked mean matches a per-row mean."""
    spec = ColumnSpec({"flux": Nested("lc", "flux", dtype=torch.float64)})
    result = predict(
        nested_catalog,
        MaskedMean(),
        spec,
        {"mean_flux": "float64"},
        inputs=["flux", "flux_mask"],
        batch_size=9,
    ).compute(progress_bar=False)
    expected = [
        float(np.mean(v["flux"])) if isinstance(v, pd.DataFrame) and len(v) else 0.0
        for v in result["lc"].array
    ]
    np.testing.assert_allclose(result["mean_flux"].to_numpy(), expected, rtol=1e-6)


def test_multi_column_output(client, nested_catalog):
    """A (B, k) output fills k columns in order."""
    result = predict(nested_catalog, Stacked(), MAG_SPEC, ["plus", "minus"], append=False).compute(
        progress_bar=False
    )
    np.testing.assert_allclose(result["plus"], -result["minus"])


def test_wrong_row_count_fails(client, nested_catalog):
    """A model that drops rows fails loudly instead of misaligning predictions."""
    lazy = predict(nested_catalog, DropsRow(), MAG_SPEC, {"pred": "float32"})
    with pytest.raises(Exception, match="exactly one prediction per input row"):
        lazy.compute(progress_bar=False)


def test_validation(client, unloadable_catalog):
    """Bad specs, outputs and arguments fail before any data is read."""
    with pytest.raises(ValueError, match="already exist"):
        predict(unloadable_catalog, AffineMag(), MAG_SPEC, {"mag": "float32"})
    with pytest.raises(ValueError, match="pass inputs="):
        predict(unloadable_catalog, AffineMag(), ColumnSpec({"a": Flat("ra"), "b": Flat("dec")}), ["p"])
    with pytest.raises(ValueError, match="not in the catalog"):
        predict(unloadable_catalog, AffineMag(), ColumnSpec({"a": Flat("nope")}), ["p"])
    with pytest.raises(ValueError, match="batch_size"):
        predict(unloadable_catalog, AffineMag(), MAG_SPEC, ["p"], batch_size=0)
    with pytest.raises(TypeError, match="ModelSpec or torch.nn.Module"):
        ModelDeployment("model")


def test_closed_deployment(client, nested_catalog):
    """After close, the model is gone from the workers and computing fails clearly."""
    deployed = ModelDeployment(ModelSpec(AffineMag, device="cpu", torch_threads=None))
    lazy = deployed.predict(nested_catalog, MAG_SPEC, {"pred": "float32"})
    deployed.close()
    deployed.close()
    with pytest.raises(Exception, match="is not deployed"):
        lazy.compute(progress_bar=False)
    with pytest.raises(RuntimeError, match="closed"):
        deployed.predict(nested_catalog, MAG_SPEC, {"pred": "float32"})


def test_weights_from_training_checkpoint(client, nested_catalog, tmp_path):
    """ModelSpec loads a state dict or a TrainSession checkpoint on each worker."""
    from lsdb_ml.training import save_checkpoint

    trained = AffineMag()
    with torch.no_grad():
        trained.linear.bias.fill_(5.0)
    save_checkpoint(tmp_path / "ckpt.pt", trained)
    torch.save(trained.state_dict(), tmp_path / "state.pt")
    for weights in ("ckpt.pt", "state.pt"):
        spec = ModelSpec(AffineMag, weights=tmp_path / weights)
        result = predict(nested_catalog, spec, MAG_SPEC, {"pred": "float64"}).compute(progress_bar=False)
        np.testing.assert_allclose(result["pred"], 2 * result["mag"].astype(np.float64) + 5, rtol=1e-6)


def test_output_spec_errors():
    """Outputs that do not match the declared columns are rejected."""
    spec = OutputSpec({"a": "float32", "b": "float32"})
    with pytest.raises(ValueError, match="expected \\(3, 2\\)"):
        spec.to_columns(torch.zeros(3), 3)
    with pytest.raises(ValueError, match="returned keys"):
        spec.to_columns({"a": torch.zeros(3)}, 3)
    with pytest.raises(ValueError, match="expected \\(3,\\)"):
        spec.to_columns({"a": torch.zeros(3), "b": torch.zeros(3, 2)}, 3)
    with pytest.raises(TypeError, match="torch.Tensor"):
        spec.to_columns([1, 2, 3], 3)
    with pytest.raises(ValueError, match="at least one column"):
        OutputSpec([])
    assert set(spec.to_columns({"a": torch.zeros(3, 1), "b": torch.zeros(3)}, 3)) == {"a", "b"}
