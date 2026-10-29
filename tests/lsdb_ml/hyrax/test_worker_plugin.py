import lsdb
import numpy as np
import pandas as pd
import pytest
from distributed import Client, WorkerPlugin

from lsdb_ml.hyrax.config_utils import ModelConfig
from lsdb_ml.hyrax.predict import predict
from lsdb_ml.hyrax.worker_plugin import DEFAULT_PLUGIN_NAME

OUTPUT_FIELDS = pd.DataFrame({"pred": pd.Series(dtype="float32")})


class StubModelPlugin(WorkerPlugin):
    """Minimal stand-in for `TorchModelPlugin` that exposes only `model_config`,
    so the tests don't need to load a Hyrax model."""

    def __init__(self, name, model_config):
        self.name = name
        self.model_config = model_config


@pytest.fixture(scope="module")
def client():
    """Multi-process client, so the plugin lives in a different process than `predict`."""
    with Client(n_workers=1, threads_per_worker=1, processes=True, dashboard_address=None) as client:
        model_config = ModelConfig(
            model="HyraxLoopback",
            model_weights_file="fake_weights.pth",
            output_fields=OUTPUT_FIELDS,
        )
        client.register_plugin(StubModelPlugin(DEFAULT_PLUGIN_NAME, model_config))
        yield client


@pytest.fixture
def small_catalog():
    """A small LSDB catalog to call `predict` on."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "id": np.arange(10),
            "ra": rng.uniform(0, 360, 10),
            "dec": rng.uniform(-90, 90, 10),
        }
    )
    return lsdb.from_dataframe(df)


def test_predict_builds_meta_from_worker_plugin(client, small_catalog):
    """`predict` runs on the client, so it must fetch the output fields from the workers."""
    result = predict(small_catalog)

    assert list(result.meta.columns) == [*small_catalog.meta.columns, "pred"]
    assert result.meta["pred"].dtype == np.float32


def test_predict_raises_when_plugin_not_registered(client, small_catalog):
    """A clear error is raised when no plugin is registered under the requested name."""
    with pytest.raises(ValueError, match="not registered on every worker"):
        predict(small_catalog, plugin_name="not_a_registered_plugin")
