"""Inference without a distributed client: run in a module with no client fixture."""

import dask
import pytest
from inference_models import BUILDS, MAG_SPEC, AffineMag, check_alignment, counting_factory

from lsdb_ml import ModelDeployment, ModelSpec
from lsdb_ml.inference.plugin import get_model


def test_without_distributed_client(nested_catalog, nested_frame):
    """Without a client the model is built once in-process and works with local schedulers."""
    BUILDS["count"] = 0
    with dask.config.set(scheduler="threads"), ModelDeployment(ModelSpec(counting_factory)) as deployed:
        assert deployed.client is None
        result = deployed.predict(nested_catalog, MAG_SPEC, {"pred": "float64"}).compute(progress_bar=False)
    assert BUILDS["count"] == 1
    check_alignment(result, nested_frame)
    with pytest.raises(RuntimeError, match="is not deployed on this process"):
        get_model(deployed.name)


def test_processes_scheduler_rejected():
    """The multiprocessing scheduler cannot share an in-process model."""
    with dask.config.set(scheduler="processes"), pytest.raises(RuntimeError, match="synchronous or threaded"):
        ModelDeployment(AffineMag())
