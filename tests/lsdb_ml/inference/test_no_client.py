"""Behavior without a distributed client; kept apart from the module-scoped client fixture."""

import pytest
from inference_models import AffineMag

from lsdb_ml import ModelDeployment


def test_requires_distributed_client():
    """Without an active client, deploying fails with a clear message."""
    with pytest.raises(RuntimeError, match="needs a dask.distributed Client"):
        ModelDeployment(AffineMag())
