"""Running a model over every row of an LSDB catalog."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import dask
import numpy as np
import pandas as pd
import torch
from torch import nn

from lsdb_ml.data.dataset import iter_slices
from lsdb_ml.device import move_batch
from lsdb_ml.inference.outputs import OutputSpec
from lsdb_ml.inference.plugin import ModelPlugin, ModelSpec, deploy_locally, get_model, undeploy_locally
from lsdb_ml.inputs import InputsLike, model_args
from lsdb_ml.spec import ColumnSpec, ResolvedSpec

logger = logging.getLogger(__name__)

# (client id, plugin name) pairs registered from this process, so redeploying the same
# model to the same client does not rebuild it on every worker.
_REGISTERED: set[tuple[str, str]] = set()


def _active_client():
    try:
        from distributed import get_client

        return get_client()
    except ValueError:
        return None


class ModelDeployment:
    """A model installed on every Dask worker, ready to predict on catalogs.

    With a ``distributed`` client, a :class:`~lsdb_ml.inference.plugin.ModelPlugin` builds
    the model once on each worker, including workers that join later. Without one (the
    synchronous or threaded scheduler), the model is built once in this process.

    Use it as a context manager to remove the model from the workers afterwards::

        with ModelDeployment(ModelSpec(make_model, weights="model.pt")) as deployed:
            result = deployed.predict(catalog, spec, inputs="x", outputs={"z": "float32"}).compute()

    Parameters
    ----------
    model : ModelSpec or nn.Module
        The model to deploy. A module is wrapped with :meth:`ModelSpec.from_module`.
    client : distributed.Client, optional
        Client to deploy to. Defaults to the active client, if any.
    """

    def __init__(self, model: ModelSpec | nn.Module, client=None):
        self.spec = ModelSpec.from_module(model) if isinstance(model, nn.Module) else model
        if not isinstance(self.spec, ModelSpec):
            raise TypeError(f"Expected a ModelSpec or torch.nn.Module, got {type(model).__name__}.")
        self.name = f"lsdb-ml-model-{self.spec.token()}"
        self.client = client if client is not None else _active_client()
        self._closed = False
        if self.client is None:
            scheduler = dask.config.get("scheduler", None)
            if scheduler in ("processes", "multiprocessing"):
                raise RuntimeError(
                    "Inference without a distributed client needs the synchronous or threaded "
                    "scheduler; create a distributed Client to use multiple processes."
                )
            deploy_locally(self.spec, self.name)
        elif (self.client.id, self.name) not in _REGISTERED:
            self.client.register_plugin(ModelPlugin(self.spec, self.name), name=self.name)
            _REGISTERED.add((self.client.id, self.name))

    def predict(
        self,
        catalog,
        spec: ColumnSpec | ResolvedSpec,
        outputs: OutputSpec | Mapping[str, object] | Sequence[str],
        *,
        inputs: InputsLike | None = None,
        batch_size: int = 4096,
        append: bool = True,
    ):
        """Return a lazy catalog with the model's predictions for every row.

        See :func:`predict` for the parameters.
        """
        if self._closed:
            raise RuntimeError("This deployment is closed.")
        resolved = spec if isinstance(spec, ResolvedSpec) else spec.resolve(catalog)
        if isinstance(batch_size, bool) or int(batch_size) < 1:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}.")
        outputs = outputs if isinstance(outputs, OutputSpec) else OutputSpec(outputs)
        if inputs is None:
            if len(resolved.fields) != 1:
                raise ValueError(
                    f"The spec has several fields {list(resolved.fields)}; pass inputs= to say which "
                    "batch keys the model takes."
                )
            inputs = next(iter(resolved.fields))
        if append:
            clashes = sorted(set(outputs.columns) & set(catalog.columns))
            if clashes:
                raise ValueError(f"Output columns {clashes} already exist in the catalog; rename them.")

        meta = outputs.meta(catalog.meta, append)
        return catalog.map_partitions(
            _predict_partition,
            model_name=self.name,
            resolved=resolved,
            inputs=inputs,
            outputs=outputs,
            batch_size=int(batch_size),
            append=append,
            meta=meta,
        )

    def close(self) -> None:
        """Remove the model from the workers (or this process). Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        if self.client is None:
            undeploy_locally(self.name)
        else:
            self.client.unregister_worker_plugin(self.name)
            _REGISTERED.discard((self.client.id, self.name))

    def __enter__(self) -> ModelDeployment:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def predict(
    catalog,
    model: ModelSpec | nn.Module,
    spec: ColumnSpec | ResolvedSpec,
    outputs: OutputSpec | Mapping[str, object] | Sequence[str],
    *,
    inputs: InputsLike | None = None,
    batch_size: int = 4096,
    append: bool = True,
    client=None,
):
    """Run a model over every row of a catalog, returning a lazy catalog of predictions.

    The model is installed once per Dask worker (see :class:`ModelDeployment`) and stays
    installed so the returned catalog can be computed later; use
    :class:`ModelDeployment` directly to remove it when done.

    Each partition is processed in order, in batches of up to ``batch_size`` rows, under
    ``torch.inference_mode()``. Predictions are attached to the rows they were computed
    from by position within the partition, never by joining on the index, so the output
    is aligned with the input even when ``_healpix_29`` values repeat.

    Parameters
    ----------
    catalog : lsdb.Catalog
        The input catalog.
    model : ModelSpec or nn.Module
        The model, or how to build it on each worker.
    spec : ColumnSpec or ResolvedSpec
        Which columns become which tensors. Validated against ``catalog.meta`` now,
        before any data is read.
    outputs : OutputSpec, mapping or sequence
        Output column names and dtypes, e.g. ``{"z_phot": "float32"}``.
    inputs : str, sequence of str or callable, optional
        Batch key(s) passed positionally to the model, or ``callable(batch)``. Defaults to
        the spec's only field.
    batch_size : int, default 4096
        Maximum rows per forward pass.
    append : bool, default True
        Keep the input columns. With False, the result holds only the predictions (on the
        same index).
    client : distributed.Client, optional
        Client to deploy to. Defaults to the active client; without one, the model is
        built in this process and the catalog must be computed with the synchronous or
        threaded scheduler.

    Returns
    -------
    lsdb.Catalog
        Lazy catalog whose ``meta`` includes the output columns.
    """
    deployment = ModelDeployment(model, client=client)
    return deployment.predict(catalog, spec, outputs, inputs=inputs, batch_size=batch_size, append=append)


def _predict_partition(
    partition: pd.DataFrame,
    *,
    model_name: str,
    resolved: ResolvedSpec,
    inputs: InputsLike,
    outputs: OutputSpec,
    batch_size: int,
    append: bool,
) -> pd.DataFrame:
    """Predict on one partition. All state is local to this call except the read-only model."""
    if len(partition) == 0:
        empty = {name: np.empty(0) for name in outputs.columns}
        return outputs.build_frame(partition, empty, append)
    model, device = get_model(model_name)
    pieces: list[dict[str, np.ndarray]] = []
    with torch.inference_mode():
        for rows in iter_slices(partition, batch_size):
            batch = move_batch(resolved.collate(rows), device)
            pieces.append(outputs.to_columns(model(*model_args(batch, inputs)), len(rows)))
    columns = {name: np.concatenate([piece[name] for piece in pieces]) for name in outputs.columns}
    return outputs.build_frame(partition, columns, append)
