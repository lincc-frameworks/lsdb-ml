"""Running a model over every row of an LSDB catalog."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from lsdb_ml.batch import InputPlan, plan_inputs
from lsdb_ml.collate import TruncatePolicy
from lsdb_ml.device import move_batch
from lsdb_ml.inference.outputs import OutputSpec
from lsdb_ml.inference.plugin import ModelPlugin, ModelSpec, get_model

logger = logging.getLogger(__name__)

# (client id, plugin name) pairs registered from this process, so redeploying the same
# model to the same client does not rebuild it on every worker.
_REGISTERED: set[tuple[str, str]] = set()

#: ``forward(model, batch) -> Tensor``: how to call the model on a batch dict.
Forward = Callable[[nn.Module, dict[str, torch.Tensor]], torch.Tensor]


def _active_client():
    from distributed import get_client

    try:
        return get_client()
    except ValueError as err:
        raise RuntimeError(
            "lsdb-ml inference needs a dask.distributed Client, e.g. Client(LocalCluster(...)) "
            "or Client(processes=False); none is active."
        ) from err


def _iter_row_slices(frame: pd.DataFrame, batch_size: int) -> Iterator[pd.DataFrame]:
    """Consecutive slices of at most ``batch_size`` rows, in order, covering ``frame`` once."""
    for start in range(0, len(frame), batch_size):
        yield frame.iloc[start : start + batch_size]


class ModelDeployment:
    """A model installed on every Dask worker, ready to predict on catalogs.

    A :class:`~lsdb_ml.inference.plugin.ModelPlugin` builds the model once on each worker
    of a ``dask.distributed`` client, including workers that join later.

    Use it as a context manager to remove the model from the workers afterwards::

        with ModelDeployment(ModelSpec(make_model, weights="model.pt")) as deployed:
            result = deployed.predict(catalog, ["mag", "lc.flux"], {"z": "float32"}).compute()

    Parameters
    ----------
    model : ModelSpec or nn.Module
        The model to deploy. A module is wrapped with :meth:`ModelSpec.from_module`.
    client : distributed.Client, optional
        Client to deploy to. Defaults to the active client.

    Raises
    ------
    RuntimeError
        If no client is given and none is active.
    """

    def __init__(self, model: ModelSpec | nn.Module, client=None):
        self.spec = ModelSpec.from_module(model) if isinstance(model, nn.Module) else model
        if not isinstance(self.spec, ModelSpec):
            raise TypeError(f"Expected a ModelSpec or torch.nn.Module, got {type(model).__name__}.")
        self.name = f"lsdb-ml-model-{self.spec.token()}"
        self.client = client if client is not None else _active_client()
        self._closed = False
        if (self.client.id, self.name) not in _REGISTERED:
            self.client.register_plugin(ModelPlugin(self.spec, self.name), name=self.name)
            _REGISTERED.add((self.client.id, self.name))

    def predict(
        self,
        catalog,
        inputs: str | Sequence[str],
        outputs: OutputSpec | Mapping[str, object] | Sequence[str],
        *,
        forward: Forward | None = None,
        batch_size: int = 4096,
        append: bool = True,
        float_dtype: torch.dtype = torch.float32,
        max_length: int | None = None,
        truncate: TruncatePolicy = "raise",
    ):
        """Return a lazy catalog with the model's predictions for every row.

        See :func:`predict` for the parameters.
        """
        if self._closed:
            raise RuntimeError("This deployment is closed.")
        plan = plan_inputs(catalog, inputs, float_dtype=float_dtype, max_length=max_length, truncate=truncate)
        plan.to_tensors(catalog.meta)  # dry run on the empty meta frame: no data is read
        if isinstance(batch_size, bool) or int(batch_size) < 1:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}.")
        if forward is not None and not callable(forward):
            raise TypeError(f"forward must be callable(model, batch), got {type(forward).__name__}.")
        outputs = outputs if isinstance(outputs, OutputSpec) else OutputSpec(outputs)
        if append:
            clashes = sorted(set(outputs.columns) & set(catalog.columns))
            if clashes:
                raise ValueError(f"Output columns {clashes} already exist in the catalog; rename them.")

        return catalog.map_partitions(
            _predict_partition,
            model_name=self.name,
            plan=plan,
            forward=forward,
            outputs=outputs,
            batch_size=int(batch_size),
            append=append,
            meta=outputs.meta(catalog.meta, append),
        )

    def close(self) -> None:
        """Remove the model from the workers. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        self.client.unregister_worker_plugin(self.name)
        _REGISTERED.discard((self.client.id, self.name))

    def __enter__(self) -> ModelDeployment:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def predict(
    catalog,
    model: ModelSpec | nn.Module,
    inputs: str | Sequence[str],
    outputs: OutputSpec | Mapping[str, object] | Sequence[str],
    *,
    forward: Forward | None = None,
    batch_size: int = 4096,
    append: bool = True,
    float_dtype: torch.dtype = torch.float32,
    max_length: int | None = None,
    truncate: TruncatePolicy = "raise",
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
    inputs : str or sequence of str
        Input columns: ``"mag"`` for a flat column, ``"lc.flux"`` for a nested sub-column.
        Checked against ``catalog.meta`` now, before any data is read. See
        :mod:`lsdb_ml.batch` for the tensors each produces.
    outputs : OutputSpec, mapping or sequence
        Output column names and dtypes, e.g. ``{"z_phot": "float32"}``.
    forward : callable, optional
        ``forward(model, batch) -> Tensor``, for models that do not take the batch dict
        directly, e.g. ``lambda m, b: m(b["mag"], b["lc.flux"], b["lc._mask"])``. By
        default the model is called as ``model(batch)``.
    batch_size : int, default 4096
        Maximum rows per forward pass.
    append : bool, default True
        Keep the input columns. With False, the result holds only the predictions (on the
        same index).
    float_dtype : torch.dtype, default torch.float32
        Dtype for floating-point input columns.
    max_length : int, optional
        Longest nested sequence to keep; see ``truncate``. Nested inputs are padded to the
        longest row in each batch, so one very long row makes the whole batch that wide.
    truncate : {"raise", "head", "tail"}, default "raise"
        For nested rows longer than ``max_length``: raise, keep the first values, or keep
        the last values.
    client : distributed.Client, optional
        Client to deploy to. Defaults to the active client; one is required.

    Returns
    -------
    lsdb.Catalog
        Lazy catalog whose ``meta`` includes the output columns.
    """
    deployment = ModelDeployment(model, client=client)
    return deployment.predict(
        catalog,
        inputs,
        outputs,
        forward=forward,
        batch_size=batch_size,
        append=append,
        float_dtype=float_dtype,
        max_length=max_length,
        truncate=truncate,
    )


def _predict_partition(
    partition: pd.DataFrame,
    *,
    model_name: str,
    plan: InputPlan,
    forward: Forward | None,
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
        for rows in _iter_row_slices(partition, batch_size):
            batch = move_batch(plan.to_tensors(rows), device)
            output = model(batch) if forward is None else forward(model, batch)
            pieces.append(outputs.to_columns(output, len(rows)))
    columns = {name: np.concatenate([piece[name] for piece in pieces]) for name in outputs.columns}
    return outputs.build_frame(partition, columns, append)
