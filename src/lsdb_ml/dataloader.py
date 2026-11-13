"""Stream an LSDB catalog into PyTorch as batches of tensors."""

import warnings
from collections.abc import Callable, Iterator
from functools import partial
from typing import Any

import dask
import nested_pandas as npd
import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from lsdb import Catalog
from lsdb.streams import CatalogStream
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


def collate_catalog(frame: npd.NestedFrame, max_length: int | dict[str, int] | None = None) -> dict[str, Any]:
    """Convert a batch of catalog rows into a dictionary of tensors.

    This is the default ``collate_fn`` of `CatalogDataset`.

    - A flat column becomes a 1-D tensor of shape ``[B]``, with its numpy dtype preserved
      (e.g. ``float64`` stays ``float64``; cast in your model code if needed).
    - A nested column becomes a dictionary with one zero-padded ``[B, L]`` tensor per
      field, plus a boolean ``"mask"`` tensor of shape ``[B, L]`` that is True where values
      are real and False on padding. By default ``L`` is the longest row in this batch, so
      it changes from batch to batch. Set ``max_length`` to get the same ``L`` for every
      batch, e.g. for models with a fixed input size.

    For example, a frame with a flat column ``a`` and a nested column ``lc`` with fields
    ``t`` and ``flux`` collates to::

        {"a": Tensor[B], "lc": {"t": Tensor[B, L], "flux": Tensor[B, L], "mask": Tensor[B, L]}}

    Missing (null) values are not supported.

    Parameters
    ----------
    frame : nested_pandas.NestedFrame
        The rows of one batch.
    max_length : int, dict[str, int], or None
        Fixed padded length for nested columns. An int applies to every nested column; a
        dict maps nested column names to lengths, and unlisted columns are padded per
        batch. With a fixed length, **rows longer than it are truncated**: values past the
        first ``max_length`` of a row are dropped, and the mask covers only the kept values.
        If None, every nested column is padded to the longest row in the batch and never
        truncated.

    Returns
    -------
    dict[str, Any]
        Tensors keyed by column name, with nested columns as dictionaries of tensors.
    """
    lengths = max_length if isinstance(max_length, dict) else dict.fromkeys(frame.columns, max_length)
    return {name: _collate_column(frame[name], lengths.get(name)) for name in frame.columns}


def _collate_column(series: pd.Series, max_length: int | None) -> torch.Tensor | dict[str, torch.Tensor]:
    # Nested Tensor columns (a newer nested-pandas feature) would get their own branch here.
    if isinstance(series.dtype, npd.NestedDtype):
        return _collate_nested(series, max_length)
    # Copy, because arrow-backed arrays convert to read-only numpy views.
    return torch.tensor(_to_numpy(series))


def _collate_nested(series: pd.Series, max_length: int | None) -> dict[str, torch.Tensor]:
    lengths = series.nest.len()
    if max_length is None:
        max_length = int(lengths.max(initial=0))
    mask = np.arange(max_length) < lengths[:, None]
    # Position of each flat value within its row; values at or past max_length are truncated.
    starts = np.repeat(np.cumsum(lengths) - lengths, lengths)
    keep = np.arange(lengths.sum()) - starts < max_length
    out = {}
    for field in series.nest.columns:
        values = _to_numpy(series.nest[field])
        padded = np.zeros((len(series), max_length), dtype=values.dtype)
        # Flat values are stored row by row, which is the same order a boolean mask fills in.
        padded[mask] = values[keep]
        out[field] = torch.from_numpy(padded)
    out["mask"] = torch.from_numpy(mask)
    return out


def _to_numpy(series: pd.Series) -> np.ndarray:
    # Columns are usually pyarrow-backed; ask for the matching numpy dtype explicitly.
    dtype = getattr(series.dtype, "numpy_dtype", None)
    return series.to_numpy(dtype=dtype)


def _is_numeric_arrow_type(pa_type: pa.DataType) -> bool:
    return pa.types.is_integer(pa_type) or pa.types.is_floating(pa_type) or pa.types.is_boolean(pa_type)


def _check_max_length(meta: npd.NestedFrame, max_length: int | dict[str, int]) -> None:
    """Raise early if a ``max_length`` dict names a column that is not a nested column."""
    if not isinstance(max_length, dict):
        return
    nested = [name for name, dtype in meta.dtypes.items() if isinstance(dtype, npd.NestedDtype)]
    unknown = [name for name in max_length if name not in nested]
    if unknown:
        raise ValueError(
            f"max_length names {unknown}, which are not nested columns. Nested columns: {nested}."
        )


def _check_collatable(meta: npd.NestedFrame) -> None:
    """Raise early if the default collate cannot turn every column into a tensor."""
    bad = []
    for name, dtype in meta.dtypes.items():
        if isinstance(dtype, npd.NestedDtype):
            if "mask" in dtype.column_dtypes:
                raise ValueError(
                    f"Nested column '{name}' has a field named 'mask', which collides with the "
                    "padding mask produced by `collate_catalog`. Rename the field or pass a collate_fn."
                )
            bad += [
                f"{name}.{field}"
                for field, pa_type in dtype.column_dtypes.items()
                if not _is_numeric_arrow_type(pa_type)
            ]
        elif not (pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)):
            bad.append(name)
    if bad:
        raise TypeError(
            f"Columns {bad} are not numeric and cannot be converted to tensors. Select only numeric "
            "columns from the catalog first, or pass a custom collate_fn."
        )


class CatalogDataset(IterableDataset):
    """A PyTorch ``IterableDataset`` that streams batches from an LSDB catalog.

    Data is computed a few partitions at a time with `lsdb.streams.CatalogStream`, so
    the catalog is never fully materialized. The dataset yields ready-made batches (it
    batches and collates itself, which is much faster than collating row by row). Iterate
    it with `CatalogDataLoader`, which also advances the shuffle order every epoch::

        loader = CatalogDataLoader(CatalogDataset(catalog, batch_size=64, seed=0))

    A plain ``DataLoader(dataset, batch_size=None)`` works too, but then you must call
    `set_epoch` yourself. Without ``batch_size=None``, torch batches the batches and
    silently adds a leading dimension.

    **Batches.** Each batch is ``collate_fn`` applied to a `nested_pandas.NestedFrame` of
    ``batch_size`` rows; see `collate_catalog` for the default output. Each data stream
    (the main process, or each DataLoader worker) can end with a smaller final batch,
    unless ``drop_last=True``.

    **Shuffling.** A true global shuffle would need the whole catalog in memory, so
    shuffling is approximate, in two layers:

    1. The order of partitions is randomly permuted each epoch.
    2. Rows within each chunk of ``partitions_per_chunk`` partitions are shuffled.

    A batch is drawn from a single chunk, except that rows left over at the end of a chunk
    fill the start of the next batch. Since partitions are spatially contiguous, a batch
    is drawn from only a few regions of the sky. A larger ``partitions_per_chunk`` mixes
    more, at the cost of memory.

    With a ``seed``, the order is reproducible for a given epoch and number of workers.
    The epoch selects the order: `CatalogDataLoader` advances it on every pass, and with a
    plain ``DataLoader`` you call `set_epoch` before each epoch, as with
    ``torch.utils.data.DistributedSampler``. Otherwise every epoch repeats the same order.
    Without a ``seed``, a random one is drawn when the dataset is created.

    **Multiple workers.** With ``num_workers=k`` on the loader, each worker streams a disjoint
    subset of partitions, so every row is yielded exactly once per epoch. Workers compute
    their partitions in-process with dask's synchronous scheduler, because dask's default
    threaded scheduler can hang in forked worker processes. The ``client`` is used only when
    ``num_workers=0``. With ``persistent_workers=True``, the epoch does not reach the
    workers, so every epoch repeats the same order.

    Parameters
    ----------
    catalog : lsdb.Catalog
        The catalog to stream. Select the columns you need first (e.g.
        ``catalog[["a", "lc"]]``); with the default ``collate_fn`` all of them must be
        numeric.
    batch_size : int
        Number of rows per batch.
    shuffle : bool
        Whether to shuffle (see above). If False, every partition is streamed and
        batched without any shuffling.
    seed : int or None
        Seed for all shuffling.
    partitions_per_chunk : int
        Number of partitions computed together into one chunk.
    drop_last : bool
        Whether to drop the final batch of each stream if it is smaller than ``batch_size``.
    max_length : int, dict[str, int], or None
        Fixed padded length for nested columns, so every batch has the same shape. An int
        applies to every nested column; a dict sets it per nested column, e.g.
        ``{"lc": 100}``. **Longer rows are truncated.** None pads each batch to its longest
        row. See `collate_catalog`. Cannot be combined with ``collate_fn``.
    collate_fn : callable or None
        Function from a `nested_pandas.NestedFrame` of rows to a batch. Defaults to
        `collate_catalog`.
    client : dask.distributed.Client or None
        Dask client used to compute partitions when ``num_workers=0``. None computes with
        the default dask scheduler.

    Examples
    --------
    A minimal training loop on a small synthetic catalog:

    >>> import lsdb, torch
    >>> from lsdb_ml import CatalogDataLoader, CatalogDataset
    >>> catalog = lsdb.generate_catalog(200, 10, seed=1)[["a", "b"]]
    >>> loader = CatalogDataLoader(CatalogDataset(catalog, batch_size=32, seed=0))
    >>> model = torch.nn.Linear(1, 1)
    >>> optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    >>> for epoch in range(2):
    ...     for batch in loader:
    ...         x, y = batch["a"].float()[:, None], batch["b"].float()[:, None]
    ...         loss = torch.nn.functional.mse_loss(model(x), y)
    ...         optimizer.zero_grad()
    ...         loss.backward()
    ...         optimizer.step()
    """

    def __init__(
        self,
        catalog: Catalog,
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int | None = None,
        partitions_per_chunk: int = 1,
        drop_last: bool = False,
        max_length: int | dict[str, int] | None = None,
        collate_fn: Callable[[npd.NestedFrame], Any] | None = None,
        client=None,
    ) -> None:
        super().__init__()
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if collate_fn is None:
            _check_collatable(catalog.meta)
            _check_max_length(catalog.meta, max_length)
            # A partial (unlike a lambda) can be pickled to spawned DataLoader workers.
            collate_fn = partial(collate_catalog, max_length=max_length)
        elif max_length is not None:
            raise ValueError(
                "max_length only applies to the default collate_fn; handle it in your collate_fn instead."
            )
        self.catalog = catalog
        self.batch_size = batch_size
        self.shuffle = shuffle
        # Every worker must derive the same partition permutation, so they need a shared seed.
        self.seed = seed if seed is not None else int(np.random.SeedSequence().generate_state(1)[0])
        self.partitions_per_chunk = partitions_per_chunk
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.client = client
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch, which selects the shuffle order for the next iteration.

        Parameters
        ----------
        epoch : int
            The epoch number.
        """
        self.epoch = epoch

    def __iter__(self) -> Iterator[Any]:
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id, num_workers, client = 0, 1, self.client
        else:
            worker_id, num_workers, client = worker_info.id, worker_info.num_workers, None
            # This process belongs to this worker only, so changing the global config is safe.
            dask.config.set(scheduler="synchronous")

        npartitions = self.catalog.npartitions
        if self.shuffle:
            order = np.random.default_rng([self.seed, self.epoch]).permutation(npartitions)
        else:
            order = np.arange(npartitions)
        shard = order[worker_id::num_workers]
        if len(shard) == 0:
            return

        rng = np.random.default_rng([self.seed, self.epoch, worker_id])
        stream = CatalogStream(
            self.catalog.partitions[shard.tolist()],
            client=client,
            partitions_per_chunk=self.partitions_per_chunk,
            shuffle=self.shuffle,
            seed=int(rng.integers(2**31)),
        )
        yield from self._batches(stream)

    def _batches(self, chunks: Iterator[npd.NestedFrame]) -> Iterator[Any]:
        # CatalogStream already shuffles the rows of each chunk, so just cut them into batches.
        leftover = None
        for chunk in chunks:
            rows = chunk if leftover is None else pd.concat([leftover, chunk])
            n_full = len(rows) // self.batch_size * self.batch_size
            for start in range(0, n_full, self.batch_size):
                yield self.collate_fn(rows.iloc[start : start + self.batch_size])
            # Carry the rows that don't fill a batch over to the next chunk.
            leftover = rows.iloc[n_full:]

        if leftover is not None and len(leftover) > 0 and not self.drop_last:
            yield self.collate_fn(leftover)


class CatalogDataLoader(DataLoader):
    """A ``torch.utils.data.DataLoader`` for a `CatalogDataset`.

    `CatalogDataset` already batches and collates, so this loader sets ``batch_size=None``
    and rejects the options that belong on the dataset instead (``batch_size``,
    ``shuffle``, ``sampler``, ``batch_sampler``, ``collate_fn``, ``drop_last``).

    It also advances the dataset's epoch at the start of every pass after the first, so
    each epoch gets a new shuffle order. Calling `CatalogDataset.set_epoch` yourself still
    works (e.g. to resume at a given epoch): if the epoch was changed since the last pass,
    that epoch is used as is.

    Everything else (``num_workers``, ``pin_memory``, ``prefetch_factor``, ...) is passed to
    ``DataLoader`` unchanged. With ``persistent_workers=True`` the workers keep their copy
    of the dataset, so the epoch does not reach them and every epoch repeats the same order.

    Parameters
    ----------
    dataset : CatalogDataset
        The dataset to iterate.
    **kwargs
        Other ``torch.utils.data.DataLoader`` arguments.
    """

    _DATASET_OPTIONS = ("batch_size", "shuffle", "sampler", "batch_sampler", "collate_fn", "drop_last")

    def __init__(self, dataset: CatalogDataset, **kwargs) -> None:
        if not isinstance(dataset, CatalogDataset):
            raise TypeError(f"CatalogDataLoader needs a CatalogDataset, not {type(dataset).__name__}.")
        # Allow the default values, e.g. when a framework rebuilds the loader from its attributes.
        conflicts = [key for key in self._DATASET_OPTIONS if kwargs.pop(key, None)]
        if conflicts:
            raise ValueError(
                f"{conflicts} are set on the CatalogDataset, not the CatalogDataLoader, "
                "because the dataset already yields batches."
            )
        if kwargs.get("persistent_workers") and kwargs.get("num_workers", 0) > 0 and dataset.shuffle:
            warnings.warn(
                "With persistent_workers=True, the epoch does not reach the workers, "
                "so every epoch repeats the same shuffle order."
            )
        super().__init__(dataset, batch_size=None, **kwargs)
        self._last_epoch = None

    def __iter__(self):
        # Set the epoch before the workers start, since they copy the dataset at that point.
        if self._last_epoch is not None and self.dataset.epoch == self._last_epoch:
            self.dataset.set_epoch(self._last_epoch + 1)
        self._last_epoch = self.dataset.epoch
        return super().__iter__()
