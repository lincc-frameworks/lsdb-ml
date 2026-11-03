"""Stream an LSDB catalog into PyTorch as batches of tensors."""

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
from torch.utils.data import IterableDataset, get_worker_info


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
    batches and collates itself, which is much faster than collating row by row), so wrap
    it in a ``DataLoader`` with ``batch_size=None``::

        loader = DataLoader(CatalogDataset(catalog, batch_size=64, seed=0), batch_size=None)

    **Batches.** Each batch is ``collate_fn`` applied to a `nested_pandas.NestedFrame` of
    ``batch_size`` rows; see `collate_catalog` for the default output. Each data stream
    (the main process, or each DataLoader worker) can end with a smaller final batch,
    unless ``drop_last=True``.

    **Shuffling.** A true global shuffle would need the whole catalog in memory, so
    shuffling is approximate, in three layers:

    1. The order of partitions is randomly permuted each epoch.
    2. Rows within each chunk of ``partitions_per_chunk`` partitions are shuffled.
    3. A buffer of ``shuffle_buffer_size`` rows is carried from chunk to chunk and shuffled
       with each new chunk, so batches mix rows from consecutive chunks.

    Rows from partitions that are far apart in the streaming order never share a batch.
    Since partitions are spatially contiguous, a batch is drawn from only a few regions of
    the sky. A larger ``partitions_per_chunk`` or ``shuffle_buffer_size`` mixes more, at the
    cost of memory.

    With a ``seed``, the order is reproducible for a given epoch and number of workers.
    Call `set_epoch` before each epoch to get a new order, as with
    ``torch.utils.data.DistributedSampler``; otherwise every epoch repeats the same order.
    Without a ``seed``, a random one is drawn when the dataset is created.

    **Multiple workers.** With ``DataLoader(num_workers=k)``, each worker streams a disjoint
    subset of partitions, so every row is yielded exactly once per epoch. Workers compute
    their partitions in-process with dask's synchronous scheduler, because dask's default
    threaded scheduler can hang in forked worker processes. The ``client`` is used only when
    ``num_workers=0``. With ``persistent_workers=True``, `set_epoch` does not reach the
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
    shuffle_buffer_size : int
        Number of rows carried over between chunks to mix with the next chunk. 0 means
        rows only mix within a chunk. Ignored if ``shuffle`` is False.
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
    >>> from torch.utils.data import DataLoader
    >>> from lsdb_ml import CatalogDataset
    >>> catalog = lsdb.generate_catalog(200, 10, seed=1)[["a", "b"]]
    >>> dataset = CatalogDataset(catalog, batch_size=32, seed=0)
    >>> loader = DataLoader(dataset, batch_size=None)
    >>> model = torch.nn.Linear(1, 1)
    >>> optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    >>> for epoch in range(2):
    ...     dataset.set_epoch(epoch)
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
        shuffle_buffer_size: int = 1024,
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
        self.shuffle_buffer_size = shuffle_buffer_size if shuffle else 0
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
        yield from self._batches(stream, rng)

    def _batches(self, chunks: Iterator[npd.NestedFrame], rng: np.random.Generator) -> Iterator[Any]:
        buffer = None
        for chunk in chunks:
            buffer = chunk if buffer is None else pd.concat([buffer, chunk])
            if self.shuffle:
                buffer = buffer.iloc[rng.permutation(len(buffer))]
            # Emit full batches, keeping up to `shuffle_buffer_size` rows to mix with the next chunk.
            n_batches = max(len(buffer) - self.shuffle_buffer_size, 0) // self.batch_size
            for i in range(n_batches):
                yield self.collate_fn(buffer.iloc[i * self.batch_size : (i + 1) * self.batch_size])
            buffer = buffer.iloc[n_batches * self.batch_size :]

        if buffer is None:
            return
        for start in range(0, len(buffer), self.batch_size):
            rows = buffer.iloc[start : start + self.batch_size]
            if len(rows) < self.batch_size and self.drop_last:
                break
            yield self.collate_fn(rows)
