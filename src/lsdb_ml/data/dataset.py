"""An ``IterableDataset`` that streams collated batches from an LSDB catalog."""

from __future__ import annotations

import itertools
import logging
import warnings
from collections.abc import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, get_worker_info

from lsdb_ml.data.sharding import ShardInfo, assign_partitions, distributed_rank_and_world_size, shard_seed
from lsdb_ml.data.source import CatalogStreamSource, PartitionSource
from lsdb_ml.spec import ColumnSpec, ResolvedSpec

logger = logging.getLogger(__name__)


def iter_slices(frame: pd.DataFrame, batch_size: int | None) -> Iterator[pd.DataFrame]:
    """Cut one chunk into consecutive row slices; slices never span chunks.

    Parameters
    ----------
    frame : pd.DataFrame
        One chunk.
    batch_size : int or None
        Maximum rows per slice. None yields the whole chunk as one slice.

    Yields
    ------
    pd.DataFrame
        Non-empty slices, in row order, covering every row of ``frame`` once.

    Examples
    --------
    >>> [len(s) for s in iter_slices(pd.DataFrame({"a": range(7)}), 3)]
    [3, 3, 1]
    >>> [len(s) for s in iter_slices(pd.DataFrame({"a": []}), None)]
    []
    """
    if len(frame) == 0:
        return
    if batch_size is None:
        yield frame
        return
    for start in range(0, len(frame), batch_size):
        yield frame.iloc[start : start + batch_size]


class CatalogIterableDataset(IterableDataset):
    """Stream batches of tensors from an LSDB catalog for training.

    Each catalog partition is read by exactly one consumer: partitions are sharded across
    DDP ranks and DataLoader workers, so with any ``num_workers`` and world size every row
    appears exactly once per epoch. Batches are cut from the chunks the stream produces
    (one or more stacked partitions) and never span two chunks, so batch sizes are ragged.
    Use it with ``DataLoader(dataset, batch_size=None, num_workers=...)``: batches arrive
    already collated.

    Parameters
    ----------
    catalog : lsdb.Catalog
        The catalog to stream. It is projected to the columns the spec needs.
    spec : ColumnSpec or ResolvedSpec
        Which columns become which tensors. Validated against ``catalog.meta`` here,
        without computing data.
    batch_size : int or None, default None
        Maximum rows per batch. None yields each chunk as one batch.
    shuffle : bool, default True
        Shuffle partition order each epoch and rows within each chunk. Mixing is limited
        to rows within one chunk; raise ``partitions_per_chunk`` for more.
    seed : int, optional
        Base seed. If None, one is drawn at construction, which DataLoader workers then
        share. Required when ``shuffle`` is used with ``world_size > 1``.
    epochs : int or None, default 1
        Passes over the catalog per iteration. Each pass is reshuffled reproducibly. None
        streams forever.
    partitions_per_chunk : int, default 1
        Partitions stacked per chunk. Bigger chunks use more memory but give more Dask
        parallelism per fetch and better row mixing.
    rank, world_size : int, optional
        DDP rank and world size. Default to ``torch.distributed`` values when it is
        initialized, otherwise ``0`` and ``1``. They are read here rather than during
        iteration because ``torch.distributed`` is not initialized in DataLoader workers.
    use_dask_client, scheduler_address, worker_scheduler
        Passed to :class:`~lsdb_ml.data.source.CatalogStreamSource`.
    source : PartitionSource, optional
        Replace the default ``CatalogStreamSource``.

    Notes
    -----
    Every yielded batch is a ``dict[str, torch.Tensor]`` with the keys described by
    ``dataset.spec.describe()``, including the row index under ``"_index"`` by default.

    :meth:`set_epoch` takes effect at the next ``iter()``. With
    ``DataLoader(persistent_workers=True)`` workers keep their own copy of the dataset, so
    ``set_epoch`` does not reach them; use ``epochs`` instead.
    """

    def __init__(
        self,
        catalog,
        spec: ColumnSpec | ResolvedSpec,
        *,
        batch_size: int | None = None,
        shuffle: bool = True,
        seed: int | None = None,
        epochs: int | None = 1,
        partitions_per_chunk: int = 1,
        rank: int | None = None,
        world_size: int | None = None,
        use_dask_client: bool = True,
        scheduler_address: str | None = None,
        worker_scheduler: str = "synchronous",
        source: PartitionSource | None = None,
    ):
        super().__init__()
        self.spec = spec if isinstance(spec, ResolvedSpec) else spec.resolve(catalog)
        if batch_size is not None and (isinstance(batch_size, bool) or int(batch_size) < 1):
            raise ValueError(f"batch_size must be a positive integer or None, got {batch_size!r}.")
        if epochs is not None and (isinstance(epochs, bool) or int(epochs) < 1):
            raise ValueError(f"epochs must be a positive integer or None, got {epochs!r}.")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epochs = epochs
        self.epoch = 0

        default_rank, default_world_size = distributed_rank_and_world_size()
        self.rank = default_rank if rank is None else rank
        self.world_size = default_world_size if world_size is None else world_size
        if not 0 <= self.rank < self.world_size:
            raise ValueError(
                f"rank must be in [0, world_size), got rank={self.rank}, world_size={self.world_size}."
            )

        if seed is None:
            if shuffle and self.world_size > 1:
                raise ValueError(
                    "seed is required when shuffle=True and world_size > 1: every rank must "
                    "compute the same partition permutation, or shards would overlap."
                )
            seed = int(np.random.default_rng().integers(2**31))
        self.seed = int(seed)

        if source is None:
            source = CatalogStreamSource(
                catalog[self.spec.columns],
                partitions_per_chunk=partitions_per_chunk,
                use_dask_client=use_dask_client,
                scheduler_address=scheduler_address,
                worker_scheduler=worker_scheduler,
            )
        self.source = source

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch the next iteration starts from (like ``DistributedSampler.set_epoch``)."""
        self.epoch = int(epoch)

    def shard(self) -> ShardInfo:
        """Return the shard for the current process (call during iteration in workers)."""
        info = get_worker_info()
        worker_id, num_workers = (0, 1) if info is None else (info.id, info.num_workers)
        return ShardInfo.from_parts(self.rank, self.world_size, worker_id, num_workers)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        shard = self.shard()
        npartitions = self.source.npartitions
        if shard.num_shards > npartitions:
            warnings.warn(
                f"There are {shard.num_shards} shards (ranks x DataLoader workers) but only "
                f"{npartitions} partitions, so some shards get no data in an epoch. Use fewer "
                "workers or a catalog with more partitions.",
                stacklevel=2,
            )
        epochs = (
            itertools.count(self.epoch)
            if self.epochs is None
            else range(self.epoch, self.epoch + self.epochs)
        )
        for epoch in epochs:
            partitions = assign_partitions(npartitions, shard, epoch, self.seed, self.shuffle)
            if len(partitions) == 0 and self.epochs is None and not self.shuffle:
                return  # this shard never gets data; don't spin forever
            logger.debug("Shard %s epoch %d reads partitions %s", shard, epoch, partitions)
            chunks = self.source.iter_chunks(
                partitions, shuffle=self.shuffle, seed=shard_seed(self.seed, epoch, shard.shard_id)
            )
            for chunk in chunks:
                for rows in iter_slices(chunk, self.batch_size):
                    yield self.spec.collate(rows)
