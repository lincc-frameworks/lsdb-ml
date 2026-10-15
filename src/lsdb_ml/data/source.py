"""Partition sources: the Dask-side producer that feeds the torch-side consumer."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator, Sequence
from typing import Protocol, runtime_checkable

import dask
import numpy as np
import pandas as pd
from torch.utils.data import get_worker_info

logger = logging.getLogger(__name__)


@runtime_checkable
class PartitionSource(Protocol):
    """Streams chunks of rows for a given set of partitions.

    This is the producer/consumer seam: everything downstream (slicing, collation, the
    training session) only sees the frames yielded here.
    """

    @property
    def npartitions(self) -> int:
        """Number of partitions available to stream."""

    def iter_chunks(self, partitions: Sequence[int], *, shuffle: bool, seed: int) -> Iterator[pd.DataFrame]:
        """Yield frames covering exactly the rows of ``partitions``, each row once."""


class CatalogStreamSource:
    """A :class:`PartitionSource` backed by lsdb's ``CatalogStream``.

    The requested partitions are selected as a sub-catalog and streamed with
    ``CatalogStream``, which keeps lsdb's one-chunk prefetch and, with ``shuffle``, its
    partition-order and within-chunk row shuffling.

    Parameters
    ----------
    catalog : lsdb.Catalog
        The catalog to stream.
    partitions_per_chunk : int, default 1
        Partitions stacked into each chunk.
    use_dask_client : bool, default True
        In the main process, stream through the active ``distributed`` client if there is
        one, so the next chunk is computed while the current one is consumed.
    scheduler_address : str, optional
        Connect to this Dask scheduler instead, from the main process and from every
        DataLoader worker process.
    worker_scheduler : str, default "synchronous"
        Dask scheduler used inside DataLoader worker processes when no
        ``scheduler_address`` is given. Workers run in parallel already, so the default
        avoids oversubscribing cores with a thread pool per worker.

    Notes
    -----
    Without ``shuffle``, ``CatalogStream`` streams partitions in reverse pixel order.
    """

    def __init__(
        self,
        catalog,
        partitions_per_chunk: int = 1,
        use_dask_client: bool = True,
        scheduler_address: str | None = None,
        worker_scheduler: str = "synchronous",
    ):
        if isinstance(partitions_per_chunk, bool) or int(partitions_per_chunk) < 1:
            raise ValueError(
                f"partitions_per_chunk must be a positive integer, got {partitions_per_chunk!r}."
            )
        self.catalog = catalog
        self.partitions_per_chunk = int(partitions_per_chunk)
        self.use_dask_client = use_dask_client
        self.scheduler_address = scheduler_address
        self.worker_scheduler = worker_scheduler
        self._owned_client = None

    @property
    def npartitions(self) -> int:
        """Number of partitions in the catalog."""
        return self.catalog.npartitions

    def __getstate__(self):
        # A Client cannot cross into DataLoader worker processes; each one reconnects.
        state = self.__dict__.copy()
        state["_owned_client"] = None
        return state

    def _client(self):
        if self.scheduler_address is not None:
            if self._owned_client is None:
                from distributed import Client

                self._owned_client = Client(self.scheduler_address, set_as_default=False)
            return self._owned_client
        if get_worker_info() is not None or not self.use_dask_client:
            return None
        try:
            from distributed import get_client

            return get_client()
        except (ImportError, ValueError):
            return None

    def iter_chunks(self, partitions: Sequence[int], *, shuffle: bool, seed: int) -> Iterator[pd.DataFrame]:
        """Stream the rows of ``partitions`` in chunks of ``partitions_per_chunk`` partitions.

        Parameters
        ----------
        partitions : Sequence[int]
            Partition indices to stream.
        shuffle : bool
            Shuffle partition order and rows within each chunk.
        seed : int
            Seed for the shuffle.

        Yields
        ------
        pd.DataFrame
            One chunk at a time (a ``NestedFrame`` for nested catalogs). Chunks may be
            empty when partitions are empty.
        """
        from lsdb.streams import CatalogStream

        partitions = np.asarray(partitions, dtype=np.int64)
        if len(partitions) == 0:
            return
        sub_catalog = self.catalog.partitions[partitions.tolist()]
        client = self._client()

        in_worker = get_worker_info() is not None
        config = (
            dask.config.set(scheduler=self.worker_scheduler)
            if in_worker and client is None
            else contextlib.nullcontext()
        )
        with config:
            yield from CatalogStream(
                sub_catalog,
                client=client,
                partitions_per_chunk=self.partitions_per_chunk,
                shuffle=shuffle,
                seed=seed,
            )
