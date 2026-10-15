"""Partition-level sharding across DDP ranks and DataLoader workers.

Every shard computes the same per-epoch permutation of partition indices from a shared
seed and takes a strided slice of it, so shards are disjoint and together cover every
partition exactly once per epoch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch.distributed as dist


@dataclass(frozen=True)
class ShardInfo:
    """Which slice of the partitions a consumer reads.

    Attributes
    ----------
    shard_id : int
        This consumer's shard, ``rank * num_workers + worker_id``.
    num_shards : int
        Total number of shards, ``world_size * num_workers``.
    """

    shard_id: int
    num_shards: int

    @classmethod
    def from_parts(cls, rank: int, world_size: int, worker_id: int, num_workers: int) -> ShardInfo:
        """Build the shard for a DDP rank and a DataLoader worker.

        Parameters
        ----------
        rank : int
            The DDP rank (0 when not distributed).
        world_size : int
            The DDP world size (1 when not distributed).
        worker_id : int
            The DataLoader worker id (0 in the main process).
        num_workers : int
            The number of DataLoader workers (1 in the main process).

        Returns
        -------
        ShardInfo
            The shard.

        Examples
        --------
        >>> ShardInfo.from_parts(rank=1, world_size=2, worker_id=0, num_workers=3)
        ShardInfo(shard_id=3, num_shards=6)
        """
        return cls(shard_id=rank * num_workers + worker_id, num_shards=world_size * num_workers)


def distributed_rank_and_world_size() -> tuple[int, int]:
    """Return ``(rank, world_size)`` from ``torch.distributed``, or ``(0, 1)`` if not initialized."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def assign_partitions(npartitions: int, shard: ShardInfo, epoch: int, seed: int, shuffle: bool) -> np.ndarray:
    """Return the partition indices a shard reads in a given epoch.

    Parameters
    ----------
    npartitions : int
        Number of partitions in the catalog.
    shard : ShardInfo
        The shard.
    epoch : int
        The epoch; with ``shuffle``, each epoch uses a different permutation.
    seed : int
        Base seed shared by every shard.
    shuffle : bool
        Whether to permute partitions before striding.

    Returns
    -------
    np.ndarray
        Partition indices for this shard.

    Examples
    --------
    >>> assign_partitions(5, ShardInfo(0, 2), epoch=0, seed=0, shuffle=False)
    array([0, 2, 4])
    """
    if shuffle:
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(npartitions)
    else:
        order = np.arange(npartitions)
    return order[shard.shard_id :: shard.num_shards]


def shard_seed(seed: int, epoch: int, shard_id: int) -> int:
    """Derive the seed one shard uses for its own ordering in one epoch.

    Parameters
    ----------
    seed : int
        Base seed.
    epoch : int
        The epoch.
    shard_id : int
        The shard.

    Returns
    -------
    int
        A 32-bit seed.
    """
    return int(np.random.SeedSequence([seed, epoch, shard_id]).generate_state(1)[0])
