import numpy as np
import pytest

from lsdb_ml.data.sharding import ShardInfo, assign_partitions, shard_seed


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("num_shards", [1, 2, 3, 7, 40])
def test_shards_are_disjoint_and_complete(shuffle, num_shards):
    """Every partition goes to exactly one shard, for any shard count."""
    for epoch in range(3):
        parts = [
            assign_partitions(33, ShardInfo(i, num_shards), epoch, seed=5, shuffle=shuffle)
            for i in range(num_shards)
        ]
        combined = np.concatenate(parts)
        assert sorted(combined.tolist()) == list(range(33))


def test_shuffle_changes_by_epoch_and_is_reproducible():
    """Each epoch has its own permutation, determined by the seed."""
    shard = ShardInfo(0, 1)
    first = assign_partitions(33, shard, 0, seed=1, shuffle=True)
    assert np.array_equal(first, assign_partitions(33, shard, 0, seed=1, shuffle=True))
    assert not np.array_equal(first, assign_partitions(33, shard, 1, seed=1, shuffle=True))
    assert not np.array_equal(first, assign_partitions(33, shard, 0, seed=2, shuffle=True))


def test_shard_seed_varies():
    """Shards and epochs get different derived seeds."""
    seeds = {shard_seed(0, epoch, shard) for epoch in range(3) for shard in range(3)}
    assert len(seeds) == 9
