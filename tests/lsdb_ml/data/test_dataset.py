import itertools
import math
from collections import Counter

import numpy as np
import pytest
import torch
from distributed import Client
from torch.utils.data import DataLoader

from lsdb_ml import CatalogIterableDataset, ColumnSpec, Flat, Nested
from lsdb_ml.data import iter_slices

SPEC = ColumnSpec({"id": Flat("id", dtype=torch.int64), "lc": Nested("lc", ["t", "flux"])})


def collect_ids(dataset, num_workers=0):
    """Iterate a dataset through a DataLoader and return the ids of every batch."""
    loader = DataLoader(dataset, batch_size=None, num_workers=num_workers)
    return [batch["id"].tolist() for batch in loader]


def flatten(batches):
    """Concatenate per-batch id lists."""
    return [i for batch in batches for i in batch]


# Spawned DataLoader workers cost a few seconds each to start, so the multi-worker cases
# are a subset of the full grid.
@pytest.mark.parametrize(
    "num_workers,world_size,shuffle",
    [(0, 1, True), (0, 2, True), (0, 1, False), (0, 2, False), (2, 1, True), (2, 2, False)],
)
def test_every_row_exactly_once(nested_catalog, num_workers, world_size, shuffle):
    """Across DataLoader workers and (simulated) DDP ranks, no row is duplicated or dropped."""
    ids = []
    for rank in range(world_size):
        dataset = CatalogIterableDataset(
            nested_catalog, SPEC, batch_size=16, shuffle=shuffle, seed=3, rank=rank, world_size=world_size
        )
        ids += flatten(collect_ids(dataset, num_workers))
    assert sorted(ids) == list(range(500))


def test_index_key_matches_catalog(nested_catalog):
    """The _index tensor carries the catalog's _healpix_29 values, once each."""
    dataset = CatalogIterableDataset(nested_catalog, SPEC, batch_size=50, seed=0)
    index = np.concatenate([batch["_index"].numpy() for batch in dataset])
    expected = nested_catalog.compute(progress_bar=False).index.to_numpy()
    assert sorted(index.tolist()) == sorted(expected.tolist())


def test_epochs_reshuffle_reproducibly(nested_catalog):
    """Each epoch covers every row; order changes per epoch and is fixed by the seed."""

    def run(seed):
        dataset = CatalogIterableDataset(nested_catalog, SPEC, batch_size=16, seed=seed, epochs=2)
        return flatten(collect_ids(dataset))

    ids = run(seed=7)
    assert Counter(ids) == Counter({i: 2 for i in range(500)})
    first, second = ids[:500], ids[500:]
    assert sorted(first) == list(range(500)) and first != second
    assert run(seed=7) == ids
    assert run(seed=8) != ids


def test_set_epoch(nested_catalog):
    """set_epoch selects the epoch's permutation."""
    dataset = CatalogIterableDataset(nested_catalog, SPEC, batch_size=None, seed=1, epochs=2)
    two_epochs = flatten(collect_ids(dataset))
    dataset = CatalogIterableDataset(nested_catalog, SPEC, batch_size=None, seed=1, epochs=1)
    dataset.set_epoch(1)
    assert flatten(collect_ids(dataset)) == two_epochs[500:]


def test_batches_never_span_chunks(nested_catalog):
    """Batches are cut within one partition: sizes follow each partition's row count."""
    partition_of = {}
    sizes = []
    for i, partition in enumerate(nested_catalog.partitions):
        ids = partition.compute(progress_bar=False)["id"].tolist()
        partition_of.update(dict.fromkeys(ids, i))
        sizes.append(len(ids))
    dataset = CatalogIterableDataset(nested_catalog, SPEC, batch_size=7, seed=2, partitions_per_chunk=1)
    batches = collect_ids(dataset)
    assert all(len({partition_of[i] for i in batch}) == 1 for batch in batches)
    assert len(batches) == sum(math.ceil(n / 7) for n in sizes)


@pytest.mark.parametrize("partitions_per_chunk", [1, 3])
def test_batch_size_none_yields_whole_chunks(nested_catalog, partitions_per_chunk):
    """With batch_size=None every chunk is one batch."""
    dataset = CatalogIterableDataset(
        nested_catalog, SPEC, batch_size=None, seed=2, partitions_per_chunk=partitions_per_chunk
    )
    batches = collect_ids(dataset)
    assert len(batches) == math.ceil(nested_catalog.npartitions / partitions_per_chunk)
    assert sorted(flatten(batches)) == list(range(500))


def test_open_ended_stream(nested_catalog):
    """epochs=None keeps streaming; every pass covers every row once."""
    dataset = CatalogIterableDataset(nested_catalog, SPEC, batch_size=None, seed=4, epochs=None)
    per_epoch = nested_catalog.npartitions
    batches = [b["id"].tolist() for b in itertools.islice(iter(dataset), 3 * per_epoch)]
    for epoch in range(3):
        assert sorted(flatten(batches[epoch * per_epoch : (epoch + 1) * per_epoch])) == list(range(500))


def test_empty_partitions(nested_catalog):
    """Partitions emptied by a filter are skipped, and every remaining row arrives once."""
    filtered = nested_catalog.query("mag > 21")
    expected = sorted(filtered.compute(progress_bar=False)["id"].tolist())
    sizes = [len(p.compute(progress_bar=False)) for p in filtered.partitions]
    assert 0 in sizes, "the filter should empty some partitions"
    batches = collect_ids(CatalogIterableDataset(filtered, SPEC, batch_size=4, seed=0))
    assert all(batches)
    assert sorted(flatten(batches)) == expected


def test_more_shards_than_partitions(nested_catalog):
    """With more shards than partitions some shards are empty; coverage stays exact."""
    small = nested_catalog.partitions[[0, 1, 2]]
    expected = sorted(small.compute(progress_bar=False)["id"].tolist())
    ids = []
    with pytest.warns(UserWarning, match="some shards get no data"):
        for rank in range(5):
            dataset = CatalogIterableDataset(small, SPEC, batch_size=8, seed=0, rank=rank, world_size=5)
            ids += flatten(collect_ids(dataset))
    assert sorted(ids) == expected


def test_with_distributed_client(nested_catalog):
    """In the main process the stream uses the active distributed client."""
    with Client(processes=False, n_workers=2, threads_per_worker=1, dashboard_address=None):
        dataset = CatalogIterableDataset(nested_catalog, SPEC, batch_size=32, seed=0, partitions_per_chunk=4)
        assert sorted(flatten(collect_ids(dataset))) == list(range(500))


def test_validation(nested_catalog, unloadable_catalog):
    """Bad arguments and specs fail at construction, before any data is read."""
    with pytest.raises(ValueError, match="seed is required"):
        CatalogIterableDataset(unloadable_catalog, SPEC, rank=0, world_size=2)
    CatalogIterableDataset(unloadable_catalog, SPEC, rank=0, world_size=2, shuffle=False)
    with pytest.raises(ValueError, match="rank must be in"):
        CatalogIterableDataset(unloadable_catalog, SPEC, seed=0, rank=2, world_size=2)
    with pytest.raises(ValueError, match="batch_size"):
        CatalogIterableDataset(unloadable_catalog, SPEC, batch_size=0)
    with pytest.raises(ValueError, match="epochs"):
        CatalogIterableDataset(unloadable_catalog, SPEC, epochs=0)
    with pytest.raises(ValueError, match="partitions_per_chunk"):
        CatalogIterableDataset(unloadable_catalog, SPEC, partitions_per_chunk=0)
    with pytest.raises(ValueError, match="not in the catalog"):
        CatalogIterableDataset(unloadable_catalog, ColumnSpec({"x": Flat("nope")}))


def test_iter_slices_cover_frame(nested_frame):
    """Slices are in order and cover every row once."""
    slices = list(iter_slices(nested_frame, 64))
    assert [len(s) for s in slices] == [64] * 7 + [52]
    assert np.array_equal(np.concatenate([s["id"].to_numpy() for s in slices]), nested_frame["id"].to_numpy())
