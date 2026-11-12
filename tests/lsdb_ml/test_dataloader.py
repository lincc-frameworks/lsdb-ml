import lsdb
import numpy as np
import pytest
import torch
from conftest import N_ROWS
from torch.utils.data import DataLoader

from lsdb_ml import CatalogDataLoader, CatalogDataset, collate_catalog


def collect(dataset, **loader_kwargs):
    """Iterate a dataset through a DataLoader and return all batches."""
    return list(DataLoader(dataset, batch_size=None, **loader_kwargs))


def all_ids(batches):
    """Concatenate the ``id`` column of all batches."""
    return torch.cat([batch["id"] for batch in batches]).numpy()


def epoch_order(catalog, seed, epoch):
    """The row order of one epoch, iterated with a plain DataLoader and an explicit epoch."""
    dataset = CatalogDataset(catalog[["id"]], batch_size=10, seed=seed)
    dataset.set_epoch(epoch)
    return all_ids(collect(dataset)).tolist()


def test_flat_columns(nested_catalog):
    """Flat columns become 1-D tensors with their numpy dtype, and every row appears once."""
    batches = collect(CatalogDataset(nested_catalog[["id", "x"]], batch_size=16, seed=0))

    batch = batches[0]
    assert set(batch) == {"id", "x"}
    assert batch["id"].dtype == torch.int64
    assert batch["x"].dtype == torch.float32
    assert torch.equal(batch["x"], batch["id"].float())
    assert sorted(all_ids(batches)) == list(range(N_ROWS))


def test_nested_columns_are_padded_with_mask(nested_catalog):
    """Nested fields are zero-padded to the longest row in the batch, with a matching mask."""
    batches = collect(CatalogDataset(nested_catalog[["id", "lc"]], batch_size=16, seed=0))

    for batch in batches:
        lc = batch["lc"]
        assert set(lc) == {"t", "flux", "mask"}
        lengths = batch["id"] % 4
        assert lc["mask"].shape == (len(batch["id"]), lengths.max())
        assert torch.equal(lc["mask"].sum(dim=1), lengths)
        for row, n in enumerate(lengths):
            assert torch.equal(lc["t"][row, :n], torch.arange(n, dtype=torch.float64))
            assert torch.all(lc["flux"][row, :n] == batch["id"][row])
            assert torch.all(lc["t"][row, n:] == 0)
            assert torch.all(lc["flux"][row, n:] == 0)


@pytest.mark.parametrize(("max_length", "width"), [(2, 2), (5, 5), ({"lc": 2}, 2)])
def test_nested_max_length_pads_and_truncates(nested_catalog, max_length, width):
    """With max_length, every batch has the same padded width and longer rows are truncated."""
    dataset = CatalogDataset(nested_catalog[["id", "lc"]], batch_size=16, seed=0, max_length=max_length)

    for batch in collect(dataset):
        lc = batch["lc"]
        kept = torch.clamp(batch["id"] % 4, max=width)
        assert lc["t"].shape == lc["flux"].shape == lc["mask"].shape == (len(batch["id"]), width)
        assert torch.equal(lc["mask"].sum(dim=1), kept)
        for row, n in enumerate(kept):
            assert torch.equal(lc["t"][row, :n], torch.arange(n, dtype=torch.float64))
            assert torch.all(lc["flux"][row, :n] == batch["id"][row])
            assert torch.all(lc["t"][row, n:] == 0)
            assert torch.all(lc["flux"][row, n:] == 0)


def test_max_length_dict_pads_unlisted_columns_per_batch(nested_catalog):
    """Nested columns missing from a max_length dict keep per-batch padding."""
    frame = nested_catalog.compute().iloc[:16]
    frame["lc2"] = frame["lc"]

    batch = collate_catalog(frame, max_length={"lc": 5})

    assert batch["lc"]["mask"].shape == (16, 5)
    assert batch["lc2"]["mask"].shape == (16, (batch["id"] % 4).max())


def test_max_length_errors(nested_catalog):
    """max_length must name nested columns, and cannot be combined with a custom collate_fn."""
    with pytest.raises(ValueError, match="lcs"):
        CatalogDataset(nested_catalog, batch_size=10, max_length={"lcs": 5})
    with pytest.raises(ValueError, match="not nested columns"):
        CatalogDataset(nested_catalog, batch_size=10, max_length={"x": 5})
    with pytest.raises(ValueError, match="collate_fn"):
        CatalogDataset(nested_catalog, batch_size=10, max_length=5, collate_fn=collate_catalog)


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("partitions_per_chunk", [1, 3])
@pytest.mark.parametrize("drop_last", [False, True])
def test_batch_sizes(nested_catalog, shuffle, partitions_per_chunk, drop_last):
    """All batches are full except a final partial batch, which drop_last removes."""
    dataset = CatalogDataset(
        nested_catalog[["id"]],
        batch_size=7,
        shuffle=shuffle,
        seed=0,
        partitions_per_chunk=partitions_per_chunk,
        drop_last=drop_last,
    )
    batches = collect(dataset)
    sizes = [len(batch["id"]) for batch in batches]

    expected = [7] * (N_ROWS // 7) + ([] if drop_last else [N_ROWS % 7])
    assert sizes == expected
    ids = all_ids(batches)
    assert len(set(ids)) == len(ids)
    if not drop_last:
        assert sorted(ids) == list(range(N_ROWS))


def test_shuffle_is_seeded(nested_catalog):
    """The same seed and epoch give the same order; another epoch or seed gives another order."""
    first = epoch_order(nested_catalog, seed=1, epoch=0)
    assert epoch_order(nested_catalog, seed=1, epoch=0) == first
    assert epoch_order(nested_catalog, seed=1, epoch=1) != first
    assert epoch_order(nested_catalog, seed=2, epoch=0) != first
    assert sorted(first) == list(range(N_ROWS))
    assert first != list(range(N_ROWS))


def test_no_shuffle_keeps_row_order_within_partitions(nested_catalog):
    """Without shuffling, rows of each partition stay together and in catalog order."""
    dataset = CatalogDataset(nested_catalog[["id"]], batch_size=10, shuffle=False)
    ids = all_ids(collect(dataset))

    assert sorted(ids) == list(range(N_ROWS))
    for i in range(nested_catalog.npartitions):
        partition_ids = nested_catalog.partitions[i].compute()["id"].to_numpy()
        start = np.flatnonzero(ids == partition_ids[0])[0]
        assert np.array_equal(ids[start : start + len(partition_ids)], partition_ids)


@pytest.mark.parametrize("context", ["fork", "spawn"])
def test_workers_yield_each_row_once(nested_catalog, context):
    """Partitions are split across DataLoader workers without duplicating or dropping rows."""
    dataset = CatalogDataset(nested_catalog[["id", "lc"]], batch_size=8, seed=0)
    batches = collect(dataset, num_workers=2, multiprocessing_context=context)

    assert sorted(all_ids(batches)) == list(range(N_ROWS))


def test_non_numeric_columns_raise():
    """The default collate rejects string columns up front, naming them."""
    catalog = lsdb.generate_catalog(100, 10, seed=1)
    with pytest.raises(TypeError, match=r"nested\.band"):
        CatalogDataset(catalog, batch_size=10)


def test_custom_collate_fn():
    """A custom collate_fn receives each batch as a NestedFrame and can handle any column."""
    catalog = lsdb.generate_catalog(100, 10, seed=1)

    def collate(frame):
        return {"id": torch.tensor(frame["id"].to_numpy()), "bands": frame["nested.band"].tolist()}

    batches = collect(CatalogDataset(catalog, batch_size=10, seed=0, collate_fn=collate))

    assert len(batches) == 10
    assert set(batches[0]) == {"id", "bands"}
    assert set().union(*(batch["bands"] for batch in batches)) <= {"g", "r"}


def test_loader_yields_unwrapped_batches(nested_catalog):
    """CatalogDataLoader does not add a leading batch dimension to the dataset's batches."""
    loader = CatalogDataLoader(CatalogDataset(nested_catalog[["id", "lc"]], batch_size=16, seed=0))
    batches = list(loader)

    assert batches[0]["id"].shape == (16,)
    assert batches[0]["lc"]["mask"].shape[0] == 16
    assert sorted(all_ids(batches)) == list(range(N_ROWS))


@pytest.mark.parametrize(
    "kwargs",
    [{"batch_size": 4}, {"shuffle": True}, {"drop_last": True}, {"collate_fn": collate_catalog}],
)
def test_loader_rejects_dataset_options(nested_catalog, kwargs):
    """Options that belong on the dataset raise; their default values are accepted."""
    dataset = CatalogDataset(nested_catalog[["id"]], batch_size=10)
    with pytest.raises(ValueError, match="set on the CatalogDataset"):
        CatalogDataLoader(dataset, **kwargs)
    CatalogDataLoader(dataset, batch_size=None, shuffle=False, drop_last=False, collate_fn=None)


def test_loader_needs_catalog_dataset():
    """CatalogDataLoader only wraps a CatalogDataset."""
    with pytest.raises(TypeError, match="CatalogDataset"):
        CatalogDataLoader([1, 2, 3])


def test_loader_advances_epoch(nested_catalog):
    """Each pass over the loader uses the next epoch's shuffle order."""
    dataset = CatalogDataset(nested_catalog[["id"]], batch_size=10, seed=1)
    loader = CatalogDataLoader(dataset)

    passes = [all_ids(list(loader)).tolist() for _ in range(3)]

    assert passes == [epoch_order(nested_catalog, seed=1, epoch=epoch) for epoch in range(3)]
    assert dataset.epoch == 2


def test_loader_respects_set_epoch(nested_catalog):
    """An epoch set by the user is used as is, and the loader continues from it."""
    dataset = CatalogDataset(nested_catalog[["id"]], batch_size=10, seed=1)
    loader = CatalogDataLoader(dataset)

    dataset.set_epoch(5)
    assert all_ids(list(loader)).tolist() == epoch_order(nested_catalog, seed=1, epoch=5)
    assert all_ids(list(loader)).tolist() == epoch_order(nested_catalog, seed=1, epoch=6)
    # Setting the epoch every pass, DistributedSampler style, is not double-advanced.
    dataset.set_epoch(7)
    list(loader)
    assert dataset.epoch == 7


def test_loader_advances_epoch_in_workers(nested_catalog):
    """Workers see the new epoch, since it is set before they copy the dataset."""
    dataset = CatalogDataset(nested_catalog[["id"]], batch_size=8, seed=0)
    loader = CatalogDataLoader(dataset, num_workers=2, multiprocessing_context="fork")

    first, second = all_ids(list(loader)), all_ids(list(loader))

    assert sorted(first) == sorted(second) == list(range(N_ROWS))
    assert first.tolist() != second.tolist()


def test_loader_warns_on_persistent_workers(nested_catalog):
    """Persistent workers never see the new epoch, so the loader warns."""
    dataset = CatalogDataset(nested_catalog[["id"]], batch_size=10, seed=0)
    with pytest.warns(UserWarning, match="persistent_workers"):
        CatalogDataLoader(dataset, num_workers=2, persistent_workers=True)
