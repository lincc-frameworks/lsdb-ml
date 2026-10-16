import nested_pandas as npd
import numpy as np
import pandas as pd
import pytest
import torch

from lsdb_ml import InputError, plan_inputs
from lsdb_ml.collate import pad_ragged


def reference_rows(frame, column, sub):
    """Per-row python reference: one array of sub-column values per row."""
    rows = []
    for value in frame[column].array:
        rows.append(value[sub].to_numpy(np.float64) if isinstance(value, pd.DataFrame) else np.zeros(0))
    return rows


def reference_pad(rows, width, keep="head"):
    """Pad reference rows to ``width`` the slow way."""
    padded = np.zeros((len(rows), width))
    mask = np.zeros((len(rows), width), dtype=bool)
    for i, row in enumerate(rows):
        row = row[:width] if keep == "head" else row[max(len(row) - width, 0) :]
        padded[i, : len(row)] = row
        mask[i, : len(row)] = True
    return padded, mask


@pytest.fixture
def frame(nested_catalog):
    """A frame mixing a sliced partition, a whole partition, and a concatenation."""
    parts = [nested_catalog.partitions[i].compute(progress_bar=False) for i in range(3)]
    return pd.concat([parts[0].iloc[3:], parts[1], parts[2].iloc[:5]])


# Validation against meta


def test_unloadable_catalog_fixture_raises(unloadable_catalog):
    """Guard: the no-compute fixture really fails when a partition is computed."""
    with pytest.raises(Exception, match="no data should be read"):
        unloadable_catalog.partitions[0].compute(progress_bar=False)


def test_plan_from_catalog_without_compute(unloadable_catalog):
    """Flat vs nested and dtypes are inferred from meta; nothing is computed."""
    plan = plan_inputs(unloadable_catalog, ["mag", "id", "flag", "lc.t", "lc.flux"])
    assert plan.describe() == {
        "mag": torch.float32,
        "id": torch.int64,
        "flag": torch.bool,
        "lc.t": torch.float32,
        "lc.flux": torch.float32,
        "lc._mask": torch.bool,
        "lc._lengths": torch.int64,
    }
    batch = plan.to_tensors(unloadable_catalog.meta)  # the empty meta frame is a valid dry run
    assert {k: v.shape[0] for k, v in batch.items()} == dict.fromkeys(plan.describe(), 0)


def test_all_problems_reported_together(unloadable_catalog):
    """Every problem is reported in one error."""
    with pytest.raises(InputError) as err:
        plan_inputs(unloadable_catalog, ["nope", "lc", "name", "lc.nope", "ra.x", "lc.band"], max_length=0)
    message = str(err.value)
    for expected in [
        "'nope' is not a column",
        "'lc' is a nested column; name its sub-columns",
        "'name' has non-numeric dtype",
        "has no sub-column 'nope'",
        "'ra' is not a nested column",
        "'lc.band' has non-numeric type",
        "max_length must be a positive integer",
        "others may exist on disk",
    ]:
        assert expected in message


def test_other_argument_errors(unloadable_catalog):
    """Bad truncate/float_dtype values, empty inputs and bad schemas are rejected."""
    with pytest.raises(InputError, match="truncate must be"):
        plan_inputs(unloadable_catalog, "mag", truncate="middle")
    with pytest.raises(InputError, match="float_dtype"):
        plan_inputs(unloadable_catalog, "mag", float_dtype=torch.int32)
    with pytest.raises(InputError, match="No input columns"):
        plan_inputs(unloadable_catalog, [])
    with pytest.raises(TypeError, match="Expected an lsdb catalog"):
        plan_inputs(42, "mag")


def test_reserved_mask_name():
    """A real sub-column named like the generated mask is rejected."""
    nf = npd.NestedFrame({"a": [1.0]}, index=[0]).join_nested(
        npd.NestedFrame({"_mask": [1.0]}, index=[0]), "lc"
    )
    with pytest.raises(InputError, match="reserved"):
        plan_inputs(nf, ["lc._mask"])


# Conversion


def test_flat_columns(frame):
    """Flat columns become (B,) tensors with inferred dtypes, in row order."""
    batch = plan_inputs(frame, ["ra", "mag", "id", "flag"]).to_tensors(frame)
    np.testing.assert_allclose(batch["ra"].numpy(), frame["ra"].to_numpy(np.float32))
    assert batch["ra"].dtype == torch.float32 and batch["ra"].shape == (len(frame),)
    np.testing.assert_array_equal(batch["id"].numpy(), frame["id"].to_numpy())
    assert batch["flag"].dtype == torch.bool
    assert all(t.numpy().flags.writeable for t in batch.values())


def test_float_dtype(frame):
    """float_dtype controls floating-point columns only."""
    batch = plan_inputs(frame, ["ra", "id", "lc.t"], float_dtype=torch.float64).to_tensors(frame)
    assert batch["ra"].dtype == torch.float64 and batch["lc.t"].dtype == torch.float64
    assert batch["id"].dtype == torch.int64


def test_nested_matches_reference(frame):
    """Padding matches a per-row reference, including empty and null nested rows."""
    batch = plan_inputs(frame, ["lc.t", "lc.flux"], float_dtype=torch.float64).to_tensors(frame)
    lengths = np.array([len(r) for r in reference_rows(frame, "lc", "t")])
    assert (lengths == 0).any(), "fixture should include empty / null nested rows"
    for sub in ("t", "flux"):
        expected, expected_mask = reference_pad(reference_rows(frame, "lc", sub), lengths.max())
        np.testing.assert_allclose(batch[f"lc.{sub}"].numpy(), expected)
        np.testing.assert_array_equal(batch["lc._mask"].numpy(), expected_mask)
    np.testing.assert_array_equal(batch["lc._lengths"].numpy(), lengths)


@pytest.mark.parametrize("keep", ["head", "tail"])
def test_truncation(frame, keep):
    """Rows over max_length keep their first or last values."""
    batch = plan_inputs(frame, "lc.t", float_dtype=torch.float64, max_length=2, truncate=keep).to_tensors(
        frame
    )
    rows = reference_rows(frame, "lc", "t")
    expected, expected_mask = reference_pad(rows, 2, keep=keep)
    np.testing.assert_allclose(batch["lc.t"].numpy(), expected)
    np.testing.assert_array_equal(batch["lc._mask"].numpy(), expected_mask)
    np.testing.assert_array_equal(batch["lc._lengths"].numpy(), np.minimum([len(r) for r in rows], 2))


def test_truncation_raises_by_default(frame):
    """Rows over max_length raise unless a truncate policy is given."""
    with pytest.raises(ValueError, match="over max_length=2"):
        plan_inputs(frame, "lc.t", max_length=2).to_tensors(frame)


def test_all_rows_empty_nested(frame):
    """A batch where every nested row is empty pads to width 0."""
    empty_rows = frame[np.asarray(frame["lc"].nest.len()) == 0]
    batch = plan_inputs(frame, "lc.flux").to_tensors(empty_rows)
    assert batch["lc.flux"].shape == (len(empty_rows), 0)
    assert not batch["lc._mask"].any()


def test_missing_values_raise():
    """Missing flat values and missing values inside nested lists raise."""
    nf = npd.NestedFrame({"a": [1.0, None, 3.0]}, index=[0, 1, 2])
    nf = nf.join_nested(npd.NestedFrame({"f": [1.0, None, 2.0]}, index=[0, 0, 2]), "lc")
    with pytest.raises(ValueError, match="1 row\\(s\\) with missing values"):
        plan_inputs(nf, "a").to_tensors(nf)
    with pytest.raises(ValueError, match="missing values"):
        plan_inputs(nf, "lc.f").to_tensors(nf)


def test_row_order_preserved(frame):
    """Row i of every tensor comes from row i of the frame, even for a shuffled frame."""
    shuffled = frame.sample(frac=1, random_state=1)
    batch = plan_inputs(shuffled, ["id", "lc.t"], float_dtype=torch.float64).to_tensors(shuffled)
    np.testing.assert_array_equal(batch["id"].numpy(), shuffled["id"].to_numpy())
    np.testing.assert_array_equal(batch["lc._lengths"].numpy(), np.asarray(shuffled["lc"].nest.len()))


def test_pad_ragged_length_mismatch():
    """Inconsistent lengths and values are rejected."""
    with pytest.raises(ValueError, match="lengths sum to 3"):
        pad_ragged(np.zeros((2, 1)), np.array([1, 2]))
