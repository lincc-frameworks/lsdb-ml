import nested_pandas as npd
import numpy as np
import pandas as pd
import pytest
import torch

from lsdb_ml import ColumnSpec, Flat, Nested
from lsdb_ml.collate import pad_ragged


def reference_rows(frame, column, fields):
    """Per-row python reference: a list of (length, n_fields) arrays."""
    rows = []
    for value in frame[column].array:
        if value is None or not isinstance(value, pd.DataFrame):
            rows.append(np.zeros((0, len(fields))))
        else:
            rows.append(value[fields].to_numpy(dtype=np.float64))
    return rows


def reference_pad(rows, width, keep="head"):
    """Pad reference rows to ``width`` the slow way."""
    padded = np.zeros((len(rows), width, rows[0].shape[1] if rows else 0))
    mask = np.zeros((len(rows), width), dtype=bool)
    for i, row in enumerate(rows):
        row = row[:width] if keep == "head" else row[max(len(row) - width, 0) :]
        padded[i, : len(row)] = row
        mask[i, : len(row)] = True
    return padded, mask


@pytest.fixture
def frame(nested_catalog):
    """A frame mixing a sliced partition, a whole partition, and a concatenation."""
    # A computed partition, plus a slice and a concatenation, all exercise the
    # flat-values/offsets bookkeeping on non-trivial arrow layouts.
    parts = [nested_catalog.partitions[i].compute(progress_bar=False) for i in range(3)]
    return pd.concat([parts[0].iloc[3:], parts[1], parts[2].iloc[:5]])


def test_flat_columns(frame):
    """Flat columns stack into (B,) or (B, F) tensors with the requested dtype."""
    resolved = ColumnSpec(
        {"x": Flat(["ra", "dec", "mag"]), "mag": Flat("mag"), "flag": Flat("flag", dtype=torch.bool)}
    ).resolve(frame)
    batch = resolved.collate(frame)
    assert batch["x"].shape == (len(frame), 3) and batch["x"].dtype == torch.float32
    np.testing.assert_allclose(batch["x"].numpy(), frame[["ra", "dec", "mag"]].to_numpy(dtype=np.float32))
    assert batch["mag"].shape == (len(frame),)
    assert batch["flag"].dtype == torch.bool
    np.testing.assert_array_equal(batch["flag"].numpy(), frame["flag"].to_numpy(dtype=bool))
    np.testing.assert_array_equal(batch["_index"].numpy(), frame.index.to_numpy())
    assert all(t.numpy().flags.writeable for t in batch.values())


@pytest.mark.parametrize("fields", [["t", "flux"], ["flux"]])
def test_nested_matches_reference(frame, fields):
    """Vectorized nested padding matches a per-row reference, including empty rows."""
    resolved = ColumnSpec({"lc": Nested("lc", fields, dtype=torch.float64)}).resolve(frame)
    batch = resolved.collate(frame)
    rows = reference_rows(frame, "lc", fields)
    lengths = np.array([len(r) for r in rows])
    assert (lengths == 0).any(), "fixture should include empty / null nested rows"
    expected, expected_mask = reference_pad(rows, lengths.max())
    np.testing.assert_allclose(batch["lc"].numpy(), expected)
    np.testing.assert_array_equal(batch["lc_mask"].numpy(), expected_mask)
    np.testing.assert_array_equal(batch["lc_lengths"].numpy(), lengths)


def test_nested_single_field_is_2d(frame):
    """A single nested sub-column produces (B, L) values."""
    batch = ColumnSpec({"f": Nested("lc", "flux")}).resolve(frame).collate(frame)
    assert batch["f"].ndim == 2
    assert batch["f"].shape == batch["f_mask"].shape


@pytest.mark.parametrize("keep", ["head", "tail"])
def test_truncation(frame, keep):
    """Rows over max_length keep their first or last values."""
    resolved = ColumnSpec({"lc": Nested("lc", ["t"], max_length=2, truncate=keep, dtype=torch.float64)})
    batch = resolved.resolve(frame).collate(frame)
    rows = reference_rows(frame, "lc", ["t"])
    expected, expected_mask = reference_pad(rows, 2, keep=keep)
    np.testing.assert_allclose(batch["lc"].numpy(), expected)
    np.testing.assert_array_equal(batch["lc_mask"].numpy(), expected_mask)
    np.testing.assert_array_equal(batch["lc_lengths"].numpy(), np.minimum([len(r) for r in rows], 2))


def test_truncation_raises_by_default(frame):
    """Rows over max_length raise unless a truncate policy is given."""
    with pytest.raises(ValueError, match="over max_length=2"):
        ColumnSpec({"lc": Nested("lc", "t", max_length=2)}).resolve(frame).collate(frame)


def test_empty_frame(frame):
    """Zero-row frames collate to zero-row tensors."""
    resolved = ColumnSpec({"x": Flat(["ra", "dec"]), "lc": Nested("lc", ["t", "flux"])}).resolve(frame)
    batch = resolved.collate(frame.iloc[:0])
    assert batch["x"].shape == (0, 2)
    assert batch["lc"].shape == (0, 0, 2)
    assert batch["lc_mask"].shape == (0, 0)
    assert batch["_index"].shape == (0,)


def test_all_rows_empty_nested(frame):
    """A batch where every nested row is empty pads to width 0."""
    lengths = np.asarray(frame["lc"].nest.len())
    empty_rows = frame[lengths == 0]
    batch = ColumnSpec({"lc": Nested("lc", "flux")}).resolve(frame).collate(empty_rows)
    assert batch["lc"].shape == (len(empty_rows), 0)
    assert not batch["lc_mask"].any()


def test_null_values_raise():
    """Missing flat values and missing values inside nested lists raise."""
    nf = npd.NestedFrame({"a": [1.0, None, 3.0]}, index=[0, 1, 2])
    nf = nf.join_nested(npd.NestedFrame({"f": [1.0, None, 2.0]}, index=[0, 0, 2]), "lc")
    with pytest.raises(ValueError, match="1 row\\(s\\) with missing values"):
        ColumnSpec({"a": Flat("a")}).resolve(nf).collate(nf)
    with pytest.raises(ValueError, match="missing values"):
        ColumnSpec({"lc": Nested("lc", "f")}).resolve(nf).collate(nf)


def test_pad_ragged_length_mismatch():
    """Inconsistent lengths and values are rejected."""
    with pytest.raises(ValueError, match="lengths sum to 3"):
        pad_ragged(np.zeros((2, 1)), np.array([1, 2]))


def test_row_order_preserved(frame):
    """Row i of every tensor comes from row i of the frame, even for a shuffled frame."""
    shuffled = frame.sample(frac=1, random_state=1)
    batch = ColumnSpec({"id": Flat("id", dtype=torch.int64)}).resolve(shuffled).collate(shuffled)
    np.testing.assert_array_equal(batch["id"].numpy(), shuffled["id"].to_numpy())
    np.testing.assert_array_equal(batch["_index"].numpy(), shuffled.index.to_numpy())
