import pytest
import torch

from lsdb_ml import ColumnSpec, Flat, Nested, SpecError, TensorInfo


def test_resolve_valid_spec_from_catalog_without_compute(unloadable_catalog):
    """A valid spec resolves against a catalog without computing any partition."""
    spec = ColumnSpec(
        {
            "x": Flat(["ra", "dec", "mag"]),
            "flag": Flat("flag", dtype=torch.bool),
            "lc": Nested("lc", ["t", "flux"], max_length=4, truncate="tail"),
            "flux": Nested("lc", "flux", max_length=8, truncate="head"),
        }
    )
    resolved = spec.resolve(unloadable_catalog)
    assert resolved.columns == ["ra", "dec", "mag", "flag", "lc"]
    assert resolved.describe() == {
        "x": TensorInfo(torch.float32, (None, 3)),
        "flag": TensorInfo(torch.bool, (None,)),
        "lc": TensorInfo(torch.float32, (None, None, 2)),
        "lc_mask": TensorInfo(torch.bool, (None, None)),
        "lc_lengths": TensorInfo(torch.int64, (None,)),
        "flux": TensorInfo(torch.float32, (None, None)),
        "flux_mask": TensorInfo(torch.bool, (None, None)),
        "flux_lengths": TensorInfo(torch.int64, (None,)),
        "_index": TensorInfo(torch.int64, (None,)),
    }


def test_unloadable_catalog_fixture_raises(unloadable_catalog):
    """Guard: the no-compute fixture really fails when a partition is computed."""
    with pytest.raises(Exception, match="no data should be read"):
        unloadable_catalog.partitions[0].compute(progress_bar=False)


def test_resolve_accepts_meta_frame(nested_catalog):
    """resolve() also accepts the meta frame directly."""
    resolved = ColumnSpec({"x": Flat("ra")}).resolve(nested_catalog.meta)
    assert resolved.columns == ["ra"]


def test_all_problems_reported_together(unloadable_catalog):
    """Every problem in a spec is reported in one error."""
    spec = ColumnSpec(
        {
            "missing": Flat("nope"),
            "nested_as_flat": Flat("lc"),
            "string": Flat("name"),
            "flat_as_nested": Nested("ra", "x"),
            "missing_sub": Nested("lc", ["flux", "nope"]),
            "string_sub": Nested("lc", "band"),
            "bad_trunc": Nested("lc", "flux", truncate="middle"),
            "bad_len": Nested("lc", "flux", max_length=0),
        }
    )
    with pytest.raises(SpecError) as err:
        spec.resolve(unloadable_catalog)
    message = str(err.value)
    for expected in [
        "column 'nope' is not in the catalog",
        "column 'lc' is nested",
        "'name' has non-numeric dtype",
        "column 'ra' is not nested",
        "has no sub-column 'nope'",
        "'lc.band' has non-numeric type",
        "truncate must be",
        "max_length must be a positive integer",
    ]:
        assert expected in message
    assert "not loaded" in message  # hint for missing columns on a catalog


def test_key_collisions(unloadable_catalog):
    """Output keys may not collide with each other or the index key."""
    with pytest.raises(SpecError, match="'lc_mask' from 'lc' collides"):
        ColumnSpec({"lc_mask": Flat("ra"), "lc": Nested("lc", "flux")}).resolve(unloadable_catalog)
    with pytest.raises(SpecError, match="'_index' from '_index' collides with the index"):
        ColumnSpec({"_index": Flat("ra")}).resolve(unloadable_catalog)
    ColumnSpec({"_index": Flat("ra")}, index_key=None).resolve(unloadable_catalog)


def test_non_field_and_empty_spec(unloadable_catalog):
    """Non-Field values and empty specs are rejected."""
    with pytest.raises(SpecError, match="expected a Field"):
        ColumnSpec({"x": "ra"}).resolve(unloadable_catalog)
    with pytest.raises(SpecError, match="no fields"):
        ColumnSpec({}).resolve(unloadable_catalog)


def test_unsupported_dtype(unloadable_catalog):
    """Torch dtypes without a numpy equivalent are rejected."""
    with pytest.raises(SpecError, match="no numpy equivalent"):
        ColumnSpec({"x": Flat("ra", dtype=torch.bfloat16)}).resolve(unloadable_catalog)


def test_non_integer_index(nested_frame):
    """The index key requires an integer index, and can be disabled."""
    frame = nested_frame.reset_index(drop=True).set_index("name")
    with pytest.raises(SpecError, match="needs an integer index"):
        ColumnSpec({"x": Flat("ra")}).resolve(frame)
    ColumnSpec({"x": Flat("ra")}, index_key=None).resolve(frame)


def test_bad_schema_type():
    """resolve() rejects objects that are neither catalogs nor frames."""
    with pytest.raises(TypeError, match="Expected an lsdb catalog"):
        ColumnSpec({"x": Flat("ra")}).resolve(42)
