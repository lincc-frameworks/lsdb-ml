import lsdb
import nested_pandas as npd
import numpy as np
import pandas as pd
import pytest


def make_nested_frame(n_rows: int = 500, seed: int = 0, max_length: int = 6) -> npd.NestedFrame:
    """Synthetic object table with a ragged nested light curve column.

    Light curve lengths range from 0 to ``max_length`` - 1, so some rows have empty
    (null) nested values.
    """
    rng = np.random.default_rng(seed)
    base = pd.DataFrame(
        {
            "id": np.arange(n_rows, dtype=np.int64),
            "ra": rng.uniform(0, 360, n_rows),
            "dec": rng.uniform(-60, 60, n_rows),
            "mag": rng.normal(20, 1, n_rows).astype(np.float32),
            "flag": rng.integers(0, 2, n_rows).astype(bool),
            "name": [f"obj{i}" for i in range(n_rows)],
        }
    )
    lengths = rng.integers(0, max_length, n_rows)
    n_flat = int(lengths.sum())
    flat = pd.DataFrame(
        {
            "t": rng.uniform(0, 100, n_flat),
            "flux": rng.normal(0, 1, n_flat).astype(np.float32),
            "band": rng.choice(["g", "r"], n_flat),
        },
        index=np.repeat(np.arange(n_rows), lengths),
    )
    return npd.NestedFrame(base).join_nested(flat, "lc")


@pytest.fixture(scope="session")
def nested_frame() -> npd.NestedFrame:
    """The in-memory frame behind ``nested_catalog``, indexed by row number."""
    return make_nested_frame()


@pytest.fixture(scope="session")
def nested_catalog(nested_frame) -> lsdb.Catalog:
    """A lazy catalog with many small partitions and a ragged nested column."""
    return lsdb.from_dataframe(nested_frame, partition_rows=40, catalog_name="toy")


@pytest.fixture
def unloadable_catalog(nested_catalog) -> lsdb.Catalog:
    """Same schema as ``nested_catalog``, but computing any partition raises."""

    def fail(_):
        raise AssertionError("A partition was computed, but no data should be read here.")

    return nested_catalog.map_partitions(fail, meta=nested_catalog.meta)
