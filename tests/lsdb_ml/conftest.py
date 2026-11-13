import lsdb
import nested_pandas as npd
import numpy as np
import pandas as pd
import pytest

N_ROWS = 100


@pytest.fixture
def nested_catalog():
    """A small catalog in 12 partitions, with a flat column and a nested light curve.

    Row ``id`` has ``x == id``, and ``id % 4`` light curve points with ``t == 0, 1, ...`` and
    ``flux == id``, so tests can check every value and the amount of padding.
    """
    rng = np.random.default_rng(0)
    ids = np.arange(N_ROWS)
    base = pd.DataFrame(
        {
            "id": ids,
            "ra": rng.uniform(0, 360, N_ROWS),
            "dec": rng.uniform(-60, 60, N_ROWS),
            "x": ids.astype(np.float32),
        }
    )
    lengths = ids % 4
    light_curves = pd.DataFrame(
        {
            "t": np.concatenate([np.arange(n, dtype=np.float64) for n in lengths]),
            "flux": np.repeat(ids, lengths).astype(np.float64),
        },
        index=np.repeat(ids, lengths),
    )
    frame = npd.NestedFrame(base).join_nested(light_curves, "lc")
    return lsdb.from_dataframe(frame, partition_rows=20, catalog_name="test")
