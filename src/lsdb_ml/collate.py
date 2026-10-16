"""Vectorized conversion of (nested) dataframe columns into padded numpy arrays.

These helpers do the numeric work behind :func:`lsdb_ml.batch.plan_inputs`. Nothing here
loops over rows in Python.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

TruncatePolicy = Literal["raise", "head", "tail"]


def frame_to_array(frame: pd.DataFrame, dtype: np.dtype, what: str) -> np.ndarray:
    """Convert the columns of ``frame`` into a writable ``(N, F)`` array.

    Parameters
    ----------
    frame : pd.DataFrame
        The columns to convert; may be arrow- or numpy-backed.
    dtype : np.dtype
        Output dtype.
    what : str
        Description of the data used in error messages.

    Returns
    -------
    np.ndarray
        A C-contiguous, writable array of shape ``(len(frame), len(frame.columns))``.

    Raises
    ------
    ValueError
        If any value is missing.
    """
    missing = frame.isna().to_numpy()
    if missing.any():
        n_missing = int(missing.any(axis=1).sum())
        raise ValueError(
            f"{what} contains {n_missing} row(s) with missing values. Drop or fill them before "
            "collation, e.g. with catalog.dropna() or catalog.map_partitions()."
        )
    # np.array (not np.asarray) always copies: arrow-backed columns can convert to read-only
    # views, which torch.from_numpy would warn about and share.
    return np.array(frame.to_numpy(dtype=dtype), dtype=dtype, order="C", copy=True)


def pad_ragged(
    values: np.ndarray,
    lengths: np.ndarray,
    *,
    max_length: int | None = None,
    truncate: TruncatePolicy = "raise",
    what: str = "nested column",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scatter flat ragged values into a zero-padded ``(B, L, F)`` array with a mask.

    Parameters
    ----------
    values : np.ndarray
        Flat values of shape ``(sum(lengths), F)``, rows in order.
    lengths : np.ndarray
        Number of values in each of the ``B`` rows.
    max_length : int or None, default None
        Longest sequence to keep. Longer rows are handled by ``truncate``.
    truncate : {"raise", "head", "tail"}, default "raise"
        For rows over ``max_length``: raise, keep the first values, or keep the last values.
    what : str, default "nested column"
        Description of the data used in error messages.

    Returns
    -------
    padded : np.ndarray
        Shape ``(B, L, F)``, where ``L`` is the longest kept row.
    mask : np.ndarray
        Boolean, shape ``(B, L)``; True where ``padded`` holds real data.
    kept_lengths : np.ndarray
        int64, shape ``(B,)``; number of real values per row after truncation.

    Examples
    --------
    >>> padded, mask, kept = pad_ragged(np.arange(5.0)[:, None], np.array([2, 0, 3]))
    >>> padded[..., 0]
    array([[0., 1., 0.],
           [0., 0., 0.],
           [2., 3., 4.]])
    >>> mask.astype(int)
    array([[1, 1, 0],
           [0, 0, 0],
           [1, 1, 1]])
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    n_rows = len(lengths)
    n_features = values.shape[1]
    if int(lengths.sum()) != len(values):
        raise ValueError(f"{what}: lengths sum to {int(lengths.sum())} but there are {len(values)} values.")

    starts = np.cumsum(lengths) - lengths
    kept = lengths
    source_starts = starts
    if max_length is not None and n_rows and int(lengths.max()) > max_length:
        if truncate == "raise":
            raise ValueError(
                f"{what} has rows of length up to {int(lengths.max())}, over max_length={max_length}. "
                "Pass truncate='head' or truncate='tail' to cut them."
            )
        kept = np.minimum(lengths, max_length)
        if truncate == "tail":
            source_starts = starts + lengths - kept
        elif truncate != "head":
            raise ValueError(f"truncate must be 'raise', 'head' or 'tail', got {truncate!r}.")

    width = int(kept.max()) if n_rows else 0
    padded = np.zeros((n_rows, width, n_features), dtype=values.dtype)
    mask = np.zeros((n_rows, width), dtype=bool)

    total = int(kept.sum())
    if total:
        row = np.repeat(np.arange(n_rows), kept)
        position = np.arange(total) - np.repeat(np.cumsum(kept) - kept, kept)
        source = np.repeat(source_starts, kept) + position
        padded[row, position] = values[source]
        mask[row, position] = True
    return padded, mask, kept.astype(np.int64)
