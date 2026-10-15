"""Mapping model outputs to catalog columns."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import nested_pandas as npd
import numpy as np
import pandas as pd
import torch


class OutputSpec:
    """Names and dtypes of the columns a model's output becomes.

    The model's output for a batch of ``B`` rows must be a tensor of shape ``(B,)`` or
    ``(B, 1)`` for a single column, or ``(B, k)`` for ``k`` columns, in order.

    Parameters
    ----------
    columns : Mapping[str, dtype] or sequence of str
        Column name to pandas/numpy dtype (e.g. ``{"z": "float32"}``), or names that all
        use ``dtype``.
    dtype : default "float32"
        The dtype for columns given as a sequence of names.

    Examples
    --------
    >>> spec = OutputSpec(["p_star", "p_galaxy"])
    >>> spec.columns
    {'p_star': dtype('float32'), 'p_galaxy': dtype('float32')}
    >>> spec.to_columns(torch.tensor([[0.1, 0.9], [0.8, 0.2]]), n_rows=2)["p_galaxy"]
    array([0.9, 0.2], dtype=float32)
    """

    def __init__(self, columns: Mapping[str, object] | Sequence[str], dtype="float32"):
        if isinstance(columns, str):
            columns = [columns]
        if not isinstance(columns, Mapping):
            columns = dict.fromkeys(columns, dtype)
        if not columns:
            raise ValueError("OutputSpec needs at least one column.")
        self.columns = {name: pd.api.types.pandas_dtype(d) for name, d in columns.items()}

    def meta(self, input_meta: pd.DataFrame, append: bool) -> npd.NestedFrame:
        """Return the empty output frame (Dask ``meta``) for an input catalog's meta.

        Parameters
        ----------
        input_meta : pd.DataFrame
            The input catalog's meta.
        append : bool
            Keep the input columns before the new ones.

        Returns
        -------
        npd.NestedFrame
            Empty frame with the output columns and dtypes.
        """
        empty = {name: pd.Series(dtype=d).array for name, d in self.columns.items()}
        if append:
            result = input_meta.copy()
            for name, values in empty.items():
                result[name] = values
            return result
        return npd.NestedFrame(empty, index=input_meta.index[:0])

    def to_columns(self, output, n_rows: int) -> dict[str, np.ndarray]:
        """Convert one batch of model output to numpy columns, validating the shape.

        Parameters
        ----------
        output : torch.Tensor
            Model output for a batch.
        n_rows : int
            Rows in the batch; every column must have exactly this many values.

        Returns
        -------
        dict[str, np.ndarray]
            One 1-d array per output column.

        Raises
        ------
        ValueError
            If the output does not match the spec.
        """
        names = list(self.columns)
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Expected the model output to be a torch.Tensor, got {type(output).__name__}.")
        array = output.detach().to("cpu").numpy()
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2 or array.shape[1] != len(names):
            raise ValueError(
                f"Model returned shape {tuple(array.shape)}, but {len(names)} output column(s) "
                f"{names} were declared; expected ({n_rows}, {len(names)})."
            )
        arrays = {name: array[:, i] for i, name in enumerate(names)}
        for name, array in arrays.items():
            if len(array) != n_rows:
                raise ValueError(
                    f"Output '{name}' has {len(array)} values for a batch of {n_rows} rows; "
                    "the model must return exactly one prediction per input row."
                )
        return arrays

    def build_frame(
        self, partition: pd.DataFrame, columns: dict[str, np.ndarray], append: bool
    ) -> pd.DataFrame:
        """Attach prediction columns to their partition, by position.

        Parameters
        ----------
        partition : pd.DataFrame
            The input partition the predictions were computed from, in the same row order.
        columns : dict[str, np.ndarray]
            One array per output column, ``len(partition)`` values each.
        append : bool
            Keep the input columns.

        Returns
        -------
        pd.DataFrame
            The output partition, with ``partition``'s index.
        """
        for name, values in columns.items():
            if len(values) != len(partition):
                raise ValueError(
                    f"Output '{name}' has {len(values)} values for a partition of {len(partition)} rows."
                )
        # Assigning arrays (not Series) is positional: no index alignment, which would be
        # wrong for a non-unique _healpix_29 index.
        typed = {
            name: pd.Series(columns[name], dtype=d, copy=False).array for name, d in self.columns.items()
        }
        if append:
            result = partition.copy()
            for name, values in typed.items():
                result[name] = values
            return result
        result = npd.NestedFrame(index=partition.index)
        for name, values in typed.items():
            result[name] = values
        return result
