"""Turning catalog columns into batches of tensors.

Inputs are named with plain column names. Flat columns use their name (``"mag"``), and
nested sub-columns use LSDB's dotted names (``"lc.flux"``). Whether a column is flat or
nested, and its dtype, are read from the catalog's ``meta``, so inputs can be checked
before any data is computed.

A batch is a ``dict[str, torch.Tensor]`` keyed by those same names:

- a flat column becomes a ``(B,)`` tensor;
- a nested sub-column becomes a ``(B, L)`` tensor, zero-padded to the longest row in the
  batch (after optional truncation);
- each nested column with selected sub-columns also gets ``"<column>._mask"`` (bool,
  ``(B, L)``, True where a value is real) and ``"<column>._lengths"`` (int64, ``(B,)``).
  Sub-columns of one nested column share offsets, so they share one mask.

Floating-point columns are cast to ``float_dtype`` (float32 by default). Integer columns
become int64 and boolean columns stay bool.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from nested_pandas.series.dtype import NestedDtype

from lsdb_ml.collate import TruncatePolicy, frame_to_array, pad_ragged

_TRUNCATE_POLICIES = ("raise", "head", "tail")

MASK_SUFFIX = "._mask"
LENGTHS_SUFFIX = "._lengths"


class InputError(ValueError):
    """Raised when requested input columns do not match a catalog's schema."""


def _kind(dtype) -> str | None:
    """Classify a pandas/numpy/pyarrow dtype as "float", "int", "bool", or None (unsupported)."""
    if isinstance(dtype, NestedDtype):
        return None
    if isinstance(dtype, pd.ArrowDtype):
        dtype = dtype.pyarrow_dtype
    if isinstance(dtype, pa.DataType):
        if pa.types.is_floating(dtype):
            return "float"
        if pa.types.is_integer(dtype):
            return "int"
        if pa.types.is_boolean(dtype):
            return "bool"
        return None
    if pd.api.types.is_bool_dtype(dtype):
        return "bool"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    if pd.api.types.is_integer_dtype(dtype):
        return "int"
    return None


def _torch_dtype(kind: str, float_dtype: torch.dtype) -> torch.dtype:
    return {"float": float_dtype, "int": torch.int64, "bool": torch.bool}[kind]


@dataclass(frozen=True)
class _Column:
    name: str
    kind: str


class InputPlan:
    """Validated inputs for a catalog: which columns become which tensors.

    Create it with :func:`plan_inputs`. It is picklable, so it can be shipped to Dask
    tasks.
    """

    def __init__(
        self,
        flat: list[_Column],
        nested: dict[str, list[_Column]],
        float_dtype: torch.dtype,
        max_length: dict[str, int | None],
        truncate: dict[str, TruncatePolicy],
    ):
        self.flat = flat
        self.nested = nested
        self.float_dtype = float_dtype
        #: Per nested column: longest sequence to keep (None for no limit), and the policy.
        self.max_length = max_length
        self.truncate = truncate

    def describe(self) -> dict[str, torch.dtype]:
        """Return the dtype of every tensor in a batch, keyed by batch key."""
        info = {c.name: _torch_dtype(c.kind, self.float_dtype) for c in self.flat}
        for parent, subs in self.nested.items():
            for sub in subs:
                info[f"{parent}.{sub.name}"] = _torch_dtype(sub.kind, self.float_dtype)
            info[parent + MASK_SUFFIX] = torch.bool
            info[parent + LENGTHS_SUFFIX] = torch.int64
        return info

    def to_tensors(self, frame: pd.DataFrame) -> dict[str, torch.Tensor]:
        """Convert the rows of ``frame`` to a batch of tensors.

        Row ``i`` of every tensor comes from row ``i`` of ``frame``.

        Parameters
        ----------
        frame : pd.DataFrame
            Rows with the planned columns, e.g. a slice of a catalog partition. An empty
            frame, such as ``catalog.meta``, gives zero-row tensors.

        Returns
        -------
        dict[str, torch.Tensor]
            One entry per key in :meth:`describe`.

        Raises
        ------
        ValueError
            If a value is missing, or a nested row is longer than its column's
            ``max_length`` with ``truncate="raise"``.
        """
        batch: dict[str, torch.Tensor] = {}
        for column in self.flat:
            dtype = _numpy_dtype(column.kind, self.float_dtype)
            batch[column.name] = torch.from_numpy(
                frame_to_array(frame[[column.name]], dtype, repr(column.name))[:, 0].copy()
            )
        for parent, subs in self.nested.items():
            series = frame[parent]
            # nest.len() counts null rows as 0; to_flat() returns every row's values
            # concatenated in row order. Both stay consistent across iloc slices.
            lengths = np.asarray(series.nest.len(), dtype=np.int64)
            flat_values = series.nest.to_flat([sub.name for sub in subs])
            for sub in subs:
                key = f"{parent}.{sub.name}"
                values = frame_to_array(
                    flat_values[[sub.name]], _numpy_dtype(sub.kind, self.float_dtype), repr(key)
                )
                padded, mask, kept = pad_ragged(
                    values,
                    lengths,
                    max_length=self.max_length[parent],
                    truncate=self.truncate[parent],
                    what=repr(key),
                )
                batch[key] = torch.from_numpy(np.ascontiguousarray(padded[..., 0]))
            batch[parent + MASK_SUFFIX] = torch.from_numpy(mask)
            batch[parent + LENGTHS_SUFFIX] = torch.from_numpy(kept)
        return batch


def _numpy_dtype(kind: str, float_dtype: torch.dtype) -> np.dtype:
    return torch.empty(0, dtype=_torch_dtype(kind, float_dtype)).numpy().dtype


def _valid_max_length(value) -> bool:
    return value is None or (isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _per_column(option, value, default, nested_columns, valid, describe, problems) -> dict:
    """Expand a single value or a ``{nested column: value}`` mapping to every nested input."""
    if not isinstance(value, Mapping):
        if not valid(value):
            problems.append(f"{option} must be {describe}, got {value!r}.")
        return dict.fromkeys(nested_columns, value)
    resolved = dict.fromkeys(nested_columns, default)
    for key, item in value.items():
        parent = str(key).partition(".")[0]
        if key in nested_columns:
            if valid(item):
                resolved[key] = item
            else:
                problems.append(f"{option}['{key}'] must be {describe}, got {item!r}.")
        elif parent in nested_columns:
            problems.append(
                f"{option} key '{key}': use the nested column name '{parent}'; its sub-columns share "
                "one mask, so they are truncated together."
            )
        else:
            problems.append(
                f"{option} key '{key}' is not a nested column among the inputs "
                f"(nested inputs: {sorted(nested_columns)})."
            )
    return resolved


def plan_inputs(
    schema,
    inputs: str | Sequence[str],
    *,
    float_dtype: torch.dtype = torch.float32,
    max_length: int | Mapping[str, int] | None = None,
    truncate: TruncatePolicy | Mapping[str, TruncatePolicy] = "raise",
) -> InputPlan:
    """Check input column names against a catalog's schema, without computing data.

    Parameters
    ----------
    schema : lsdb.Catalog or pd.DataFrame
        A catalog (its ``meta`` is used), or a frame with the same columns and dtypes.
    inputs : str or sequence of str
        Column names: ``"mag"`` for a flat column, ``"lc.flux"`` for a nested sub-column.
    float_dtype : torch.dtype, default torch.float32
        Dtype for floating-point columns.
    max_length : int or Mapping[str, int], optional
        Longest nested sequence to keep; see ``truncate``. An int applies to every nested
        input. A mapping sets it per nested column, keyed by the column name (``"lc"``,
        not ``"lc.flux"``, since sub-columns share one mask); unlisted columns have no
        limit.
    truncate : {"raise", "head", "tail"} or Mapping[str, str], default "raise"
        For nested rows longer than ``max_length``: raise, keep the first values, or keep
        the last values. A mapping sets it per nested column, like ``max_length``;
        unlisted columns use "raise".

    Returns
    -------
    InputPlan
        The validated plan.

    Raises
    ------
    InputError
        Listing every problem found.

    Examples
    --------
    >>> import nested_pandas as npd
    >>> nf = npd.NestedFrame({"a": [1.0, 2.0]}, index=[10, 11])
    >>> nf = nf.join_nested(npd.NestedFrame({"f": [1.0, 2.0, 3.0]}, index=[10, 10, 11]), "lc")
    >>> batch = plan_inputs(nf, ["a", "lc.f"]).to_tensors(nf)
    >>> batch["lc.f"]
    tensor([[1., 2.],
            [3., 0.]])
    >>> batch["lc._mask"]
    tensor([[ True,  True],
            [ True, False]])
    """
    meta = schema if isinstance(schema, pd.DataFrame) else getattr(schema, "meta", None)
    if not isinstance(meta, pd.DataFrame):
        raise TypeError(f"Expected an lsdb catalog or a DataFrame, got {type(schema).__name__}.")
    names = [inputs] if isinstance(inputs, str) else list(inputs)

    problems: list[str] = []
    if not names:
        problems.append("No input columns were given.")
    if float_dtype not in (torch.float16, torch.float32, torch.float64):
        problems.append(f"float_dtype must be torch.float16, float32 or float64, got {float_dtype}.")

    flat: list[_Column] = []
    nested: dict[str, list[_Column]] = {}
    for name in dict.fromkeys(names):  # drop duplicates, keep order
        if name in meta.columns:
            dtype = meta.dtypes[name]
            if isinstance(dtype, NestedDtype):
                problems.append(
                    f"'{name}' is a nested column; name its sub-columns instead, e.g. "
                    f"{[f'{name}.{sub}' for sub in list(dtype.column_dtypes)[:2]]}."
                )
            elif _kind(dtype) is None:
                problems.append(f"'{name}' has non-numeric dtype {dtype}.")
            else:
                flat.append(_Column(name, _kind(dtype)))
            continue
        parent, _, sub = name.partition(".")
        if not sub or parent not in meta.columns:
            problems.append(f"'{name}' is not a column of the catalog.")
        elif not isinstance(meta.dtypes[parent], NestedDtype):
            problems.append(f"'{name}': '{parent}' is not a nested column.")
        elif sub not in meta.dtypes[parent].column_dtypes:
            available = sorted(meta.dtypes[parent].column_dtypes)
            problems.append(
                f"'{name}': nested column '{parent}' has no sub-column '{sub}' (available: {available})."
            )
        elif _kind(meta.dtypes[parent].column_dtypes[sub]) is None:
            problems.append(f"'{name}' has non-numeric type {meta.dtypes[parent].column_dtypes[sub]}.")
        else:
            nested.setdefault(parent, []).append(_Column(sub, _kind(meta.dtypes[parent].column_dtypes[sub])))

    lengths_limits = _per_column(
        "max_length", max_length, None, nested, _valid_max_length, "a positive integer or None", problems
    )
    policies = _per_column(
        "truncate",
        truncate,
        "raise",
        nested,
        lambda v: v in _TRUNCATE_POLICIES,
        "'raise', 'head' or 'tail'",
        problems,
    )

    for parent, subs in nested.items():
        for suffix in (MASK_SUFFIX, LENGTHS_SUFFIX):
            if any(f"{parent}.{sub.name}" == parent + suffix for sub in subs):
                problems.append(
                    f"'{parent + suffix}' is reserved for the generated mask/lengths of '{parent}'."
                )

    if problems:
        hint = ""
        if any("is not a column" in p for p in problems) and getattr(schema, "all_columns", None) is not None:
            hint = f"\nThe catalog was loaded with columns {list(meta.columns)}; others may exist on disk."
        raise InputError("Input columns do not match the catalog:\n  - " + "\n  - ".join(problems) + hint)
    return InputPlan(flat, nested, float_dtype, lengths_limits, policies)
