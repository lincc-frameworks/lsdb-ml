"""Column specifications: map flat and nested catalog columns to tensors.

A :class:`ColumnSpec` names the tensors a model needs and where each comes from. It is
checked against a catalog's ``meta`` (an empty ``NestedFrame`` that carries every dtype,
including nested sub-column dtypes) before any data is computed, and the resulting
:class:`ResolvedSpec` collates dataframe rows into a ``dict[str, torch.Tensor]``.

>>> import lsdb_ml
>>> spec = lsdb_ml.ColumnSpec({"x": lsdb_ml.Flat(["ra", "dec"]), "lc": lsdb_ml.Nested("lc", ["flux"])})
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from nested_pandas.series.dtype import NestedDtype

from lsdb_ml.collate import TruncatePolicy, frame_to_array, numpy_dtype, pad_ragged


class SpecError(ValueError):
    """Raised when a column specification does not match a catalog's schema."""


@dataclass(frozen=True)
class TensorInfo:
    """Description of one tensor a resolved spec produces.

    Attributes
    ----------
    dtype : torch.dtype
        The tensor dtype.
    shape : tuple
        The shape; ``None`` marks a dimension that varies by batch (always the first one).
    """

    dtype: torch.dtype
    shape: tuple[int | None, ...]


def _as_list(columns: str | Sequence[str]) -> list[str]:
    return [columns] if isinstance(columns, str) else list(columns)


def _is_numeric(dtype) -> bool:
    """True for pandas/numpy/pyarrow integer, float, and bool dtypes."""
    if isinstance(dtype, pa.DataType):
        return pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_boolean(dtype)
    if isinstance(dtype, pd.ArrowDtype):
        return _is_numeric(dtype.pyarrow_dtype)
    if isinstance(dtype, NestedDtype):
        return False
    try:
        return pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
    except TypeError:
        return False


class Field(ABC):
    """Base class for one entry of a :class:`ColumnSpec`.

    Subclasses implement :meth:`resolve`, which validates the field against catalog meta
    and returns a :class:`ResolvedField` that does the actual collation. New column
    representations (e.g. ragged tensors) are added as new ``Field`` types.
    """

    @abstractmethod
    def resolve(self, key: str, meta: pd.DataFrame, problems: list[str]) -> ResolvedField | None:
        """Validate against ``meta`` and build the collator.

        Parameters
        ----------
        key : str
            The output key this field was registered under.
        meta : pd.DataFrame
            The catalog meta (an empty frame with full dtypes).
        problems : list[str]
            Append a message for every problem found, instead of raising, so that all
            problems in a spec are reported together.

        Returns
        -------
        ResolvedField or None
            None if any problem was found.
        """


class ResolvedField(ABC):
    """A field validated against a schema, ready to collate frames."""

    #: Top-level columns this field reads.
    columns: list[str]

    @abstractmethod
    def describe(self) -> dict[str, TensorInfo]:
        """Return the tensors this field produces, keyed by output key."""

    @abstractmethod
    def collate(self, frame: pd.DataFrame) -> dict[str, torch.Tensor]:
        """Convert the rows of ``frame`` to tensors."""


@dataclass
class Flat(Field):
    """One or more flat (non-nested) numeric columns, stacked into a single tensor.

    Parameters
    ----------
    columns : str or sequence of str
        A single column name produces a ``(B,)`` tensor; a list produces ``(B, F)``.
    dtype : torch.dtype, default torch.float32
        Output dtype.

    Notes
    -----
    Missing values raise an error during collation; drop or fill them in the catalog first.
    """

    columns: str | Sequence[str]
    dtype: torch.dtype = torch.float32

    def resolve(self, key: str, meta: pd.DataFrame, problems: list[str]) -> ResolvedField | None:
        """Validate the columns against ``meta``; see :meth:`Field.resolve`."""
        columns = _as_list(self.columns)
        start = len(problems)
        if not columns:
            problems.append(f"'{key}': Flat needs at least one column.")
        for column in columns:
            if column not in meta.columns:
                problems.append(f"'{key}': column '{column}' is not in the catalog.")
            elif isinstance(meta.dtypes[column], NestedDtype):
                problems.append(f"'{key}': column '{column}' is nested; use Nested('{column}', [...]).")
            elif not _is_numeric(meta.dtypes[column]):
                problems.append(f"'{key}': column '{column}' has non-numeric dtype {meta.dtypes[column]}.")
        try:
            np_dtype = numpy_dtype(self.dtype)
        except TypeError as err:
            problems.append(f"'{key}': {err}")
        if len(problems) > start:
            return None
        return _ResolvedFlat(key, columns, isinstance(self.columns, str), self, np_dtype)


class _ResolvedFlat(ResolvedField):
    def __init__(self, key: str, columns: list[str], scalar: bool, spec: Flat, np_dtype: np.dtype):
        self.key = key
        self.columns = columns
        self.scalar = scalar
        self.spec = spec
        self.np_dtype = np_dtype

    def describe(self) -> dict[str, TensorInfo]:
        shape = (None,) if self.scalar else (None, len(self.columns))
        return {self.key: TensorInfo(self.spec.dtype, shape)}

    def collate(self, frame: pd.DataFrame) -> dict[str, torch.Tensor]:
        array = frame_to_array(frame[self.columns], self.np_dtype, f"'{self.key}' (columns {self.columns})")
        if self.scalar:
            array = array[:, 0].copy()
        return {self.key: torch.from_numpy(array)}


@dataclass
class Nested(Field):
    """Numeric sub-columns of a nested column, zero-padded to a rectangular tensor with a mask.

    For output key ``k`` this produces:

    - ``k``: values, ``(B, L)`` for a single sub-column or ``(B, L, F)`` for a list;
    - ``k_mask``: bool ``(B, L)``, True where a value is real rather than padding;
    - ``k_lengths``: int64 ``(B,)``, the number of real values per row.

    ``L`` is the longest row in the batch, after truncation. Rows whose nested value is
    null or empty have length 0. Padding is zero, which carries no meaning; always use
    the mask.

    Parameters
    ----------
    column : str
        The nested column.
    fields : str or sequence of str
        Sub-column name(s) to read.
    dtype : torch.dtype, default torch.float32
        Output dtype of the values.
    max_length : int, optional
        Longest sequence to keep; see ``truncate``.
    truncate : {"raise", "head", "tail"}, default "raise"
        What to do with rows longer than ``max_length``: raise, keep the first values, or
        keep the last values.

    Notes
    -----
    Missing values inside a list raise an error during collation.
    """

    column: str
    fields: str | Sequence[str]
    dtype: torch.dtype = torch.float32
    max_length: int | None = None
    truncate: TruncatePolicy = "raise"

    def resolve(self, key: str, meta: pd.DataFrame, problems: list[str]) -> ResolvedField | None:
        """Validate the nested column and sub-columns against ``meta``; see :meth:`Field.resolve`."""
        subcolumns = _as_list(self.fields)
        start = len(problems)
        if not subcolumns:
            problems.append(f"'{key}': Nested needs at least one sub-column.")
        if self.truncate not in ("raise", "head", "tail"):
            problems.append(f"'{key}': truncate must be 'raise', 'head' or 'tail', got {self.truncate!r}.")
        limit = self.max_length
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
            problems.append(f"'{key}': max_length must be a positive integer, got {limit!r}.")
        if self.column not in meta.columns:
            problems.append(f"'{key}': column '{self.column}' is not in the catalog.")
        elif not isinstance(meta.dtypes[self.column], NestedDtype):
            problems.append(f"'{key}': column '{self.column}' is not nested; use Flat('{self.column}').")
        else:
            available = meta.dtypes[self.column].column_dtypes
            for sub in subcolumns:
                if sub not in available:
                    problems.append(
                        f"'{key}': nested column '{self.column}' has no sub-column '{sub}' "
                        f"(available: {sorted(available)})."
                    )
                elif not _is_numeric(available[sub]):
                    problems.append(
                        f"'{key}': sub-column '{self.column}.{sub}' has non-numeric type {available[sub]}."
                    )
        try:
            np_dtype = numpy_dtype(self.dtype)
        except TypeError as err:
            problems.append(f"'{key}': {err}")
        if len(problems) > start:
            return None
        return _ResolvedNested(key, subcolumns, isinstance(self.fields, str), self, np_dtype)


class _ResolvedNested(ResolvedField):
    def __init__(self, key: str, subcolumns: list[str], scalar: bool, spec: Nested, np_dtype: np.dtype):
        self.key = key
        self.columns = [spec.column]
        self.subcolumns = subcolumns
        self.scalar = scalar
        self.spec = spec
        self.np_dtype = np_dtype

    def describe(self) -> dict[str, TensorInfo]:
        values_shape = (None, None) if self.scalar else (None, None, len(self.subcolumns))
        return {
            self.key: TensorInfo(self.spec.dtype, values_shape),
            f"{self.key}_mask": TensorInfo(torch.bool, (None, None)),
            f"{self.key}_lengths": TensorInfo(torch.int64, (None,)),
        }

    def collate(self, frame: pd.DataFrame) -> dict[str, torch.Tensor]:
        what = f"'{self.key}' (nested column '{self.spec.column}', fields {self.subcolumns})"
        series = frame[self.spec.column]
        # nest.len() counts null rows as 0, and to_flat() returns the values of all rows
        # concatenated in row order; both are consistent across iloc slices and concat.
        lengths = np.asarray(series.nest.len(), dtype=np.int64)
        values = frame_to_array(series.nest.to_flat(self.subcolumns), self.np_dtype, what)
        padded, mask, kept = pad_ragged(
            values, lengths, max_length=self.spec.max_length, truncate=self.spec.truncate, what=what
        )
        if self.scalar:
            padded = padded[..., 0]
        return {
            self.key: torch.from_numpy(np.ascontiguousarray(padded)),
            f"{self.key}_mask": torch.from_numpy(mask),
            f"{self.key}_lengths": torch.from_numpy(kept),
        }


@dataclass
class ColumnSpec:
    """Mapping from output keys to catalog columns.

    Parameters
    ----------
    fields : Mapping[str, Field]
        Output key to field, e.g. ``{"x": Flat(["ra", "dec"]), "lc": Nested("lc", "flux")}``.
    index_key : str or None, default "_index"
        Every batch also includes the frame index (``_healpix_29`` for LSDB catalogs) as an
        int64 ``(B,)`` tensor under this key, so batches can be traced back to rows. None
        disables it.

    Examples
    --------
    >>> import nested_pandas as npd
    >>> nf = npd.NestedFrame({"a": [1.0, 2.0]}, index=[10, 11])
    >>> nf = nf.join_nested(npd.NestedFrame({"f": [1.0, 2.0, 3.0]}, index=[10, 10, 11]), "lc")
    >>> resolved = ColumnSpec({"a": Flat("a"), "lc": Nested("lc", "f")}).resolve(nf)
    >>> batch = resolved.collate(nf)
    >>> batch["lc"]
    tensor([[1., 2.],
            [3., 0.]])
    >>> batch["lc_mask"]
    tensor([[ True,  True],
            [ True, False]])
    >>> batch["_index"]
    tensor([10, 11])
    """

    fields: Mapping[str, Field]
    index_key: str | None = "_index"

    def resolve(self, schema) -> ResolvedSpec:
        """Validate the spec against a catalog's schema without computing any data.

        Parameters
        ----------
        schema : lsdb.Catalog or pd.DataFrame
            A catalog (its ``meta`` is used) or any dataframe with the same columns and
            dtypes, such as ``catalog.meta`` or a computed partition.

        Returns
        -------
        ResolvedSpec
            The validated spec, which can collate frames.

        Raises
        ------
        SpecError
            Listing every problem found.
        """
        meta = schema.meta if hasattr(schema, "meta") and not isinstance(schema, pd.DataFrame) else schema
        if not isinstance(meta, pd.DataFrame):
            raise TypeError(f"Expected an lsdb catalog or a DataFrame, got {type(schema).__name__}.")
        problems: list[str] = []
        if not self.fields:
            problems.append("The spec has no fields.")
        resolved: dict[str, ResolvedField] = {}
        for key, spec_field in self.fields.items():
            if not isinstance(spec_field, Field):
                problems.append(f"'{key}': expected a Field such as Flat or Nested, got {type(spec_field)}.")
                continue
            result = spec_field.resolve(key, meta, problems)
            if result is not None:
                resolved[key] = result

        seen: dict[str, str] = {}
        if self.index_key is not None:
            seen[self.index_key] = "the index"
            if not pd.api.types.is_integer_dtype(meta.index.dtype):
                problems.append(
                    f"index_key={self.index_key!r} needs an integer index, but the index has dtype "
                    f"{meta.index.dtype}; pass index_key=None to disable it."
                )
        for key, result in resolved.items():
            for output_key in result.describe():
                if output_key in seen:
                    problems.append(
                        f"Output key '{output_key}' from '{key}' collides with {seen[output_key]}."
                    )
                seen[output_key] = f"'{key}'"

        if problems:
            hint = ""
            missing = [p for p in problems if "is not in the catalog" in p]
            all_columns = getattr(schema, "all_columns", None)
            if missing and all_columns is not None:
                hint = (
                    "\nSome columns may exist on disk but were not loaded; the catalog was opened "
                    f"with columns {list(meta.columns)}."
                )
            raise SpecError("Column spec does not match the catalog:\n  - " + "\n  - ".join(problems) + hint)
        return ResolvedSpec(resolved, self.index_key)


class ResolvedSpec:
    """A :class:`ColumnSpec` validated against a schema; collates frames into tensors.

    Create it with :meth:`ColumnSpec.resolve`. It is picklable, so it can be shipped to
    Dask tasks.
    """

    def __init__(self, fields: dict[str, ResolvedField], index_key: str | None):
        self.fields = fields
        self.index_key = index_key

    @property
    def columns(self) -> list[str]:
        """Top-level catalog columns needed for collation, in first-use order."""
        columns: list[str] = []
        for resolved in self.fields.values():
            columns.extend(c for c in resolved.columns if c not in columns)
        return columns

    def describe(self) -> dict[str, TensorInfo]:
        """Return the dtype and shape of every tensor a batch contains, keyed by output key."""
        info: dict[str, TensorInfo] = {}
        for resolved in self.fields.values():
            info.update(resolved.describe())
        if self.index_key is not None:
            info[self.index_key] = TensorInfo(torch.int64, (None,))
        return info

    def collate(self, frame: pd.DataFrame) -> dict[str, torch.Tensor]:
        """Convert the rows of ``frame`` to a batch of tensors.

        Parameters
        ----------
        frame : pd.DataFrame
            Rows to convert, e.g. a slice of a catalog partition. Row order is preserved:
            row ``i`` of every output tensor comes from row ``i`` of ``frame``.

        Returns
        -------
        dict[str, torch.Tensor]
            One entry per key in :meth:`describe`.
        """
        batch: dict[str, torch.Tensor] = {}
        for resolved in self.fields.values():
            batch.update(resolved.collate(frame))
        if self.index_key is not None:
            batch[self.index_key] = torch.from_numpy(np.array(frame.index.to_numpy(), dtype=np.int64))
        return batch
