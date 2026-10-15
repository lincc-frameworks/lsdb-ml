"""Selecting model inputs from a collated batch."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

#: How to build the model's positional arguments from a batch: one key, a list of keys,
#: or a callable ``batch -> argument`` (a tuple return value is unpacked into arguments).
InputsLike = str | Sequence[str] | Callable[[Mapping[str, torch.Tensor]], Any]


def model_args(batch: Mapping[str, torch.Tensor], inputs: InputsLike) -> tuple:
    """Return the positional arguments to call a model with.

    Parameters
    ----------
    batch : Mapping[str, torch.Tensor]
        A collated batch.
    inputs : str, sequence of str, or callable
        A key, keys (passed positionally in order), or ``callable(batch)``.

    Returns
    -------
    tuple
        Positional arguments.

    Examples
    --------
    >>> batch = {"x": 1, "y": 2}
    >>> model_args(batch, "x"), model_args(batch, ["y", "x"]), model_args(batch, lambda b: (b["x"], 5))
    ((1,), (2, 1), (1, 5))
    """
    if isinstance(inputs, str):
        return (batch[inputs],)
    if callable(inputs):
        result = inputs(batch)
        return result if isinstance(result, tuple) else (result,)
    return tuple(batch[key] for key in inputs)
