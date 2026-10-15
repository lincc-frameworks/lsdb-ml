"""Saving and loading training checkpoints."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module of a ``DistributedDataParallel``/``DataParallel`` wrapper."""
    return (
        model.module if isinstance(model, (nn.parallel.DistributedDataParallel, nn.DataParallel)) else model
    )


def save_checkpoint(path: str | Path, model: nn.Module, step: Any = None, **counters: Any) -> Path:
    """Atomically write a checkpoint.

    The file is a ``torch.save`` dict with ``"model"`` (the state dict), ``"step"`` (the
    step's ``state_dict()`` if it has one) and any extra counters.

    Parameters
    ----------
    path : str or Path
        Destination file. Written to a temporary file first, then renamed, so a crash never
        leaves a truncated checkpoint.
    model : nn.Module
        The model; DDP wrappers are unwrapped.
    step : optional
        The training step; saved if it defines ``state_dict()``.
    **counters
        Extra values to store, e.g. ``batch_count=100``.

    Returns
    -------
    Path
        The checkpoint path.
    """
    from lsdb_ml import __version__

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": unwrap_model(model).state_dict(), "lsdb_ml_version": __version__, **counters}
    if step is not None and hasattr(step, "state_dict"):
        payload["step"] = step.state_dict()
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return path


def load_checkpoint(
    path: str | Path, model: nn.Module, step: Any = None, map_location="cpu"
) -> dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint` into ``model`` (and ``step``).

    Parameters
    ----------
    path : str or Path
        Checkpoint file.
    model : nn.Module
        Model to load weights into; DDP wrappers are unwrapped.
    step : optional
        Training step to restore, if it defines ``load_state_dict()``.
    map_location : default "cpu"
        Passed to ``torch.load``.

    Returns
    -------
    dict
        The remaining entries (counters such as ``batch_count``).
    """
    payload = torch.load(path, map_location=map_location, weights_only=True)
    unwrap_model(model).load_state_dict(payload.pop("model"))
    step_state = payload.pop("step", None)
    if step_state is not None and step is not None and hasattr(step, "load_state_dict"):
        step.load_state_dict(step_state)
    return payload
