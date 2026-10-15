"""Device resolution and batch movement helpers."""

from __future__ import annotations

from collections.abc import Mapping

import torch

DeviceLike = str | torch.device | None


def resolve_device(device: DeviceLike = "cpu") -> torch.device:
    """Turn a device request into a concrete ``torch.device``.

    Parameters
    ----------
    device : str, torch.device or None, default "cpu"
        ``"cpu"`` (or ``None``) for the CPU, ``"auto"`` to pick CUDA if available, then
        MPS, then CPU. Any other value (e.g. ``"cuda"``, ``"cuda:1"``, ``"mps"``, or a
        ``torch.device``) is passed to ``torch.device`` and checked for availability.

    Returns
    -------
    torch.device
        The resolved device.

    Raises
    ------
    RuntimeError
        If an explicitly requested accelerator is not available.

    Examples
    --------
    >>> resolve_device("cpu")
    device(type='cpu')
    >>> resolve_device(None)
    device(type='cpu')
    """
    if device is None:
        return torch.device("cpu")
    if isinstance(device, str) and device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Device {resolved} was requested, but CUDA is not available.")
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(f"Device {resolved} was requested, but MPS is not available.")
    return resolved


def move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move every tensor in a batch dict to ``device``.

    Non-tensor values are passed through unchanged.

    Parameters
    ----------
    batch : Mapping[str, torch.Tensor]
        A collated batch, as produced by :meth:`lsdb_ml.ResolvedSpec.collate`.
    device : torch.device
        The target device.

    Returns
    -------
    dict[str, torch.Tensor]
        A new dict with the tensors on ``device``.
    """
    non_blocking = device.type == "cuda"
    return {
        key: value.to(device, non_blocking=non_blocking) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
