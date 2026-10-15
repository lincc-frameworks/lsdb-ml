"""Installing a model once per Dask worker."""

from __future__ import annotations

import copy
import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cloudpickle
import torch
from distributed import WorkerPlugin
from torch import nn

from lsdb_ml.device import DeviceLike, resolve_device

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """How to build a model on a worker.

    Only the spec travels to the workers: each worker calls ``factory()``, loads
    ``weights`` itself, switches to ``eval()`` and moves the model to ``device``. Prefer a
    factory plus a weights path for large models, so the weights never pass through the
    Dask scheduler.

    Parameters
    ----------
    factory : callable
        Picklable ``factory() -> nn.Module``.
    weights : str or Path, optional
        A saved ``state_dict`` (``torch.save(model.state_dict(), path)``), readable from
        every worker.
    device : str or torch.device, default "cpu"
        Resolved on each worker; ``"auto"`` picks CUDA, then MPS, then CPU.
    torch_threads : int, optional
        Calls ``torch.set_num_threads`` on the worker, to avoid oversubscribing cores with
        Dask threads times torch intra-op threads. It affects the whole worker process.
    """

    factory: Callable[[], nn.Module]
    weights: str | Path | None = None
    device: DeviceLike = "cpu"
    torch_threads: int | None = None

    @classmethod
    def from_module(cls, module: nn.Module, device: DeviceLike = "cpu") -> ModelSpec:
        """Wrap an in-memory module; it is pickled into the plugin once per worker."""
        return cls(factory=_CopyFactory(copy.deepcopy(module).to("cpu")), device=device)

    def token(self) -> str:
        """A short content hash, used to name the worker plugin."""
        payload = cloudpickle.dumps((self.factory, str(self.weights), str(self.device), self.torch_threads))
        return hashlib.sha256(payload).hexdigest()[:16]

    def build(self) -> tuple[nn.Module, torch.device]:
        """Build the model for inference: weights loaded, ``eval()`` mode, on its device."""
        device = resolve_device(self.device)
        if self.torch_threads is not None:
            torch.set_num_threads(self.torch_threads)
        model = self.factory()
        if not isinstance(model, nn.Module):
            raise TypeError(f"The model factory returned {type(model).__name__}, not a torch.nn.Module.")
        if self.weights is not None:
            model.load_state_dict(torch.load(self.weights, map_location=device, weights_only=True))
        model.eval()
        return model.to(device), device


class _CopyFactory:
    """Picklable factory returning a fresh copy of a module."""

    def __init__(self, module: nn.Module):
        self.module = module

    def __call__(self) -> nn.Module:
        return copy.deepcopy(self.module)


class ModelPlugin(WorkerPlugin):
    """Dask worker plugin that builds a model once when a worker starts (or joins).

    Tasks look the model up with :func:`get_model`. Each worker has its own instance, so
    there is no state shared between workers, and tasks on one worker share only the
    read-only model.

    Parameters
    ----------
    spec : ModelSpec
        How to build the model.
    name : str
        Plugin name; tasks find the model by it.
    """

    def __init__(self, spec: ModelSpec, name: str):
        self.spec = spec
        self.name = name
        self.model = None
        self.device = None

    def setup(self, worker=None):
        """Build the model on this worker."""
        self.model, self.device = self.spec.build()
        logger.info("Loaded model plugin %s on %s (device %s).", self.name, worker, self.device)

    def teardown(self, worker=None):
        """Release the model."""
        self.model = None
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.empty_cache()


def get_model(name: str) -> tuple[nn.Module, torch.device]:
    """Return ``(model, device)`` for a deployed model, from inside a task.

    Raises
    ------
    RuntimeError
        If the model is not deployed where the task runs.
    """
    from distributed import get_worker

    try:
        worker = get_worker()
    except ValueError as err:
        raise RuntimeError(
            f"Model '{name}' can only be used inside a Dask worker. Compute the prediction "
            "catalog with the distributed Client it was deployed to."
        ) from err
    plugin = worker.plugins.get(name)
    if plugin is None or plugin.model is None:
        raise RuntimeError(
            f"Model '{name}' is not deployed on worker {worker.address}. The prediction catalog "
            "must be computed with the Dask client that was active (or passed) when predict() "
            "was called, and before the deployment is closed."
        )
    return plugin.model, plugin.device
