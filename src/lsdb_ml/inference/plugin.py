"""Installing a model once per Dask worker."""

from __future__ import annotations

import copy
import hashlib
import logging
import threading
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
        File readable from every worker: a ``state_dict``, or a checkpoint written by
        :class:`~lsdb_ml.TrainSession` (its ``"model"`` entry is used).
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
            state = torch.load(self.weights, map_location=device, weights_only=True)
            if "model" in state and "lsdb_ml_version" in state:
                state = state["model"]
            model.load_state_dict(state)
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


# Models deployed without a distributed client, for the synchronous and threaded
# schedulers: built once in this process and shared by its threads.
_LOCAL_MODELS: dict[str, tuple[nn.Module, torch.device]] = {}
_LOCAL_LOCK = threading.Lock()


def deploy_locally(spec: ModelSpec, name: str) -> None:
    """Build the model in this process, if it is not already there."""
    with _LOCAL_LOCK:
        if name not in _LOCAL_MODELS:
            _LOCAL_MODELS[name] = spec.build()


def undeploy_locally(name: str) -> None:
    """Drop a model built by :func:`deploy_locally`."""
    with _LOCAL_LOCK:
        _LOCAL_MODELS.pop(name, None)


def get_model(name: str) -> tuple[nn.Module, torch.device]:
    """Return ``(model, device)`` for a deployed model, from inside a task.

    Raises
    ------
    RuntimeError
        If the model is not deployed where the task runs.
    """
    try:
        from distributed import get_worker

        worker = get_worker()
    except ValueError:
        worker = None
    if worker is not None:
        plugin = worker.plugins.get(name)
        if plugin is not None and plugin.model is not None:
            return plugin.model, plugin.device
    elif name in _LOCAL_MODELS:
        return _LOCAL_MODELS[name]
    where = f"worker {worker.address}" if worker is not None else "this process"
    raise RuntimeError(
        f"Model '{name}' is not deployed on {where}. The prediction catalog must be computed "
        "with the same Dask client that was active (or passed) when predict() was called, and "
        "before the deployment is closed."
    )
