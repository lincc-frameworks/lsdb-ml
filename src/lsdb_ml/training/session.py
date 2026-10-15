"""A context-manager training session that drives the per-batch loop."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from lsdb_ml.device import DeviceLike, move_batch, resolve_device
from lsdb_ml.training.checkpoint import load_checkpoint, save_checkpoint, unwrap_model
from lsdb_ml.training.steps import TrainStep

logger = logging.getLogger(__name__)

Callback = Callable[["TrainSession", Mapping[str, torch.Tensor], Mapping[str, Any]], None]


def batch_rows(batch: Mapping[str, Any]) -> int:
    """Number of rows in a collated batch (the first dimension of its first tensor)."""
    for value in batch.values():
        if isinstance(value, torch.Tensor):
            return int(value.shape[0]) if value.dim() else 1
    return 0


class TrainSession:
    """Train a model on a stream of batches, one batch at a time.

    Streams have no length and, in open-ended mode, no epochs, so instead of an epoch
    loop the session hands control back after every batch::

        with TrainSession(model, step, data=loader, checkpoint_dir="ckpt") as session:
            for batch, metrics in session:      # metrics is None for a skipped batch
                if metrics and metrics["loss"] < 1e-3:
                    session.stop()
        model = session.close()                 # already closed by the with block

    Parameters
    ----------
    model : nn.Module
        The model; moved to ``device``. May be wrapped in ``DistributedDataParallel``, in
        which case iteration runs under ``model.join()`` so ranks with fewer batches do
        not hang the others.
    step : TrainStep
        Callable ``step(model, batch) -> metrics`` that performs one optimization step,
        e.g. :class:`~lsdb_ml.training.SupervisedStep`.
    data : iterable of batches, optional
        Usually a ``DataLoader`` over a :class:`~lsdb_ml.data.CatalogIterableDataset`.
        Without it, drive the session manually with :meth:`train_batch`.
    device : str or torch.device, default "cpu"
        Where the model and batches go; ``"auto"`` picks CUDA, then MPS, then CPU.
    checkpoint_dir : str or Path, optional
        Where checkpoints are written. None disables checkpointing.
    checkpoint_every : int, optional
        Write ``checkpoint-<batch_count>.pt`` every this many trained batches.
    keep_last : int, optional
        Keep only this many periodic checkpoints. The final checkpoint is always kept.
    min_batch_size : int, optional
        Skip batches with fewer rows, e.g. 2 for models with BatchNorm. Empty batches are
        always skipped.
    callbacks : list of callables, optional
        Each is called as ``callback(session, batch, metrics)`` after every trained batch;
        use them for logging (MLflow, TensorBoard, ...).
    resume_from : str or Path, optional
        Checkpoint to restore model weights, step state and ``batch_count`` from. The data
        position is not restored.

    Notes
    -----
    Every trained batch is one optimizer step, whatever its size, so small batches (e.g.
    the last slice of a chunk) weigh as much as full ones. Use ``min_batch_size`` to skip
    them. Under DDP only rank 0 writes checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        step: TrainStep,
        data: Iterable[Mapping[str, torch.Tensor]] | None = None,
        *,
        device: DeviceLike = "cpu",
        checkpoint_dir: str | Path | None = None,
        checkpoint_every: int | None = None,
        keep_last: int | None = None,
        min_batch_size: int | None = None,
        callbacks: Iterable[Callback] = (),
        resume_from: str | Path | None = None,
    ):
        if checkpoint_every is not None and checkpoint_dir is None:
            raise ValueError("checkpoint_every needs a checkpoint_dir.")
        for name, value in (
            ("checkpoint_every", checkpoint_every),
            ("keep_last", keep_last),
            ("min_batch_size", min_batch_size),
        ):
            if value is not None and (isinstance(value, bool) or int(value) < 1):
                raise ValueError(f"{name} must be a positive integer or None, got {value!r}.")
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.step = step
        self.data = data
        self.checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
        self.checkpoint_every = checkpoint_every
        self.keep_last = keep_last
        self.min_batch_size = min_batch_size
        self.callbacks = list(callbacks)
        self.batch_count = 0
        self.rows_seen = 0
        self._stop_requested = False
        self._closed = False
        self._periodic_checkpoints: list[Path] = []
        if resume_from is not None:
            counters = load_checkpoint(resume_from, self.model, self.step, map_location=self.device)
            self.batch_count = int(counters.get("batch_count", 0))
            self.rows_seen = int(counters.get("rows_seen", 0))

    def __iter__(self) -> Iterator[tuple[Mapping[str, torch.Tensor], Mapping[str, Any] | None]]:
        """Train on every batch from ``data``, yielding ``(batch, metrics)`` after each."""
        if self.data is None:
            raise RuntimeError("This session has no data source; call train_batch(batch) instead.")
        self._check_open()
        self._stop_requested = False
        with contextlib.ExitStack() as stack:
            if isinstance(self.model, nn.parallel.DistributedDataParallel):
                stack.enter_context(self.model.join())
            iterator = iter(self.data)
            # Closing the iterator shuts down DataLoader workers and pending fetches.
            stack.callback(lambda: getattr(iterator, "close", lambda: None)())
            for batch in iterator:
                metrics = self.train_batch(batch)
                yield batch, metrics
                if self._stop_requested:
                    break

    def train_batch(self, batch: Mapping[str, torch.Tensor]) -> Mapping[str, Any] | None:
        """Train on one batch.

        Parameters
        ----------
        batch : Mapping[str, torch.Tensor]
            A collated batch.

        Returns
        -------
        Mapping or None
            The step's metrics, or None if the batch was skipped (empty, or smaller than
            ``min_batch_size``).

        Raises
        ------
        RuntimeError
            If the session is closed.
        """
        self._check_open()
        rows = batch_rows(batch)
        if rows == 0:
            logger.debug("Skipping an empty batch.")
            return None
        if self.min_batch_size is not None and rows < self.min_batch_size:
            logger.debug("Skipping a batch of %d rows (min_batch_size=%d).", rows, self.min_batch_size)
            return None

        self.model.train()
        metrics = self.step(self.model, move_batch(batch, self.device))
        self.batch_count += 1
        self.rows_seen += rows
        for callback in self.callbacks:
            callback(self, batch, metrics)
        if self.checkpoint_every is not None and self.batch_count % self.checkpoint_every == 0:
            self._save_periodic()
        return metrics

    def stop(self) -> None:
        """End the current iteration after the batch being processed."""
        self._stop_requested = True

    def save_checkpoint(self, name: str) -> Path | None:
        """Write a checkpoint named ``name`` into ``checkpoint_dir``.

        Returns the path, or None when checkpointing is disabled or this is not rank 0.
        """
        if self.checkpoint_dir is None or not _is_rank_zero():
            return None
        return save_checkpoint(
            self.checkpoint_dir / name,
            self.model,
            self.step,
            batch_count=self.batch_count,
            rows_seen=self.rows_seen,
        )

    def _save_periodic(self) -> None:
        path = self.save_checkpoint(f"checkpoint-{self.batch_count:08d}.pt")
        if path is None:
            return
        self._periodic_checkpoints.append(path)
        if self.keep_last is not None:
            while len(self._periodic_checkpoints) > self.keep_last:
                self._periodic_checkpoints.pop(0).unlink(missing_ok=True)

    def close(self) -> nn.Module:
        """Stop, write the final checkpoint (``checkpoint-final.pt``) and return the model.

        Safe to call more than once.

        Returns
        -------
        nn.Module
            The trained model, unwrapped from any DDP wrapper.
        """
        if not self._closed:
            self.stop()
            self.save_checkpoint("checkpoint-final.pt")
            self._closed = True
        return unwrap_model(self.model)

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("This TrainSession is closed.")

    def __enter__(self) -> TrainSession:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def _is_rank_zero() -> bool:
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
