"""Training steps: the seam between the session and the model's optimization logic."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn

from lsdb_ml.inputs import InputsLike, model_args


@runtime_checkable
class TrainStep(Protocol):
    """One optimization step on one batch.

    The step owns everything between receiving a batch and returning metrics: zeroing
    gradients, the forward pass, the loss, backward, and the optimizer step. The session
    owns everything around it (device placement, skipping, counting, checkpoints).

    Steps may also define ``state_dict()`` and ``load_state_dict()``; the session then
    includes their state (e.g. the optimizer's) in checkpoints.
    """

    def __call__(self, model: nn.Module, batch: Mapping[str, torch.Tensor]) -> Mapping[str, Any]:
        """Train on ``batch`` and return metrics such as ``{"loss": 0.12}``."""


class SupervisedStep:
    """A standard supervised step: ``loss_fn(model(*inputs), batch[target])``.

    Parameters
    ----------
    loss_fn : callable
        ``loss_fn(prediction, target) -> scalar tensor``, e.g. ``nn.MSELoss()``.
    optimizer : torch.optim.Optimizer
        Optimizer over the model's parameters.
    inputs : str, sequence of str, or callable
        Batch key(s) passed positionally to the model, or ``callable(batch)``.
    target : str
        Batch key of the target.
    scheduler : optional
        Learning-rate scheduler, stepped after every optimizer step.

    Notes
    -----
    Every call steps the optimizer once, whatever the batch size: a small batch weighs as
    much as a full one.
    """

    def __init__(
        self,
        loss_fn: Callable[[Any, torch.Tensor], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        inputs: InputsLike,
        target: str,
        scheduler: Any | None = None,
    ):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.inputs = inputs
        self.target = target
        self.scheduler = scheduler

    def __call__(self, model: nn.Module, batch: Mapping[str, torch.Tensor]) -> dict[str, float]:
        """Run one optimization step and return ``{"loss": float}``."""
        self.optimizer.zero_grad(set_to_none=True)
        prediction = model(*model_args(batch, self.inputs))
        loss = self.loss_fn(prediction, batch[self.target])
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return {"loss": loss.item()}

    def state_dict(self) -> dict[str, Any]:
        """Return optimizer (and scheduler) state for checkpoints."""
        state = {"optimizer": self.optimizer.state_dict()}
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore state saved by :meth:`state_dict`."""
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])
