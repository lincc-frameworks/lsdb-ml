import numpy as np
import pytest
import torch
from torch import nn

from lsdb_ml import CatalogIterableDataset, ColumnSpec, Flat, SupervisedStep, TrainSession
from lsdb_ml.training import load_checkpoint


def scaled_dec(df):
    """Feature: declination scaled to [-1, 1], shaped (B, 1)."""
    return (df["dec"].to_numpy(dtype=np.float32) / 60)[:, None]


def linear_target(df):
    """Target: an exact linear function of the feature, shaped (B, 1)."""
    return 3 * scaled_dec(df) + 1


SPEC = ColumnSpec({"x": Flat(["dec"], transform=scaled_dec), "y": Flat(["dec"], transform=linear_target)})


def make_session(catalog, tmp_path=None, epochs=1, **kwargs):
    """A linear model, an SGD step and a single-process data stream."""
    torch.manual_seed(0)
    model = nn.Linear(1, 1)
    step = SupervisedStep(nn.MSELoss(), torch.optim.SGD(model.parameters(), lr=0.1), inputs="x", target="y")
    data = CatalogIterableDataset(catalog, SPEC, batch_size=16, seed=0, epochs=epochs)
    if tmp_path is not None:
        kwargs.setdefault("checkpoint_dir", tmp_path)
    return TrainSession(model, step, data, **kwargs), model, step


def test_session_trains(nested_catalog):
    """Iterating the session trains the model: the loss drops and the fit is close."""
    session, model, _ = make_session(nested_catalog, epochs=3)
    with session:
        losses = [metrics["loss"] for _, metrics in session]
    trained = session.close()
    assert trained is model
    assert np.mean(losses[-5:]) < 0.01 * np.mean(losses[:5])
    assert session.batch_count == len(losses)
    assert session.rows_seen == 1500
    assert model.weight.item() == pytest.approx(3, abs=0.05)
    assert model.bias.item() == pytest.approx(1, abs=0.05)


def test_periodic_and_final_checkpoints(nested_catalog, tmp_path):
    """Checkpoints are written every N batches, pruned to keep_last, and reload exactly."""
    session, model, _ = make_session(nested_catalog, tmp_path, checkpoint_every=5, keep_last=2)
    with session:
        for _ in session:
            pass
    n = session.batch_count
    periodic = sorted(p.name for p in tmp_path.glob("checkpoint-0*.pt"))
    assert periodic == [f"checkpoint-{b:08d}.pt" for b in range(5 * (n // 5) - 5, 5 * (n // 5) + 1, 5)]
    assert (tmp_path / "checkpoint-final.pt").exists()
    assert not list(tmp_path.glob(".*tmp"))

    fresh = nn.Linear(1, 1)
    counters = load_checkpoint(tmp_path / "checkpoint-final.pt", fresh)
    assert counters["batch_count"] == n
    assert torch.equal(fresh.weight, model.weight) and torch.equal(fresh.bias, model.bias)


def test_resume(nested_catalog, tmp_path):
    """resume_from restores weights, optimizer state and counters."""
    session, model, _ = make_session(nested_catalog, tmp_path)
    with session:
        for _ in session:
            pass
    resumed, resumed_model, resumed_step = make_session(
        nested_catalog, resume_from=tmp_path / "checkpoint-final.pt"
    )
    assert resumed.batch_count == session.batch_count
    assert torch.equal(resumed_model.weight, model.weight)
    assert resumed_step.optimizer.state_dict()["param_groups"][0]["lr"] == 0.1


def test_stop(nested_catalog):
    """stop() ends iteration after the current batch."""
    session, _, _ = make_session(nested_catalog)
    with session:
        for _, _ in session:
            if session.batch_count == 3:
                session.stop()
    assert session.batch_count == 3


def test_skips_empty_and_small_batches(nested_catalog):
    """Empty batches and batches under min_batch_size do not step the optimizer."""
    session, model, _ = make_session(nested_catalog, min_batch_size=4)
    weight = model.weight.detach().clone()
    assert session.train_batch({"x": torch.zeros(0, 1), "y": torch.zeros(0, 1)}) is None
    assert session.train_batch({"x": torch.ones(3, 1), "y": torch.ones(3, 1)}) is None
    assert session.batch_count == 0 and torch.equal(model.weight, weight)
    assert session.train_batch({"x": torch.ones(4, 1), "y": torch.ones(4, 1)})["loss"] >= 0
    assert session.batch_count == 1


def test_callbacks(nested_catalog):
    """Callbacks run once per trained batch with the session and metrics."""
    seen = []
    session, _, _ = make_session(nested_catalog, callbacks=[lambda s, b, m: seen.append((s.batch_count, m))])
    with session:
        for _ in session:
            pass
    assert [count for count, _ in seen] == list(range(1, session.batch_count + 1))
    assert all("loss" in metrics for _, metrics in seen)


def test_close_semantics(nested_catalog, tmp_path):
    """close() is idempotent and the session cannot be used afterwards."""
    session, model, _ = make_session(nested_catalog, tmp_path)
    assert session.close() is model
    assert session.close() is model
    with pytest.raises(RuntimeError, match="closed"):
        session.train_batch({"x": torch.ones(2, 1), "y": torch.ones(2, 1)})
    with pytest.raises(RuntimeError, match="closed"):
        next(iter(session))


def test_manual_session_without_data():
    """A session without data is driven by train_batch and cannot be iterated."""
    model = nn.Linear(1, 1)
    step = SupervisedStep(nn.MSELoss(), torch.optim.SGD(model.parameters(), lr=0.1), inputs="x", target="y")
    with TrainSession(model, step) as session:
        with pytest.raises(RuntimeError, match="no data source"):
            next(iter(session))
        assert session.train_batch({"x": torch.ones(2, 1), "y": torch.ones(2, 1)}) is not None


def test_argument_validation():
    """Inconsistent checkpoint and size arguments are rejected."""
    model = nn.Linear(1, 1)
    with pytest.raises(ValueError, match="needs a checkpoint_dir"):
        TrainSession(model, step=None, checkpoint_every=5)
    with pytest.raises(ValueError, match="min_batch_size"):
        TrainSession(model, step=None, min_batch_size=0)
