import pytest
import torch

from lsdb_ml import move_batch, resolve_device


def test_resolve_device():
    """Known device requests resolve to torch devices."""
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device(torch.device("cpu")) == torch.device("cpu")
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the error when CUDA is missing")
def test_resolve_missing_cuda():
    """Requesting CUDA without CUDA fails clearly."""
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device("cuda")


def test_move_batch():
    """Tensors move to the device and other values pass through."""
    batch = {"x": torch.zeros(2), "meta": "keep"}
    moved = move_batch(batch, torch.device("cpu"))
    assert moved["x"].device.type == "cpu"
    assert moved["meta"] == "keep"
