import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from lsdb_ml import SupervisedStep, TrainSession


def free_port():
    """Find an unused local TCP port for the process group."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def run_rank(rank, world_size, port, out_dir):
    """Train a DDP model on a rank-dependent number of batches."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(0)
        model = nn.parallel.DistributedDataParallel(nn.Linear(1, 1))
        step = SupervisedStep(
            nn.MSELoss(), torch.optim.SGD(model.parameters(), lr=0.1), inputs="x", target="y"
        )
        # Rank 0 gets 6 batches and rank 1 only 2: without join() rank 0 would hang.
        n_batches = 6 if rank == 0 else 2
        data = [{"x": torch.full((4, 1), float(rank + 1)), "y": torch.ones(4, 1)}] * n_batches
        with TrainSession(model, step, data, checkpoint_dir=out_dir) as session:
            for _ in session:
                pass
        trained = session.close()
        torch.save(
            {"batches": session.batch_count, "weight": trained.weight.detach().clone()},
            os.path.join(out_dir, f"rank{rank}.pt"),
        )
    finally:
        dist.destroy_process_group()


def test_ddp_uneven_inputs_do_not_hang(tmp_path):
    """Ranks with different batch counts finish, stay in sync, and only rank 0 checkpoints."""
    ctx = mp.start_processes(
        run_rank, args=(2, free_port(), str(tmp_path)), nprocs=2, join=False, start_method="spawn"
    )
    for process in ctx.processes:
        process.join(timeout=120)
    hung = [p for p in ctx.processes if p.is_alive()]
    for process in hung:  # don't leave stuck ranks behind to block interpreter exit
        process.kill()
        process.join()
    assert not hung, "DDP training hung"
    assert all(p.exitcode == 0 for p in ctx.processes)

    results = [torch.load(tmp_path / f"rank{r}.pt") for r in range(2)]
    assert [r["batches"] for r in results] == [6, 2]
    assert torch.equal(results[0]["weight"], results[1]["weight"])
    assert sorted(p.name for p in tmp_path.glob("checkpoint-*.pt")) == ["checkpoint-final.pt"]
