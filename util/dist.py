import os
from datetime import timedelta

import torch
import torch.distributed as dist

def setup_distributed(backend="nccl", port=None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    has_torchrun_env = all(
        key in os.environ for key in ("LOCAL_RANK", "RANK", "WORLD_SIZE")
    )
    if not has_torchrun_env:
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        torch.cuda.set_device(0)
        return 0, 1

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            world_size=world_size,
            rank=rank,
            timeout=timedelta(minutes=30),
        )
    return rank, world_size
