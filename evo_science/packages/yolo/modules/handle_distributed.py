import os
import torch


def handle_distributed():
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    distributed = world_size > 1

    if distributed:
        torch.cuda.set_device(device=local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    if local_rank == 0:
        if not os.path.exists("weights"):
            os.makedirs("weights")
