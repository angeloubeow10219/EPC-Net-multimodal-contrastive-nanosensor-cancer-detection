import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def primary(self) -> bool:
        return self.rank == 0


def initialize() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    return DistributedContext(rank, local_rank, world_size, device)


def wrap(model: nn.Module, context: DistributedContext) -> nn.Module:
    model = model.to(context.device)
    if context.world_size == 1:
        return model
    return DistributedDataParallel(
        model,
        device_ids=[context.local_rank] if context.device.type == "cuda" else None,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )


def mean(value: Tensor, context: DistributedContext) -> Tensor:
    if context.world_size == 1:
        return value
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced / context.world_size


def barrier(context: DistributedContext) -> None:
    if context.world_size > 1:
        dist.barrier()


def finalize() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
