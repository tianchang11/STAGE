"""
Distributed helpers for multi-GPU training.

Key goal: make multi-GPU contrastive / metric losses see the *global* batch
(same effective negatives / hard samples as single-GPU with IMS_PER_BATCH),
while keeping correct local gradients under DDP.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


@torch.no_grad()
def all_gather_tensor(tensor: torch.Tensor) -> list[torch.Tensor]:
    """All-gather a tensor from every rank (no gradient)."""
    world_size = get_world_size()
    if world_size == 1:
        return [tensor]
    tensor = tensor.contiguous()
    tensors = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors, tensor)
    return tensors


def gather_tensor(tensor: torch.Tensor, keep_local_grad: bool = True) -> torch.Tensor:
    """
    Concatenate ``tensor`` from all ranks along dim 0.

    If ``keep_local_grad`` is True (default), the local rank's slice is the
    original tensor (with autograd history). Remote slices are detached
    constants — standard local-loss DDP pattern for contrastive learning.
    """
    world_size = get_world_size()
    if world_size == 1:
        return tensor

    rank = get_rank()
    tensor = tensor.contiguous()
    gathered = all_gather_tensor(tensor.detach())
    if keep_local_grad:
        gathered[rank] = tensor
    return torch.cat(gathered, dim=0)


def gather_labels(labels: torch.Tensor) -> torch.Tensor:
    """All-gather integer labels (no grad)."""
    return gather_tensor(labels.contiguous(), keep_local_grad=False)
