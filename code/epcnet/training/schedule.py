import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cosine_schedule(
    optimizer: Optimizer,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int,
    minimum_ratio: float = 0.0,
) -> LambdaLR:
    total_steps = max(epochs * steps_per_epoch, 1)
    warmup_steps = max(warmup_epochs * steps_per_epoch, 1)

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return LambdaLR(optimizer, multiplier)
