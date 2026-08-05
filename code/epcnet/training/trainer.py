import logging
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from epcnet.data.batch import SignalBatch
from epcnet.losses.composite import EPCObjective
from epcnet.models.network import EPCNet
from epcnet.settings import ExperimentConfig
from epcnet.training.checkpoint import save_checkpoint
from epcnet.training.distributed import DistributedContext, mean
from epcnet.training.schedule import cosine_schedule


LOGGER = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: EPCNet,
        objective: EPCObjective,
        config: ExperimentConfig,
        context: DistributedContext,
        loader: DataLoader[SignalBatch],
        seed: int,
    ) -> None:
        self.model = model
        self.objective = objective.to(context.device)
        self.config = config
        self.context = context
        self.loader = loader
        self.seed = seed
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
            betas=(0.9, 0.95),
        )
        self.scheduler = cosine_schedule(
            self.optimizer,
            config.train.epochs,
            len(loader),
            config.train.warmup_epochs,
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=config.train.precision == "fp16" and context.device.type == "cuda",
        )
        self.step = 0
        self.best_loss = float("inf")

    def autocast_dtype(self) -> torch.dtype:
        return torch.float16 if self.config.train.precision == "fp16" else torch.bfloat16

    def train_batch(self, batch: SignalBatch) -> dict[str, float]:
        batch = batch.to(self.context.device)
        self.optimizer.zero_grad(set_to_none=True)
        enabled = self.context.device.type == "cuda" and self.config.train.precision in {
            "fp16",
            "bf16",
        }
        with torch.autocast(
            device_type=self.context.device.type,
            dtype=self.autocast_dtype(),
            enabled=enabled,
        ):
            output = self.model(batch.electrochemical, batch.plasmonic)
            losses = self.objective(
                output,
                batch.electrochemical,
                batch.plasmonic,
                batch.labels,
            )
        self.scaler.scale(losses.total).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.gradient_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.step += 1
        values = losses.detached()
        values["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        return values

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = defaultdict(float)
        batches = 0
        sampler = getattr(self.loader, "batch_sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for batch in self.loader:
            values = self.train_batch(batch)
            for key, value in values.items():
                totals[key] += value
            batches += 1
        result = {key: value / max(batches, 1) for key, value in totals.items()}
        reduced = {
            key: float(mean(torch.tensor(value, device=self.context.device), self.context))
            for key, value in result.items()
        }
        return reduced

    def fit(self) -> None:
        output = Path(self.config.train.output_dir) / f"seed-{self.seed}"
        for epoch in range(self.config.train.epochs):
            values = self.train_epoch(epoch)
            if self.context.primary:
                LOGGER.info("epoch=%d values=%s", epoch + 1, values)
                current = values.get("loss", float("inf"))
                save_checkpoint(
                    output / "last.pt",
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    self.step,
                    self.seed,
                    min(self.best_loss, current),
                )
                if current < self.best_loss:
                    self.best_loss = current
                    save_checkpoint(
                        output / "best.pt",
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        epoch,
                        self.step,
                        self.seed,
                        self.best_loss,
                    )
