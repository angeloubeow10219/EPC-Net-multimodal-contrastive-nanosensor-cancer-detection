from dataclasses import dataclass

import torch
from torch import Tensor, nn

from epcnet.losses.contrastive import InfoNCELoss
from epcnet.losses.mi_gap import MutualInformationGap
from epcnet.losses.reconstruction import MaskedReconstructionLoss
from epcnet.models.network import NetworkOutput
from epcnet.settings import LossConfig


@dataclass
class LossBreakdown:
    total: Tensor
    contrastive: Tensor
    reconstruction: Tensor
    supervised: Tensor
    mi_gap: Tensor

    def detached(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach()),
            "contrastive": float(self.contrastive.detach()),
            "reconstruction": float(self.reconstruction.detach()),
            "supervised": float(self.supervised.detach()),
            "mi_gap": float(self.mi_gap.detach()),
        }


class EPCObjective(nn.Module):
    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config
        self.contrastive = InfoNCELoss(config.temperature, config.distance_weight)
        self.mi_gap = MutualInformationGap(config.temperature, config.marginal_kl_weight)
        self.reconstruction = MaskedReconstructionLoss()

    def forward(
        self,
        output: NetworkOutput,
        electrochemical: Tensor,
        plasmonic: Tensor,
        labels: Tensor,
    ) -> LossBreakdown:
        contrastive = self.contrastive(
            output.electrochemical_embedding,
            output.plasmonic_embedding,
        )
        reconstruction = self.reconstruction(
            output.electrochemical_reconstruction,
            output.plasmonic_reconstruction,
            electrochemical,
            plasmonic,
            output.modality_mask,
        )
        supervised = torch.nn.functional.cross_entropy(output.logits, labels)
        mi_gap = self.mi_gap(
            output.electrochemical_embedding,
            output.plasmonic_embedding,
        )
        total = (
            self.config.contrast_weight * contrastive
            + self.config.reconstruction_weight * reconstruction
            + self.config.supervised_weight * supervised
            + self.config.mi_gap_weight * mi_gap
        )
        return LossBreakdown(
            total=total,
            contrastive=contrastive,
            reconstruction=reconstruction,
            supervised=supervised,
            mi_gap=mi_gap,
        )
