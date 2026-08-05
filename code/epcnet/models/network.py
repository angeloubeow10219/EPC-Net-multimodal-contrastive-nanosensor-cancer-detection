from dataclasses import dataclass

import torch
from torch import Tensor, nn

from epcnet.models.encoders import ElectrochemicalEncoder, PlasmonicEncoder
from epcnet.models.heads import ClassificationHead, ProjectionHead, ReconstructionHead
from epcnet.settings import ExperimentConfig


@dataclass
class NetworkOutput:
    electrochemical_embedding: Tensor
    plasmonic_embedding: Tensor
    joint_embedding: Tensor
    logits: Tensor
    electrochemical_reconstruction: Tensor | None
    plasmonic_reconstruction: Tensor | None
    modality_mask: Tensor


class EPCNet(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        signal = config.signal
        model = config.model
        self.modality_dropout = signal.modality_dropout
        self.electrochemical_encoder = ElectrochemicalEncoder(
            width=model.electrochemical_width,
            layers=model.electrochemical_layers,
            state_size=model.state_size,
            expansion=model.expansion,
            dropout=model.dropout,
        )
        self.plasmonic_encoder = PlasmonicEncoder(
            width=model.plasmonic_width,
            layers=model.plasmonic_layers,
            heads=model.attention_heads,
            state_size=model.state_size,
            expansion=model.expansion,
            dropout=model.dropout,
        )
        self.electrochemical_projection = ProjectionHead(
            model.electrochemical_width,
            model.projection_width,
            signal.embedding_dim,
        )
        self.plasmonic_projection = ProjectionHead(
            model.plasmonic_width,
            model.projection_width,
            signal.embedding_dim,
        )
        self.classifier = ClassificationHead(signal.embedding_dim, signal.classes, model.dropout)
        self.reconstructor = ReconstructionHead(
            signal.embedding_dim,
            signal.electrochemical_bins,
            signal.plasmonic_bins,
        )

    def sample_modality_mask(self, batch_size: int, device: torch.device) -> Tensor:
        mask = torch.zeros(batch_size, 2, device=device)
        dropped = torch.rand(batch_size, device=device) < self.modality_dropout
        selected = torch.randint(0, 2, (batch_size,), device=device)
        mask[dropped, selected[dropped]] = 1.0
        return mask

    def fuse(self, electrochemical: Tensor, plasmonic: Tensor, mask: Tensor) -> Tensor:
        electrochemical_available = 1.0 - mask[:, 0:1]
        plasmonic_available = 1.0 - mask[:, 1:2]
        numerator = electrochemical * electrochemical_available + plasmonic * plasmonic_available
        denominator = electrochemical_available + plasmonic_available
        return numerator / denominator.clamp_min(1.0)

    def forward(
        self,
        electrochemical: Tensor,
        plasmonic: Tensor,
        modality_mask: Tensor | None = None,
    ) -> NetworkOutput:
        electrochemical_embedding = self.electrochemical_projection(
            self.electrochemical_encoder(electrochemical)
        )
        plasmonic_embedding = self.plasmonic_projection(self.plasmonic_encoder(plasmonic))
        if modality_mask is None:
            modality_mask = (
                self.sample_modality_mask(electrochemical.shape[0], electrochemical.device)
                if self.training
                else torch.zeros(electrochemical.shape[0], 2, device=electrochemical.device)
            )
        joint = self.fuse(electrochemical_embedding, plasmonic_embedding, modality_mask)
        logits = self.classifier(joint)
        electrochemical_reconstruction = None
        plasmonic_reconstruction = None
        if modality_mask[:, 0].any():
            electrochemical_reconstruction = self.reconstructor(joint, "electrochemical")
        if modality_mask[:, 1].any():
            plasmonic_reconstruction = self.reconstructor(joint, "plasmonic")
        return NetworkOutput(
            electrochemical_embedding=electrochemical_embedding,
            plasmonic_embedding=plasmonic_embedding,
            joint_embedding=joint,
            logits=logits,
            electrochemical_reconstruction=electrochemical_reconstruction,
            plasmonic_reconstruction=plasmonic_reconstruction,
            modality_mask=modality_mask,
        )
