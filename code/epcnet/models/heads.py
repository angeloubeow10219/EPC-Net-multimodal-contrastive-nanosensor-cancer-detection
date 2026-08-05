import torch
from torch import Tensor, nn

from epcnet.models.blocks import RMSNorm


class ProjectionHead(nn.Module):
    def __init__(self, input_width: int, hidden_width: int, output_width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            RMSNorm(input_width),
            nn.Linear(input_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, output_width),
        )

    def forward(self, values: Tensor) -> Tensor:
        return torch.nn.functional.normalize(self.layers(values), dim=-1)


class ClassificationHead(nn.Module):
    def __init__(self, width: int, classes: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            RMSNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, classes),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.layers(values)


class ReconstructionHead(nn.Module):
    def __init__(self, width: int, electrochemical_bins: int, plasmonic_bins: int) -> None:
        super().__init__()
        hidden = width * 2
        self.electrochemical = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, electrochemical_bins),
        )
        self.plasmonic = nn.Sequential(
            nn.Linear(width, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, plasmonic_bins),
        )

    def forward(self, embedding: Tensor, target: str) -> Tensor:
        if target == "electrochemical":
            return self.electrochemical(embedding)
        if target == "plasmonic":
            return self.plasmonic(embedding)
        raise ValueError(f"unknown reconstruction target: {target}")
