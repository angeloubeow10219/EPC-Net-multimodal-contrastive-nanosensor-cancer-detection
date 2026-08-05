import torch
from torch import Tensor, nn


def similarity_matrix(left: Tensor, right: Tensor) -> Tensor:
    left = torch.nn.functional.normalize(left, dim=-1)
    right = torch.nn.functional.normalize(right, dim=-1)
    return left @ right.transpose(0, 1)


def directional_info_nce(left: Tensor, right: Tensor, temperature: float) -> Tensor:
    logits = similarity_matrix(left, right) / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    return torch.nn.functional.cross_entropy(logits, targets)


def symmetric_info_nce(left: Tensor, right: Tensor, temperature: float) -> Tensor:
    forward = directional_info_nce(left, right, temperature)
    reverse = directional_info_nce(right, left, temperature)
    return 0.5 * (forward + reverse)


def modality_distance(left: Tensor, right: Tensor) -> Tensor:
    left_norm = left.norm(dim=-1)
    right_norm = right.norm(dim=-1)
    return (left_norm - right_norm).abs().mean()


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.10, distance_weight: float = 0.10) -> None:
        super().__init__()
        self.temperature = temperature
        self.distance_weight = distance_weight

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        contrast = symmetric_info_nce(left, right, self.temperature)
        distance = modality_distance(left, right)
        return contrast + self.distance_weight * distance
