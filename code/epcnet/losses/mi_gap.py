import torch
from torch import Tensor, nn

from epcnet.losses.contrastive import directional_info_nce
from epcnet.losses.marginal import symmetric_marginal_kl


def info_nce_lower_bound(left: Tensor, right: Tensor, temperature: float) -> Tensor:
    batch_size = left.shape[0]
    loss = directional_info_nce(left, right, temperature)
    return torch.log(torch.tensor(float(batch_size), device=left.device)) - loss


class MutualInformationGap(nn.Module):
    def __init__(self, temperature: float = 0.10, kl_weight: float = 0.025) -> None:
        super().__init__()
        self.temperature = temperature
        self.kl_weight = kl_weight

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        left_to_right = info_nce_lower_bound(left, right, self.temperature)
        right_to_left = info_nce_lower_bound(right, left, self.temperature)
        directional_gap = (left_to_right - right_to_left).abs()
        marginal_gap = symmetric_marginal_kl(left, right)
        return directional_gap + self.kl_weight * marginal_gap
