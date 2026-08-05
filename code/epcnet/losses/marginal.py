import torch
from torch import Tensor, nn


def gaussian_statistics(values: Tensor, epsilon: float) -> tuple[Tensor, Tensor]:
    mean = values.mean(dim=0)
    centered = values - mean
    covariance = centered.transpose(0, 1) @ centered
    covariance = covariance / max(values.shape[0] - 1, 1)
    covariance = (
        covariance
        + torch.eye(
            covariance.shape[0],
            device=covariance.device,
            dtype=covariance.dtype,
        )
        * epsilon
    )
    return mean, covariance


def diagonal_gaussian_kl(left: Tensor, right: Tensor, epsilon: float = 1.0e-6) -> Tensor:
    left_mean = left.mean(dim=0)
    right_mean = right.mean(dim=0)
    left_variance = left.var(dim=0, unbiased=False).clamp_min(epsilon)
    right_variance = right.var(dim=0, unbiased=False).clamp_min(epsilon)
    ratio = left_variance / right_variance
    mean_term = (left_mean - right_mean).square() / right_variance
    return 0.5 * (ratio + mean_term - 1.0 - ratio.log()).mean()


def symmetric_marginal_kl(left: Tensor, right: Tensor, epsilon: float = 1.0e-6) -> Tensor:
    return 0.5 * (
        diagonal_gaussian_kl(left, right, epsilon) + diagonal_gaussian_kl(right, left, epsilon)
    )


def kernel_matrix(values: Tensor, bandwidth: float) -> Tensor:
    squared = torch.cdist(values, values).square()
    return torch.exp(-squared / (2.0 * bandwidth * bandwidth))


def maximum_mean_discrepancy(left: Tensor, right: Tensor, bandwidth: float = 1.0) -> Tensor:
    left_kernel = kernel_matrix(left, bandwidth)
    right_kernel = kernel_matrix(right, bandwidth)
    cross_kernel = torch.exp(-torch.cdist(left, right).square() / (2.0 * bandwidth * bandwidth))
    return left_kernel.mean() + right_kernel.mean() - 2.0 * cross_kernel.mean()


class MarginalBalanceLoss(nn.Module):
    def __init__(self, kl_weight: float = 0.025) -> None:
        super().__init__()
        self.kl_weight = kl_weight

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        return self.kl_weight * symmetric_marginal_kl(left, right)
