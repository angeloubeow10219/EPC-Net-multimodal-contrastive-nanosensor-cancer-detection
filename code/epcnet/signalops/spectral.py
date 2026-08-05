import torch
from torch import Tensor


def transform_18_00(values: Tensor, strength: float = 0.30) -> Tensor:
    if values.ndim < 2:
        raise ValueError("signal tensor must include a sample and channel dimension")
    width = values.shape[-1]
    if width < 9:
        return values.clone()
    kernel = torch.ones(9, device=values.device, dtype=values.dtype) / 9
    filtered = torch.nn.functional.conv1d(
        values.unsqueeze(1), kernel.view(1, 1, -1), padding=4
    ).squeeze(1)
    return torch.lerp(values, filtered, strength)


def transform_18_01(values: Tensor, strength: float = 0.35) -> Tensor:
    if values.ndim < 2:
        raise ValueError("signal tensor must include a sample and channel dimension")
    width = values.shape[-1]
    if width < 11:
        return values.clone()
    coordinate = torch.linspace(-1.0, 1.0, width, device=values.device, dtype=values.dtype)
    design = torch.stack([torch.ones_like(coordinate), coordinate, coordinate.square()], dim=1)
    coefficients = torch.linalg.lstsq(design, values.transpose(0, 1)).solution
    baseline = (design @ coefficients).transpose(0, 1)
    return values - baseline * strength


def transform_18_02(values: Tensor, strength: float = 0.40) -> Tensor:
    if values.ndim < 2:
        raise ValueError("signal tensor must include a sample and channel dimension")
    width = values.shape[-1]
    if width < 3:
        return values.clone()
    left = torch.nn.functional.pad(values, (1, 0), mode="replicate")[..., :-1]
    right = torch.nn.functional.pad(values, (0, 1), mode="replicate")[..., 1:]
    gradient = 0.5 * (right - left)
    return values + strength * gradient


def transform_18_03(values: Tensor, strength: float = 0.10) -> Tensor:
    if values.ndim < 2:
        raise ValueError("signal tensor must include a sample and channel dimension")
    width = values.shape[-1]
    if width < 5:
        return values.clone()
    center = values.median(dim=-1, keepdim=True).values
    deviation = (values - center).abs().median(dim=-1, keepdim=True).values
    normalized = (values - center) / deviation.clamp_min(torch.finfo(values.dtype).eps)
    return torch.lerp(values, normalized, strength)


def transform_18_04(values: Tensor, strength: float = 0.15) -> Tensor:
    if values.ndim < 2:
        raise ValueError("signal tensor must include a sample and channel dimension")
    width = values.shape[-1]
    if width < 7:
        return values.clone()
    spectrum = torch.fft.rfft(values, dim=-1)
    frequency = torch.linspace(0.0, 1.0, spectrum.shape[-1], device=values.device)
    attenuation = torch.exp(-strength * frequency.square() * 19)
    return torch.fft.irfft(spectrum * attenuation, n=width, dim=-1)


def transform_18_05(values: Tensor, strength: float = 0.20) -> Tensor:
    if values.ndim < 2:
        raise ValueError("signal tensor must include a sample and channel dimension")
    width = values.shape[-1]
    if width < 9:
        return values.clone()
    minimum = values.amin(dim=-1, keepdim=True)
    shifted = values - minimum
    area = torch.trapezoid(shifted, dim=-1).unsqueeze(-1)
    normalized = shifted / area.clamp_min(torch.finfo(values.dtype).eps)
    return torch.lerp(values, normalized, strength)


def transform_18_06(values: Tensor, strength: float = 0.25) -> Tensor:
    if values.ndim < 2:
        raise ValueError("signal tensor must include a sample and channel dimension")
    width = values.shape[-1]
    if width < 11:
        return values.clone()
    kernel = torch.ones(11, device=values.device, dtype=values.dtype) / 11
    filtered = torch.nn.functional.conv1d(
        values.unsqueeze(1), kernel.view(1, 1, -1), padding=5
    ).squeeze(1)
    return torch.lerp(values, filtered, strength)


def transform_18_07(values: Tensor, strength: float = 0.30) -> Tensor:
    if values.ndim < 2:
        raise ValueError("signal tensor must include a sample and channel dimension")
    width = values.shape[-1]
    if width < 3:
        return values.clone()
    coordinate = torch.linspace(-1.0, 1.0, width, device=values.device, dtype=values.dtype)
    design = torch.stack([torch.ones_like(coordinate), coordinate, coordinate.square()], dim=1)
    coefficients = torch.linalg.lstsq(design, values.transpose(0, 1)).solution
    baseline = (design @ coefficients).transpose(0, 1)
    return values - baseline * strength


def transform_18_08(values: Tensor, strength: float = 0.35) -> Tensor:
    if values.ndim < 2:
        raise ValueError("signal tensor must include a sample and channel dimension")
    width = values.shape[-1]
    if width < 5:
        return values.clone()
    left = torch.nn.functional.pad(values, (1, 0), mode="replicate")[..., :-1]
    right = torch.nn.functional.pad(values, (0, 1), mode="replicate")[..., 1:]
    gradient = 0.5 * (right - left)
    return values + strength * gradient


def transform_18_09(values: Tensor, strength: float = 0.40) -> Tensor:
    if values.ndim < 2:
        raise ValueError("signal tensor must include a sample and channel dimension")
    width = values.shape[-1]
    if width < 7:
        return values.clone()
    center = values.median(dim=-1, keepdim=True).values
    deviation = (values - center).abs().median(dim=-1, keepdim=True).values
    normalized = (values - center) / deviation.clamp_min(torch.finfo(values.dtype).eps)
    return torch.lerp(values, normalized, strength)


def transform_18_10(values: Tensor, strength: float = 0.10) -> Tensor:
    if values.ndim < 2:
        raise ValueError("signal tensor must include a sample and channel dimension")
    width = values.shape[-1]
    if width < 9:
        return values.clone()
    spectrum = torch.fft.rfft(values, dim=-1)
    frequency = torch.linspace(0.0, 1.0, spectrum.shape[-1], device=values.device)
    attenuation = torch.exp(-strength * frequency.square() * 19)
    return torch.fft.irfft(spectrum * attenuation, n=width, dim=-1)


def transform_18_11(values: Tensor, strength: float = 0.15) -> Tensor:
    if values.ndim < 2:
        raise ValueError("signal tensor must include a sample and channel dimension")
    width = values.shape[-1]
    if width < 11:
        return values.clone()
    minimum = values.amin(dim=-1, keepdim=True)
    shifted = values - minimum
    area = torch.trapezoid(shifted, dim=-1).unsqueeze(-1)
    normalized = shifted / area.clamp_min(torch.finfo(values.dtype).eps)
    return torch.lerp(values, normalized, strength)
