import math

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, values: Tensor) -> Tensor:
        scale = values.square().mean(dim=-1, keepdim=True).add(self.epsilon).rsqrt()
        return values * scale * self.weight


class DepthwiseConvGate(nn.Module):
    def __init__(self, width: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.input = nn.Linear(width, width * 2)
        self.depthwise = nn.Conv1d(
            width,
            width,
            kernel_size,
            padding=kernel_size // 2,
            groups=width,
        )
        self.output = nn.Linear(width, width)

    def forward(self, values: Tensor) -> Tensor:
        content, gate = self.input(values).chunk(2, dim=-1)
        content = self.depthwise(content.transpose(1, 2)).transpose(1, 2)
        return self.output(torch.nn.functional.silu(content) * torch.sigmoid(gate))


class SelectiveStateSpace(nn.Module):
    def __init__(self, width: int, state_size: int = 64, expansion: int = 2) -> None:
        super().__init__()
        inner = width * expansion
        self.width = width
        self.state_size = state_size
        self.inner = inner
        self.input = nn.Linear(width, inner * 2)
        self.parameter_projection = nn.Linear(inner, state_size * 2 + inner)
        self.a_log = nn.Parameter(torch.log(torch.arange(1, state_size + 1, dtype=torch.float32)))
        self.skip = nn.Parameter(torch.ones(inner))
        self.output = nn.Linear(inner, width)

    def forward(self, values: Tensor) -> Tensor:
        projected, gate = self.input(values).chunk(2, dim=-1)
        parameters = self.parameter_projection(projected)
        delta, b, c = torch.split(
            parameters, [self.inner, self.state_size, self.state_size], dim=-1
        )
        delta = torch.nn.functional.softplus(delta)
        a = -torch.exp(self.a_log).to(values.dtype)
        state = torch.zeros(
            values.shape[0],
            self.inner,
            self.state_size,
            device=values.device,
            dtype=values.dtype,
        )
        outputs: list[Tensor] = []
        for position in range(values.shape[1]):
            dt = delta[:, position].unsqueeze(-1)
            decay = torch.exp(dt * a)
            drive = dt * b[:, position].unsqueeze(1) * projected[:, position].unsqueeze(-1)
            state = decay * state + drive
            output = (state * c[:, position].unsqueeze(1)).sum(dim=-1)
            outputs.append(output + self.skip * projected[:, position])
        sequence = torch.stack(outputs, dim=1)
        return self.output(sequence * torch.nn.functional.silu(gate))


class FeedForward(nn.Module):
    def __init__(self, width: int, expansion: int, dropout: float) -> None:
        super().__init__()
        hidden = width * expansion
        self.input = nn.Linear(width, hidden * 2)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden, width)

    def forward(self, values: Tensor) -> Tensor:
        left, right = self.input(values).chunk(2, dim=-1)
        return self.output(self.dropout(torch.nn.functional.silu(left) * right))


class Attention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.heads = heads
        self.head_width = width // heads
        self.query_key_value = nn.Linear(width, width * 3)
        self.output = nn.Linear(width, width)
        self.dropout = dropout

    def forward(self, values: Tensor) -> Tensor:
        batch, length, width = values.shape
        packed = self.query_key_value(values)
        query, key, value = packed.chunk(3, dim=-1)
        shape = (batch, length, self.heads, self.head_width)
        query = query.view(shape).transpose(1, 2)
        key = key.view(shape).transpose(1, 2)
        value = value.view(shape).transpose(1, 2)
        attended = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.output(attended.transpose(1, 2).reshape(batch, length, width))


class HybridLayer(nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        state_size: int,
        expansion: int,
        dropout: float,
        attention: bool,
    ) -> None:
        super().__init__()
        self.norm_sequence = RMSNorm(width)
        self.sequence = (
            Attention(width, heads, dropout)
            if attention
            else SelectiveStateSpace(width, state_size, expansion=2)
        )
        self.norm_feedforward = RMSNorm(width)
        self.feedforward = FeedForward(width, expansion, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        values = values + self.drop(self.sequence(self.norm_sequence(values)))
        return values + self.drop(self.feedforward(self.norm_feedforward(values)))


class RotaryEmbedding(nn.Module):
    def __init__(self, width: int, maximum_length: int = 4096) -> None:
        super().__init__()
        frequency = 1.0 / (10000 ** (torch.arange(0, width, 2).float() / width))
        positions = torch.arange(maximum_length).float()
        angles = torch.outer(positions, frequency)
        self.register_buffer("cosine", angles.cos(), persistent=False)
        self.register_buffer("sine", angles.sin(), persistent=False)

    def forward(self, values: Tensor) -> Tensor:
        length = values.shape[1]
        even = values[..., 0::2]
        odd = values[..., 1::2]
        cosine = self.cosine[:length].to(values.dtype)
        sine = self.sine[:length].to(values.dtype)
        rotated_even = even * cosine - odd * sine
        rotated_odd = even * sine + odd * cosine
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


class LearnedPooling(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.query, std=width**-0.5)

    def forward(self, values: Tensor) -> Tensor:
        weights = torch.einsum("bld,d->bl", values, self.query) / math.sqrt(values.shape[-1])
        return torch.einsum("bl,bld->bd", weights.softmax(dim=-1), values)
