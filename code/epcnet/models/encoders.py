from torch import Tensor, nn

from epcnet.models.blocks import HybridLayer, LearnedPooling, RMSNorm, RotaryEmbedding


class ElectrochemicalEncoder(nn.Module):
    def __init__(
        self,
        width: int = 384,
        layers: int = 4,
        state_size: int = 64,
        expansion: int = 4,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, width // 4, 7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(width // 4, width // 2, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(width // 2, width, 3, stride=1, padding=1),
        )
        self.layers = nn.ModuleList(
            HybridLayer(width, 8, state_size, expansion, dropout, attention=index % 3 == 2)
            for index in range(layers)
        )
        self.norm = RMSNorm(width)
        self.pool = LearnedPooling(width)

    def forward(self, values: Tensor) -> Tensor:
        values = self.stem(values.unsqueeze(1)).transpose(1, 2)
        for layer in self.layers:
            values = layer(values)
        return self.pool(self.norm(values))


class PlasmonicEncoder(nn.Module):
    def __init__(
        self,
        width: int = 768,
        layers: int = 12,
        heads: int = 12,
        state_size: int = 64,
        expansion: int = 4,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.patch = nn.Conv1d(1, width, kernel_size=16, stride=8, padding=4)
        self.position = RotaryEmbedding(width)
        self.layers = nn.ModuleList(
            HybridLayer(
                width,
                heads,
                state_size,
                expansion,
                dropout,
                attention=index % 3 == 1,
            )
            for index in range(layers)
        )
        self.norm = RMSNorm(width)
        self.pool = LearnedPooling(width)

    def forward(self, values: Tensor) -> Tensor:
        values = self.patch(values.unsqueeze(1)).transpose(1, 2)
        values = self.position(values)
        for layer in self.layers:
            values = layer(values)
        return self.pool(self.norm(values))
