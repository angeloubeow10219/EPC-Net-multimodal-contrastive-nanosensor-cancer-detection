from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SignalConfig:
    electrochemical_bins: int = 100
    plasmonic_bins: int = 2000
    embedding_dim: int = 512
    classes: int = 2
    mask_fraction: float = 0.15
    modality_dropout: float = 0.30


@dataclass(frozen=True)
class ModelConfig:
    electrochemical_width: int = 384
    plasmonic_width: int = 768
    projection_width: int = 1024
    electrochemical_layers: int = 4
    plasmonic_layers: int = 12
    attention_heads: int = 12
    state_size: int = 64
    expansion: int = 4
    dropout: float = 0.10


@dataclass(frozen=True)
class LossConfig:
    temperature: float = 0.10
    contrast_weight: float = 1.0
    reconstruction_weight: float = 0.50
    supervised_weight: float = 0.30
    mi_gap_weight: float = 0.20
    marginal_kl_weight: float = 0.025
    distance_weight: float = 0.10


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    gradient_clip: float = 1.0
    workers: int = 16
    precision: str = "bf16"
    world_size: int = 4
    seeds: tuple[int, ...] = (42, 123, 314, 271, 1234)
    output_dir: str = "runs"


@dataclass(frozen=True)
class EvaluationConfig:
    specificity_target: float = 0.99
    bootstrap_resamples: int = 10000
    confidence: float = 0.95
    fdr: float = 0.05
    mcid: float = 0.05
    calibration_bins: int = 15


@dataclass(frozen=True)
class ExperimentConfig:
    signal: SignalConfig = field(default_factory=SignalConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _construct(cls: type[Any], values: dict[str, Any]) -> Any:
    names = cls.__dataclass_fields__.keys()
    return cls(**{name: value for name, value in values.items() if name in names})


def load_config(path: str | Path) -> ExperimentConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")
    return ExperimentConfig(
        signal=_construct(SignalConfig, payload.get("signal", {})),
        model=_construct(ModelConfig, payload.get("model", {})),
        loss=_construct(LossConfig, payload.get("loss", {})),
        train=_construct(TrainConfig, payload.get("train", {})),
        evaluation=_construct(EvaluationConfig, payload.get("evaluation", {})),
    )
