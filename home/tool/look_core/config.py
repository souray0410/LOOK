from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

from .paths import ProjectPaths


def _default_paths() -> ProjectPaths:
    return ProjectPaths.load()


@dataclass
class ExperimentConfig:
    data_root: Path = field(default_factory=lambda: _default_paths().dataset_root)
    output_root: Path = field(default_factory=lambda: _default_paths().runs_root)
    cache_root: Path = field(default_factory=lambda: _default_paths().cache_root)
    num_classes: int = 5
    backbone_name: str = "resnet50"
    image_size: int = 224
    seeds: List[int] = field(default_factory=lambda: [3407, 3408, 3409])
    fusion_positions: List[str] = field(
        default_factory=lambda: [
            "input", "stem", "layer1", "layer2", "layer3", "layer4", "feature"
        ]
    )
    epochs: int = 50
    patience: int = 10
    effective_batch_size: int = 32
    micro_batch_size: int = 32
    num_workers: int = 8
    pretrained_lr: float = 1e-4
    new_layer_lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    sampler_power: float = 0.5
    amp: bool = True
    world_size: int = 1
    monitor_nodes: List[str] = field(
        default_factory=lambda: ["loss", "batch_accuracy", "fusion_logits"]
    )
    monitor_edges: List[str] = field(default_factory=lambda: ["fusion_classifier_edge"])
    monitor_interval_steps: int = 50
    filling_strategies: List[str] = field(
        default_factory=lambda: ["normalized_mean", "paired_cgan"]
    )
    gan_validation_fraction: float = 0.1
    gan_epochs: int = 100
    gan_patience: int = 10
    gan_batch_size: int = 16
    gan_num_workers: int = 8
    gan_learning_rate: float = 2e-4
    gan_beta1: float = 0.5
    gan_lambda_l1: float = 100.0
    gan_base_channels: int = 64
    missing_ratios: List[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8])
    missing_patterns: List[str] = field(default_factory=lambda: ["oct_missing", "cfp_missing"])
    correction_nodes: List[str] = field(default_factory=lambda: ["all_available"])
    downsample_factors: List[int] = field(default_factory=lambda: [4, 8, 16])
    latent_dims: List[int] = field(default_factory=lambda: [16, 32, 64, 128, 256])
    max_pca_rank: int = 256
    alpha_grid: List[float] = field(
        default_factory=lambda: [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    )
    primary_metric: str = "macro_f1"

    @property
    def labels_csv(self) -> Path:
        return self.data_root / "reference_labels.csv"

    def architecture_id(self, fusion_position: str) -> str:
        return f"resnet50_oct_cfp_fusion_{fusion_position}"

    def as_dict(self) -> Dict[str, object]:
        result = asdict(self)
        result["data_root"] = str(self.data_root)
        result["output_root"] = str(self.output_root)
        result["cache_root"] = str(self.cache_root)
        return result

    @classmethod
    def from_dict(cls, values: Dict[str, object]) -> "ExperimentConfig":
        payload = dict(values)
        for key in ("data_root", "output_root", "cache_root"):
            payload[key] = Path(str(payload[key]))
        return cls(**payload)

    @property
    def per_device_micro_batch_size(self) -> int:
        return self.micro_batch_size // self.world_size

    @property
    def per_device_gan_batch_size(self) -> int:
        return self.gan_batch_size // self.world_size

    def validate(self) -> None:
        if self.num_classes != 5:
            raise ValueError("This finalized cohort contract requires exactly five classes")
        if self.backbone_name != "resnet50":
            raise ValueError("LOOK currently supports backbone_name='resnet50'")
        if self.image_size != 224:
            raise ValueError("MHD ResNet50 node shapes currently require image_size=224")
        if min(self.epochs, self.patience, self.effective_batch_size, self.micro_batch_size) < 1:
            raise ValueError("Classifier epochs, patience, and batch size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.effective_batch_size % self.micro_batch_size:
            raise ValueError("effective_batch_size must be divisible by micro_batch_size")
        if self.world_size < 1:
            raise ValueError("world_size must be positive")
        if self.micro_batch_size % self.world_size:
            raise ValueError("global micro_batch_size must be divisible by world_size")
        if not self.monitor_nodes or self.monitor_interval_steps < 1:
            raise ValueError("Monitor nodes and a positive monitor interval are required")
        if not 0.0 < self.gan_validation_fraction < 0.5:
            raise ValueError("gan_validation_fraction must be in (0, 0.5)")
        if self.gan_epochs < 1 or self.gan_batch_size < 1:
            raise ValueError("GAN epochs and batch size must be positive")
        if self.gan_batch_size % self.world_size:
            raise ValueError("global gan_batch_size must be divisible by world_size")
        if self.gan_num_workers < 0 or self.gan_base_channels < 1:
            raise ValueError("GAN workers and base channels are invalid")
        if any(not 0.0 <= ratio <= 1.0 for ratio in self.missing_ratios):
            raise ValueError("Missing ratios must be in [0, 1]")
        allowed_patterns = {"oct_missing", "cfp_missing"}
        if not self.missing_patterns or set(self.missing_patterns) - allowed_patterns:
            raise ValueError(f"Missing patterns must be selected from {sorted(allowed_patterns)}")
        if not self.correction_nodes:
            raise ValueError("At least one correction node or all_available is required")
        if not self.downsample_factors or any(factor < 1 for factor in self.downsample_factors):
            raise ValueError("LOOK downsample factors must be positive")
        if not self.latent_dims or any(dimension < 1 for dimension in self.latent_dims):
            raise ValueError("LOOK latent dimensions must be positive")
        if self.max_pca_rank < 1 or any(alpha < 0 for alpha in self.alpha_grid):
            raise ValueError("LOOK rank must be positive and alpha values non-negative")
        if not self.labels_csv.exists():
            raise FileNotFoundError(self.labels_csv)


@dataclass(frozen=True)
class ExperimentSelection:
    fusion_position: str = "layer2"
    seed: int = 3407
    filling_strategy: str = "normalized_mean"
    run_label: str = "primary"

    def validate(self, config: ExperimentConfig) -> None:
        if self.fusion_position not in config.fusion_positions:
            raise ValueError(f"Unknown fusion position: {self.fusion_position}")
        if self.seed not in config.seeds:
            raise ValueError(f"Seed {self.seed} is not preregistered in {config.seeds}")
        if self.filling_strategy not in config.filling_strategies:
            raise ValueError(f"Unknown filling strategy: {self.filling_strategy}")

    @property
    def architecture_id(self) -> str:
        return f"resnet50_oct_cfp_fusion_{self.fusion_position}"

    @property
    def experiment_id(self) -> str:
        return f"{self.architecture_id}__{self.filling_strategy}__seed{self.seed}__{self.run_label}"

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)
