from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

from .paths import ProjectPaths


def _default_paths() -> ProjectPaths:
    return ProjectPaths.load()


@dataclass
class ExperimentConfig:
    image_root: Path = field(default_factory=lambda: _default_paths().image_root)
    labels_csv: Path = field(default_factory=lambda: _default_paths().labels_csv)
    natural_labels_csv: Path = field(default_factory=lambda: _default_paths().natural_labels_csv)
    preprocess_cache_root: Path = field(default_factory=lambda: _default_paths().preprocess_cache_root)
    output_root: Path = field(default_factory=lambda: _default_paths().runs_root)
    cache_root: Path = field(default_factory=lambda: _default_paths().cache_root)
    label_profile: str = "ukb_record_glaucoma_all_evidence_binary_bilateral"
    class_names: List[str] = field(default_factory=lambda: ["normal", "glaucoma"])
    sample_unit: str = "participant_earliest_complete_bilateral_visit"
    num_classes: int = 2
    backbone_name: str = "resnet50"
    image_size: int = 224
    seeds: List[int] = field(default_factory=lambda: [3407, 3408, 3409])
    fusion_positions: List[str] = field(
        default_factory=lambda: [
            "input", "stem", "layer1", "layer2", "layer3", "layer4", "feature"
        ]
    )
    epochs: int = 100
    patience: int = 15
    effective_batch_size: int = 128
    micro_batch_size: int = 64
    num_workers: int = 8
    pretrained_lr: float = 1e-4
    new_layer_lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    training_strategy: str = "end_to_end_finetuning"
    sampling_strategy: str = "natural_without_replacement"
    loss_name: str = "cross_entropy"
    label_smoothing: float = 0.0
    classifier_dropout: float = 0.2
    amp: bool = True
    world_size: int = 1
    monitor_nodes: List[str] = field(
        default_factory=lambda: ["loss", "batch_accuracy", "fusion_logits"]
    )
    monitor_edges: List[str] = field(default_factory=lambda: ["fusion_classifier_edge"])
    monitor_interval_steps: int = 50
    filling_strategies: List[str] = field(
        default_factory=lambda: ["raw_zero", "normalized_mean", "paired_cgan"]
    )
    gan_validation_fraction: float = 0.1
    gan_epochs: int = 100
    gan_patience: int = 10
    gan_effective_batch_size: int = 448
    gan_batch_size: int = 112
    gan_num_workers: int = 8
    gan_learning_rate: float = 2e-4
    gan_beta1: float = 0.5
    gan_lambda_l1: float = 100.0
    gan_base_channels: int = 64
    missing_ratios: List[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8, 1.0])
    missing_patterns: List[str] = field(default_factory=lambda: ["oct_missing", "cfp_missing"])
    correction_nodes: List[str] = field(default_factory=lambda: ["all_available"])
    downsample_factors: List[int] = field(default_factory=lambda: [4, 8, 16])
    latent_dims: List[int] = field(default_factory=lambda: [8, 16, 32, 64, 96, 128, 192, 256, 384, 512])
    max_pca_rank: int = 512
    primary_metric: str = "macro_f1"
    evaluate_all_factors: bool = False
    baseline_auroc_target: float = 0.80
    baseline_macro_f1_target: float = 0.70
    baseline_min_sensitivity: float = 0.65
    baseline_min_specificity: float = 0.65

    def architecture_id(self, fusion_position: str) -> str:
        if fusion_position in {"oct_only", "cfp_only"}:
            return f"resnet50_{fusion_position}"
        return f"resnet50_oct_cfp_fusion_{fusion_position}"

    def as_dict(self) -> Dict[str, object]:
        result = asdict(self)
        for key in (
            "image_root",
            "labels_csv",
            "natural_labels_csv",
            "preprocess_cache_root",
            "output_root",
            "cache_root",
        ):
            result[key] = str(result[key])
        return result

    @classmethod
    def from_dict(cls, values: Dict[str, object]) -> "ExperimentConfig":
        payload = dict(values)
        for key in (
            "image_root",
            "labels_csv",
            "natural_labels_csv",
            "preprocess_cache_root",
            "output_root",
            "cache_root",
        ):
            payload[key] = Path(str(payload[key]))
        return cls(**payload)

    @property
    def per_device_micro_batch_size(self) -> int:
        return self.micro_batch_size

    @property
    def global_micro_batch_size(self) -> int:
        return self.micro_batch_size * self.world_size

    @property
    def per_device_gan_batch_size(self) -> int:
        return self.gan_batch_size

    @property
    def global_gan_micro_batch_size(self) -> int:
        return self.gan_batch_size * self.world_size

    @property
    def gan_accumulation_steps(self) -> int:
        return self.gan_effective_batch_size // self.global_gan_micro_batch_size

    def validate(self) -> None:
        if self.num_classes != 2:
            raise ValueError("The glaucoma benchmark requires exactly two classes")
        if not self.label_profile.startswith("ukb_record_"):
            raise ValueError("label_profile must identify a versioned UKB record phenotype")
        if len(self.class_names) != 2 or self.class_names[0] != "normal":
            raise ValueError("Binary class_names must be [normal, positive phenotype]")
        if self.sample_unit != "participant_earliest_complete_bilateral_visit":
            raise ValueError("The primary analysis unit must be one bilateral participant visit")
        if self.primary_metric not in {"macro_f1", "macro_auroc_ovr"}:
            raise ValueError("Unsupported primary selection metric")
        if self.backbone_name != "resnet50":
            raise ValueError("LOOK currently supports backbone_name='resnet50'")
        if self.image_size != 224:
            raise ValueError("MHD ResNet50 node shapes currently require image_size=224")
        if min(self.epochs, self.patience, self.effective_batch_size, self.micro_batch_size) < 1:
            raise ValueError("Classifier epochs, patience, and batch size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.world_size < 1:
            raise ValueError("world_size must be positive")
        if self.effective_batch_size % self.global_micro_batch_size:
            raise ValueError("effective_batch_size must be divisible by the global micro batch")
        if self.sampling_strategy != "natural_without_replacement":
            raise ValueError("Classifier search requires natural_without_replacement sampling")
        if self.training_strategy != "end_to_end_finetuning":
            raise ValueError("Complete-modality training requires end_to_end_finetuning")
        if self.loss_name != "cross_entropy":
            raise ValueError("Canonical baseline requires cross_entropy")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if not 0.0 <= self.classifier_dropout < 1.0:
            raise ValueError("classifier_dropout must be in [0, 1)")
        if not self.monitor_nodes or self.monitor_interval_steps < 1:
            raise ValueError("Monitor nodes and a positive monitor interval are required")
        if not 0.0 < self.gan_validation_fraction < 0.5:
            raise ValueError("gan_validation_fraction must be in (0, 0.5)")
        if self.gan_epochs < 1 or min(self.gan_effective_batch_size, self.gan_batch_size) < 1:
            raise ValueError("GAN epochs and batch size must be positive")
        if self.gan_effective_batch_size % self.global_gan_micro_batch_size:
            raise ValueError("gan_effective_batch_size must be divisible by the global GAN micro batch")
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
        if self.max_pca_rank < 1:
            raise ValueError("LOOK rank must be positive")
        for path in (self.labels_csv, self.natural_labels_csv):
            if not path.exists():
                raise FileNotFoundError(path)
        for name, value in (
            ("baseline_auroc_target", self.baseline_auroc_target),
            ("baseline_macro_f1_target", self.baseline_macro_f1_target),
            ("baseline_min_sensitivity", self.baseline_min_sensitivity),
            ("baseline_min_specificity", self.baseline_min_specificity),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")


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
        if self.fusion_position in {"oct_only", "cfp_only"}:
            return f"resnet50_{self.fusion_position}"
        return f"resnet50_oct_cfp_fusion_{self.fusion_position}"

    @property
    def experiment_id(self) -> str:
        return f"{self.architecture_id}__{self.filling_strategy}__seed{self.seed}__{self.run_label}"

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)
