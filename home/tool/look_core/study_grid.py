from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch

from .artifacts import build_file_manifest, validate_file_manifest
from .config import ExperimentConfig, ExperimentSelection
from .paths import ProjectPaths
from .pipeline import ExperimentRunner, PipelineOptions
from .state import atomic_write_json, stable_hash, utc_now


def _primary_classifier_profile() -> dict[str, Any]:
    return {
        "name": "primary",
        "epochs": 50,
        "patience": 10,
        "effective_batch_size": 256,
        "micro_batch_size": 128,
        "num_workers": 16,
        "pretrained_lr": 3e-4,
        "new_layer_lr": 3e-3,
        "weight_decay": 1e-4,
        "warmup_epochs": 5,
        "sampler_power": 0.5,
        "amp": True,
    }


def _primary_gan_profile() -> dict[str, Any]:
    return {
        "name": "primary",
        "gan_validation_fraction": 0.1,
        "gan_epochs": 100,
        "gan_patience": 10,
        "gan_effective_batch_size": 448,
        "gan_batch_size": 224,
        "gan_num_workers": 16,
        "gan_learning_rate": 2e-4,
        "gan_beta1": 0.5,
        "gan_lambda_l1": 100.0,
        "gan_base_channels": 64,
    }


def _primary_look_profile() -> dict[str, Any]:
    return {
        "name": "primary",
        "enabled": True,
        "evaluate_random_missing": True,
        "missing_patterns": ["oct_missing", "cfp_missing"],
        "missing_ratios": [0.2, 0.4, 0.6, 0.8],
        "correction_nodes": ["all_available"],
        "downsample_factors": [4, 8, 16],
        "latent_dims": [16, 32, 64, 128, 256],
        "max_pca_rank": 256,
        "alpha_grid": [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
        "primary_metric": "macro_f1",
    }


@dataclass
class StudyGrid:
    backbones: list[str] = field(default_factory=lambda: ["resnet50"])
    fusion_positions: list[str] = field(
        default_factory=lambda: ["input", "stem", "layer1", "layer2", "layer3", "layer4", "feature"]
    )
    seeds: list[int] = field(default_factory=lambda: [3407, 3408, 3409])
    filling_strategies: list[str] = field(default_factory=lambda: ["normalized_mean", "paired_cgan"])
    classifier_profiles: list[dict[str, Any]] = field(default_factory=lambda: [_primary_classifier_profile()])
    gan_profiles: list[dict[str, Any]] = field(default_factory=lambda: [_primary_gan_profile()])
    look_profiles: list[dict[str, Any]] = field(default_factory=lambda: [_primary_look_profile()])

    def validate(self) -> None:
        if self.backbones != ["resnet50"]:
            raise ValueError("LOOK currently supports only the resnet50 backbone")
        for label, profiles in (
            ("classifier", self.classifier_profiles),
            ("GAN", self.gan_profiles),
            ("LOOK", self.look_profiles),
        ):
            names = [profile.get("name") for profile in profiles]
            if not profiles or any(not name for name in names) or len(names) != len(set(names)):
                raise ValueError(f"{label} profile names must be present and unique")


@dataclass(frozen=True)
class StudyCase:
    config: ExperimentConfig
    selection: ExperimentSelection
    options: PipelineOptions


def _without_control_keys(profile: dict[str, Any], *keys: str) -> dict[str, Any]:
    excluded = {"name", *keys}
    return {key: value for key, value in profile.items() if key not in excluded}


def expand_study_grid(
    grid: StudyGrid,
    paths: ProjectPaths,
    *,
    phase: str = "validation",
    frozen_manifest: Path | None = None,
    resume: bool = True,
    restart: bool = False,
    smoke_limit: int | None = None,
    check_all_image_paths: bool = False,
    bootstrap_iterations: int = 2000,
    gpu_devices: tuple[int, ...] = (0,),
) -> list[StudyCase]:
    grid.validate()
    if phase not in {"validation", "test"}:
        raise ValueError("phase must be validation or test")
    if phase == "test" and frozen_manifest is None:
        raise ValueError("Test execution requires a frozen configuration manifest")
    cases: list[StudyCase] = []
    axes: Iterable[tuple[Any, ...]] = itertools.product(
        grid.backbones,
        grid.fusion_positions,
        grid.seeds,
        grid.filling_strategies,
        grid.classifier_profiles,
        grid.look_profiles,
    )
    for backbone, fusion, seed, filling, classifier, look_profile in axes:
        gan_variants = grid.gan_profiles if filling == "paired_cgan" else [{"name": "not_applicable"}]
        for gan in gan_variants:
            config_values = {
                **_without_control_keys(classifier),
                **_without_control_keys(gan),
                **_without_control_keys(look_profile, "enabled", "evaluate_random_missing"),
            }
            config = ExperimentConfig(
                data_root=paths.dataset_root,
                output_root=paths.runs_root,
                cache_root=paths.cache_root,
                backbone_name=backbone,
                seeds=[seed],
                fusion_positions=[fusion],
                filling_strategies=[filling],
                world_size=len(gpu_devices),
                **config_values,
            )
            label = f"clf-{classifier['name']}__gan-{gan['name']}__look-{look_profile['name']}"
            selection = ExperimentSelection(fusion, seed, filling, label)
            options = PipelineOptions(
                train_if_missing=phase == "validation",
                fit_look=bool(look_profile.get("enabled", True)),
                evaluate_random_missing=bool(look_profile.get("evaluate_random_missing", True)),
                phase=phase,
                frozen_manifest=str(frozen_manifest) if frozen_manifest else None,
                look_load_only=phase == "test",
                resume=resume,
                restart=restart,
                check_all_image_paths=check_all_image_paths,
                smoke_limit=smoke_limit,
                bootstrap_iterations=bootstrap_iterations,
                gpu_devices=gpu_devices,
            )
            config.validate()
            selection.validate(config)
            options.validate(selection)
            cases.append(StudyCase(config, selection, options))
    return cases


def run_study_grid(
    grid: StudyGrid,
    paths: ProjectPaths,
    device: torch.device,
    *,
    execute: bool,
    phase: str = "validation",
    frozen_manifest: Path | None = None,
    resume: bool = True,
    restart: bool = False,
    smoke_limit: int | None = None,
    check_all_image_paths: bool = False,
    bootstrap_iterations: int = 2000,
    gpu_devices: tuple[int, ...] = (0,),
) -> dict[str, Any]:
    cases = expand_study_grid(
        grid,
        paths,
        phase=phase,
        frozen_manifest=frozen_manifest,
        resume=resume,
        restart=restart,
        smoke_limit=smoke_limit,
        check_all_image_paths=check_all_image_paths,
        bootstrap_iterations=bootstrap_iterations,
        gpu_devices=gpu_devices,
    )
    runners = [ExperimentRunner(case.config, case.selection, case.options, device) for case in cases]
    frozen_payload = None
    if phase == "test":
        frozen_payload = json.loads(Path(frozen_manifest).read_text(encoding="utf-8"))
        allowed = set(frozen_payload["experiment_ids"])
        requested = {runner.experiment_id for runner in runners}
        if not requested <= allowed:
            raise ValueError(f"Test grid contains non-frozen configurations: {sorted(requested - allowed)}")
        errors = validate_file_manifest(frozen_payload["artifacts"])
        if errors:
            raise RuntimeError(f"Frozen artifacts failed verification: {errors[:10]}")
    identity = {
        "phase": phase,
        "grid": asdict(grid),
        "experiment_ids": [runner.experiment_id for runner in runners],
        "freeze_id": frozen_payload.get("freeze_id") if frozen_payload else None,
    }
    plan_id = stable_hash(identity)[:12]
    sweep_dir = paths.runs_root / "sweeps" / f"{phase}__{plan_id}"
    plan = {**identity, "plan_id": plan_id, "configuration_count": len(runners), "created_at_utc": utc_now()}
    atomic_write_json(plan, sweep_dir / "study_plan.json")
    if not execute:
        return {**plan, "status": "dry_run"}
    completed: list[dict[str, str]] = []
    for index, runner in enumerate(runners, start=1):
        result = runner.run()
        completed.append(
            {
                "experiment_id": runner.experiment_id,
                "result_file": str(runner.experiment_dir / f"{phase}_result.json"),
                "completed_at_utc": str(result["completed_at_utc"]),
            }
        )
        atomic_write_json(
            {**plan, "status": "running", "completed_count": index, "completed": completed},
            sweep_dir / "progress.json",
        )
    final = {**plan, "status": "complete", "completed_count": len(completed), "completed": completed}
    atomic_write_json(final, sweep_dir / "progress.json")
    return final


def freeze_study_grid(
    grid: StudyGrid,
    paths: ProjectPaths,
    device: torch.device,
    *,
    gpu_devices: tuple[int, ...],
) -> dict[str, Any]:
    cases = expand_study_grid(grid, paths, phase="validation", gpu_devices=gpu_devices)
    runners = [ExperimentRunner(case.config, case.selection, case.options, device) for case in cases]
    artifact_paths: list[Path] = []
    for runner in runners:
        result_path = runner.experiment_dir / "validation_result.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"Validation is incomplete: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        artifact_paths.extend([result_path, Path(result["checkpoint"]["path"])])
        for direction in result.get("filling", {}).get("directions", {}).values():
            artifact_paths.append(Path(direction["checkpoint"]))
        artifact_paths.extend(sorted((runner.experiment_dir / "look").glob("*/selected/*.pt")))
        artifact_paths.extend(sorted((runner.experiment_dir / "look").glob("*/look_complete.json")))
        artifact_paths.extend(sorted((runner.experiment_dir / "look").glob("*/matrix_analysis.json")))
    identity = {
        "grid": asdict(grid),
        "experiment_ids": [runner.experiment_id for runner in runners],
        "artifacts": build_file_manifest(artifact_paths),
    }
    freeze_id = stable_hash(identity)[:12]
    payload = {**identity, "freeze_id": freeze_id, "frozen_at_utc": utc_now(), "status": "frozen"}
    destination = paths.runs_root / "freezes" / f"freeze__{freeze_id}" / "frozen_configuration_manifest.json"
    atomic_write_json(payload, destination)
    return {**payload, "manifest_path": str(destination)}
