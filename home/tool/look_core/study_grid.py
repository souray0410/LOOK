from __future__ import annotations

import csv
import gc
import io
import itertools
import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
import pandas as pd

from .artifacts import build_file_manifest, validate_file_manifest
from .config import ExperimentConfig, ExperimentSelection
from .paths import ProjectPaths
from .pipeline import ExperimentRunner, PipelineOptions
from .quality_audit import generate_baseline_quality_audit
from .state import atomic_write_json, atomic_write_text, stable_hash, utc_now


FUSION_POSITIONS = ["input", "stem", "layer1", "layer2", "layer3", "layer4", "feature"]
CONFIRMATION_SEEDS = [3407, 3408, 3409]
BASELINE_AUROC_GATE = 0.80
BASELINE_MACRO_F1_GATE = 0.70
MIN_SENSITIVITY_GATE = 0.65
MIN_SPECIFICITY_GATE = 0.65


def classifier_profile(
    name: str,
    pretrained_lr: float,
    new_layer_lr: float,
    classifier_dropout: float,
    label_smoothing: float,
) -> dict[str, Any]:
    return {
        "name": name,
        "epochs": 100,
        "patience": 15,
        "effective_batch_size": 128,
        "micro_batch_size": 64,
        "num_workers": 8,
        "pretrained_lr": pretrained_lr,
        "new_layer_lr": new_layer_lr,
        "weight_decay": 1e-4,
        "warmup_epochs": 5,
        "sampling_strategy": "natural_without_replacement",
        "loss_name": "cross_entropy",
        "label_smoothing": label_smoothing,
        "classifier_dropout": classifier_dropout,
        "training_strategy": "end_to_end_finetuning",
        "amp": True,
        "baseline_auroc_target": BASELINE_AUROC_GATE,
        "baseline_macro_f1_target": BASELINE_MACRO_F1_GATE,
        "baseline_min_sensitivity": MIN_SENSITIVITY_GATE,
        "baseline_min_specificity": MIN_SPECIFICITY_GATE,
    }


def calibration_classifier_profiles() -> list[dict[str, Any]]:
    profiles = []
    for pretrained_lr, new_layer_lr in (
        (3e-5, 3e-4),
        (1e-4, 1e-3),
        (3e-4, 3e-3),
    ):
        for dropout in (0.0, 0.2):
            name = (
                f"lr-{pretrained_lr:.0e}-{new_layer_lr:.0e}"
                f"__dropout-{dropout:.1f}"
            )
            profiles.append(
                classifier_profile(
                    name,
                    pretrained_lr,
                    new_layer_lr,
                    dropout,
                    0.0,
                )
            )
    return profiles


def _primary_classifier_profile() -> dict[str, Any]:
    return classifier_profile("primary", 1e-4, 1e-3, 0.2, 0.0)


def _primary_gan_profile() -> dict[str, Any]:
    return {
        "name": "primary",
        "gan_validation_fraction": 0.1,
        "gan_epochs": 100,
        "gan_patience": 10,
        "gan_effective_batch_size": 448,
        "gan_batch_size": 112,
        "gan_num_workers": 8,
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
        "evaluate_missing_baselines": True,
        "missing_patterns": ["oct_missing", "cfp_missing"],
        "missing_ratios": [0.2, 0.4, 0.6, 0.8, 1.0],
        "correction_nodes": ["all_available"],
        "downsample_factors": [4, 8, 16],
        "latent_dims": [16, 32, 64, 128, 256],
        "max_pca_rank": 256,
        "primary_metric": "macro_auroc_ovr",
    }


def _disabled_look_profile() -> dict[str, Any]:
    return {
        "name": "disabled",
        "enabled": False,
        "evaluate_random_missing": False,
        "evaluate_missing_baselines": False,
    }


@dataclass
class StudyGrid:
    label_profile: str | None = None
    class_names: list[str] | None = None
    backbones: list[str] = field(default_factory=lambda: ["resnet50"])
    fusion_positions: list[str] = field(default_factory=lambda: list(FUSION_POSITIONS))
    seeds: list[int] = field(default_factory=lambda: list(CONFIRMATION_SEEDS))
    filling_strategies: list[str] = field(
        default_factory=lambda: ["raw_zero", "normalized_mean", "paired_cgan"]
    )
    classifier_profiles: list[dict[str, Any]] = field(
        default_factory=lambda: [_primary_classifier_profile()]
    )
    gan_profiles: list[dict[str, Any]] = field(
        default_factory=lambda: [_primary_gan_profile()]
    )
    look_profiles: list[dict[str, Any]] = field(
        default_factory=lambda: [_primary_look_profile()]
    )

    def validate(self) -> None:
        if self.class_names is not None and (
            len(self.class_names) != 2 or self.class_names[0] != "normal"
        ):
            raise ValueError("StudyGrid class_names must be [normal, positive phenotype]")
        if self.backbones != ["resnet50"]:
            raise ValueError("LOOK currently supports only the resnet50 backbone")
        allowed = set(FUSION_POSITIONS) | {"oct_only", "cfp_only"}
        if not self.fusion_positions or set(self.fusion_positions) - allowed:
            raise ValueError("Unknown fusion or unimodal architecture position")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("Seeds must be non-empty and unique")
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


def baseline_calibration_grid() -> StudyGrid:
    return StudyGrid(
        fusion_positions=["feature"],
        seeds=[3407],
        filling_strategies=["normalized_mean"],
        classifier_profiles=calibration_classifier_profiles(),
        gan_profiles=[{"name": "not_applicable"}],
        look_profiles=[_disabled_look_profile()],
    )


def baseline_search_grid(profile: dict[str, Any] | None = None) -> StudyGrid:
    return StudyGrid(
        fusion_positions=list(FUSION_POSITIONS),
        seeds=[3407],
        filling_strategies=["normalized_mean"],
        classifier_profiles=[profile or _primary_classifier_profile()],
        gan_profiles=[{"name": "not_applicable"}],
        look_profiles=[_disabled_look_profile()],
    )


def baseline_confirmation_grid(
    fusion_positions: list[str], profile: dict[str, Any] | None = None
) -> StudyGrid:
    if len(fusion_positions) != 3 or len(set(fusion_positions)) != 3:
        raise ValueError("Baseline confirmation requires three unique fusion positions")
    return StudyGrid(
        fusion_positions=fusion_positions,
        seeds=list(CONFIRMATION_SEEDS),
        filling_strategies=["normalized_mean"],
        classifier_profiles=[profile or _primary_classifier_profile()],
        gan_profiles=[{"name": "not_applicable"}],
        look_profiles=[_disabled_look_profile()],
    )


def unimodal_reference_grid(profile: dict[str, Any]) -> StudyGrid:
    return StudyGrid(
        fusion_positions=["oct_only", "cfp_only"],
        seeds=[3407],
        filling_strategies=["normalized_mean"],
        classifier_profiles=[profile],
        gan_profiles=[{"name": "not_applicable"}],
        look_profiles=[_disabled_look_profile()],
    )


def _without_control_keys(profile: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {
        key: value
        for key, value in profile.items()
        if key not in {"name", *keys}
    }


def _classifier_profile_name(run_label: str) -> str:
    return run_label.split("clf-", 1)[-1].split("__gan-", 1)[0]


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
    if phase == "test" and frozen_manifest is None:
        raise ValueError("Test execution requires a frozen configuration manifest")
    labels = pd.read_csv(
        paths.labels_csv,
        usecols=lambda column: column in {"label_id", "label_name", "phenotype_profile"},
    )
    mapping = dict(
        labels[["label_id", "label_name"]]
        .drop_duplicates()
        .sort_values("label_id")
        .values
    )
    class_names = grid.class_names or [str(mapping[0]), str(mapping[1])]
    if grid.label_profile is not None:
        label_profile = grid.label_profile
    elif "phenotype_profile" in labels and labels["phenotype_profile"].nunique() == 1:
        label_profile = str(labels["phenotype_profile"].iloc[0])
    else:
        raise ValueError("The reference table must define one phenotype_profile")
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
        gan_variants = (
            grid.gan_profiles
            if filling == "paired_cgan"
            else [{"name": "not_applicable"}]
        )
        for gan in gan_variants:
            config_values = {
                **_without_control_keys(classifier),
                **_without_control_keys(gan),
                **_without_control_keys(
                    look_profile,
                    "enabled",
                    "evaluate_random_missing",
                    "evaluate_missing_baselines",
                ),
            }
            config = ExperimentConfig(
                image_root=paths.image_root,
                labels_csv=paths.labels_csv,
                natural_labels_csv=paths.natural_labels_csv,
                preprocess_cache_root=paths.preprocess_cache_root,
                output_root=paths.runs_root,
                cache_root=paths.cache_root,
                label_profile=label_profile,
                class_names=class_names,
                backbone_name=backbone,
                seeds=[seed],
                fusion_positions=[fusion],
                filling_strategies=[filling],
                world_size=len(gpu_devices),
                **config_values,
            )
            label = (
                f"clf-{classifier['name']}__gan-{gan['name']}"
                f"__look-{look_profile['name']}"
            )
            selection = ExperimentSelection(fusion, seed, filling, label)
            options = PipelineOptions(
                train_if_missing=phase == "validation",
                fit_look=bool(look_profile.get("enabled", True)),
                evaluate_random_missing=bool(
                    look_profile.get("evaluate_random_missing", True)
                ),
                evaluate_missing_baselines=bool(
                    look_profile.get("evaluate_missing_baselines", True)
                ),
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
            cases.append(StudyCase(config, selection, options))
    return cases


def _ranking_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        -float(row.get("macro_auroc_ovr", -1.0)),
        -float(row.get("macro_f1", -1.0)),
        -float(row.get("balanced_accuracy", -1.0)),
        float(row.get("ece_15", float("inf"))),
    )


def _release_parent_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _write_leaderboard(rows: list[dict[str, Any]], destination: Path) -> None:
    columns = [
        "rank", "experiment_id", "backbone_id", "fusion_position", "seed",
        "classifier_profile", "training_strategy", "pretrained_lr", "new_layer_lr",
        "classifier_dropout", "label_smoothing", "effective_batch_size",
        "macro_f1", "balanced_accuracy", "macro_auroc_ovr",
        "macro_auprc_ovr", "ece_15", "accuracy", "weighted_f1", "cohen_kappa",
        "best_epoch", "f1_per_class", "sensitivity_per_class",
        "specificity_per_class", "auroc_per_class", "auprc_per_class",
        "confusion_matrix", "started_at_utc", "completed_at_utc",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for rank, row in enumerate(rows, start=1):
        values = {
            key: json.dumps(value) if isinstance(value, (list, dict)) else value
            for key, value in row.items()
            if key in columns
        }
        writer.writerow({"rank": rank, **values})
    atomic_write_text(buffer.getvalue(), destination)


def _search_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for axis in ("fusion_position", "classifier_profile"):
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row[axis]), []).append(row)
        groups[axis] = {
            key: {
                "completed": len(values),
                "mean_macro_f1": statistics.fmean(
                    float(item["macro_f1"]) for item in values
                ),
                "mean_macro_auroc": statistics.fmean(
                    float(item["macro_auroc_ovr"]) for item in values
                ),
                "best": sorted(values, key=_ranking_key)[0],
            }
            for key, values in sorted(grouped.items())
        }
    return {
        "completed_configurations": len(rows),
        "ranking_rule": [
            "macro_auroc_ovr_desc",
            "macro_f1_desc",
            "balanced_accuracy_desc",
            "ece_15_asc",
        ],
        "best_overall": rows[0] if rows else None,
        "groups": groups,
    }


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
        grid, paths, phase=phase, frozen_manifest=frozen_manifest, resume=resume,
        restart=restart, smoke_limit=smoke_limit,
        check_all_image_paths=check_all_image_paths,
        bootstrap_iterations=bootstrap_iterations, gpu_devices=gpu_devices,
    )
    runners = [ExperimentRunner(case.config, case.selection, case.options, device) for case in cases]
    frozen_payload = None
    if phase == "test":
        frozen_payload = json.loads(Path(frozen_manifest).read_text(encoding="utf-8"))
        allowed = set(frozen_payload["experiment_ids"])
        requested = {runner.experiment_id for runner in runners}
        if not requested <= allowed:
            raise ValueError("Test grid contains non-frozen configurations")
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
    plan = {
        **identity, "plan_id": plan_id, "configuration_count": len(runners),
        "created_at_utc": utc_now(),
    }
    atomic_write_json(plan, sweep_dir / "study_plan.json")
    if not execute:
        return {**plan, "status": "dry_run"}
    completed: list[dict[str, str]] = []
    leaderboard: list[dict[str, Any]] = []
    for index, runner in enumerate(runners, start=1):
        _release_parent_cuda_cache()
        atomic_write_json(
            {
                **plan, "status": "running", "completed_count": len(completed),
                "current_index": index, "current_experiment_id": runner.experiment_id,
                "current_backbone_id": runner._backbone_id(),
                "current_configuration": {
                    "fusion_position": runner.selection.fusion_position,
                    "seed": runner.selection.seed,
                    "classifier_profile": _classifier_profile_name(
                        runner.selection.run_label
                    ),
                    "pretrained_lr": runner.config.pretrained_lr,
                    "new_layer_lr": runner.config.new_layer_lr,
                    "classifier_dropout": runner.config.classifier_dropout,
                    "label_smoothing": runner.config.label_smoothing,
                    "effective_batch_size": runner.config.effective_batch_size,
                },
                "completed": completed,
            },
            sweep_dir / "progress.json",
        )
        result = runner.run()
        metrics = result.get("validation", {}).get("complete", {})
        completed.append(
            {
                "experiment_id": runner.experiment_id,
                "result_file": str(runner.experiment_dir / f"{phase}_result.json"),
                "completed_at_utc": str(result["completed_at_utc"]),
            }
        )
        leaderboard.append(
            {
                "experiment_id": runner.experiment_id,
                "backbone_id": result.get("checkpoint", {}).get("backbone_id"),
                "fusion_position": runner.selection.fusion_position,
                "seed": runner.selection.seed,
                "classifier_profile": _classifier_profile_name(
                    runner.selection.run_label
                ),
                "training_strategy": runner.config.training_strategy,
                "pretrained_lr": runner.config.pretrained_lr,
                "new_layer_lr": runner.config.new_layer_lr,
                "classifier_dropout": runner.config.classifier_dropout,
                "label_smoothing": runner.config.label_smoothing,
                "effective_batch_size": runner.config.effective_batch_size,
                "best_epoch": result.get("checkpoint", {}).get("epoch"),
                "started_at_utc": result.get("started_at_utc"),
                "completed_at_utc": result.get("completed_at_utc"),
                **metrics,
            }
        )
        leaderboard.sort(key=_ranking_key)
        atomic_write_json(leaderboard, sweep_dir / "baseline_search_results.json")
        atomic_write_json(_search_diagnostics(leaderboard), sweep_dir / "baseline_search_diagnostics.json")
        _write_leaderboard(leaderboard, sweep_dir / "leaderboard.csv")
        atomic_write_json(
            {**plan, "status": "running", "completed_count": index, "completed": completed},
            sweep_dir / "progress.json",
        )
    final = {**plan, "status": "complete", "completed_count": len(completed), "completed": completed}
    atomic_write_json(final, sweep_dir / "progress.json")
    return final


def _rows(paths: ProjectPaths, run: dict[str, Any]) -> list[dict[str, Any]]:
    path = paths.runs_root / "sweeps" / f"validation__{run['plan_id']}" / "baseline_search_results.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _profile_from_row(row: dict[str, Any]) -> dict[str, Any]:
    profile_name = str(row["classifier_profile"])
    return classifier_profile(
        profile_name, float(row["pretrained_lr"]), float(row["new_layer_lr"]),
        float(row["classifier_dropout"]), float(row["label_smoothing"]),
    )


def _aggregate_confirmation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["fusion_position"]), []).append(row)
    result = []
    for fusion_position, candidates in grouped.items():
        candidates.sort(key=lambda row: int(row["seed"]))
        seeds = [int(row["seed"]) for row in candidates]
        if seeds != CONFIRMATION_SEEDS:
            raise RuntimeError(f"{fusion_position} has incomplete confirmation seeds: {seeds}")
        item: dict[str, Any] = {
            "fusion_position": fusion_position, "seeds": seeds,
            "experiment_ids": [row["experiment_id"] for row in candidates],
            "backbone_ids": [row["backbone_id"] for row in candidates],
        }
        for metric in ("macro_auroc_ovr", "macro_f1", "balanced_accuracy", "ece_15"):
            values = [float(row[metric]) for row in candidates]
            item[f"mean_{metric}"] = statistics.fmean(values)
            item[f"std_{metric}"] = statistics.pstdev(values)
        result.append(item)
    result.sort(key=lambda row: (
        -row["mean_macro_auroc_ovr"], -row["mean_macro_f1"],
        -row["mean_balanced_accuracy"], row["mean_ece_15"], row["std_macro_auroc_ovr"],
    ))
    for rank, row in enumerate(result, start=1):
        row["rank"] = rank
    return result


def _quality_gate_audit(
    paths: ProjectPaths, winner: dict[str, Any], winner_rows: list[dict[str, Any]]
) -> Path:
    output = paths.runs_root / "baseline_selection" / "quality_gate_audit.json"
    return generate_baseline_quality_audit(
        paths.labels_csv,
        paths.image_root,
        winner,
        winner_rows,
        output,
    )


def run_baseline_selection(
    paths: ProjectPaths,
    device: torch.device,
    *,
    execute: bool,
    check_all_image_paths: bool = False,
    gpu_devices: tuple[int, ...] = (0,),
) -> dict[str, Any]:
    calibration = run_study_grid(
        baseline_calibration_grid(), paths, device, execute=execute,
        check_all_image_paths=check_all_image_paths, gpu_devices=gpu_devices,
    )
    if not execute:
        return {
            "status": "dry_run", "protocol": "ukb_glaucoma_binary_baseline_selection",
            "calibration": calibration,
            "unimodal_references": ["oct_only", "cfp_only"],
            "stage_a_fusions": list(FUSION_POSITIONS),
            "stage_b_top_k": 3, "stage_b_seeds": list(CONFIRMATION_SEEDS),
        }
    calibration_rows = _rows(paths, calibration)
    selected_profile = _profile_from_row(calibration_rows[0])
    reference = run_study_grid(
        unimodal_reference_grid(selected_profile), paths, device, execute=True,
        check_all_image_paths=False, gpu_devices=gpu_devices,
    )
    reference_rows = _rows(paths, reference)
    stage_a = run_study_grid(
        baseline_search_grid(selected_profile), paths, device, execute=True,
        check_all_image_paths=False, gpu_devices=gpu_devices,
    )
    stage_a_rows = _rows(paths, stage_a)
    top_positions = [str(row["fusion_position"]) for row in stage_a_rows[:3]]
    stage_b = run_study_grid(
        baseline_confirmation_grid(top_positions, selected_profile), paths, device,
        execute=True, check_all_image_paths=False, gpu_devices=gpu_devices,
    )
    stage_b_rows = _rows(paths, stage_b)
    ranking = _aggregate_confirmation(stage_b_rows)
    winner = ranking[0]
    winner_rows = [row for row in stage_b_rows if row["fusion_position"] == winner["fusion_position"]]
    artifact_paths: list[Path] = []
    for row in winner_rows:
        backbone_root = paths.runs_root / "backbones" / str(row["backbone_id"])
        experiment_root = paths.runs_root / "experiments" / str(row["experiment_id"])
        artifact_paths.extend([
            backbone_root / "training_complete.json", backbone_root / "best.pt",
            experiment_root / "validation_result.json",
        ])
    identity = {
        "protocol": "ukb_glaucoma_binary_baseline_selection",
        "calibration_plan_id": calibration["plan_id"],
        "stage_a_plan_id": stage_a["plan_id"],
        "stage_b_plan_id": stage_b["plan_id"],
        "unimodal_reference_plan_id": reference["plan_id"],
        "classifier_profile": selected_profile,
        "ranking_rule": [
            "mean_macro_auroc_ovr_desc", "mean_macro_f1_desc",
            "mean_balanced_accuracy_desc", "mean_ece_15_asc", "std_macro_auroc_ovr_asc",
        ],
        "quality_gate": {
            "mean_macro_auroc_ovr": BASELINE_AUROC_GATE,
            "mean_macro_f1": BASELINE_MACRO_F1_GATE,
            "positive_class_sensitivity": MIN_SENSITIVITY_GATE,
            "positive_class_specificity": MIN_SPECIFICITY_GATE,
            "multimodal_auroc_not_worse_than_best_unimodal": True,
        },
        "winner": winner, "ranking": ranking,
        "unimodal_references": reference_rows,
        "artifacts": build_file_manifest(artifact_paths),
    }
    selection_id = stable_hash(identity)[:12]
    destination = paths.runs_root / "baseline_selection" / "candidates" / f"baseline_candidate__{selection_id}.json"
    class_f1 = [
        statistics.fmean(float(row["f1_per_class"][index]) for row in winner_rows)
        for index in range(2)
    ]
    positive_class_sensitivity = statistics.fmean(
        float(row["sensitivity_per_class"][1]) for row in winner_rows
    )
    positive_class_specificity = statistics.fmean(
        float(row["specificity_per_class"][1]) for row in winner_rows
    )
    best_unimodal_auroc = max(float(row["macro_auroc_ovr"]) for row in reference_rows)
    gate_checks = {
        "mean_macro_auroc_ovr": float(winner["mean_macro_auroc_ovr"]) >= BASELINE_AUROC_GATE,
        "mean_macro_f1": float(winner["mean_macro_f1"]) >= BASELINE_MACRO_F1_GATE,
        "positive_class_sensitivity": positive_class_sensitivity >= MIN_SENSITIVITY_GATE,
        "positive_class_specificity": positive_class_specificity >= MIN_SPECIFICITY_GATE,
        "multimodal_auroc_not_worse_than_best_unimodal": (
            float(winner["mean_macro_auroc_ovr"]) >= best_unimodal_auroc
        ),
    }
    passed = all(gate_checks.values())
    audit_path = None if passed else _quality_gate_audit(paths, winner, winner_rows)
    payload = {
        **identity, "selection_id": selection_id, "selected_at_utc": utc_now(),
        "status": "candidate_selected" if passed else "quality_gate_failed",
        "decision": "scientific_review_required" if passed else "data_and_model_audit_required",
        "quality_gate_passed": passed,
        "quality_gate_checks": gate_checks,
        "mean_f1_per_class": class_f1,
        "positive_class_sensitivity": positive_class_sensitivity,
        "positive_class_specificity": positive_class_specificity,
        "best_unimodal_auroc": best_unimodal_auroc,
        "quality_audit": str(audit_path) if audit_path else None,
        "manifest_path": str(destination),
    }
    atomic_write_json(payload, destination)
    return payload


def freeze_baseline_candidate(candidate_path: Path, *, reviewer_note: str) -> dict[str, Any]:
    candidate_path = Path(candidate_path).resolve()
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if candidate.get("status") != "candidate_selected" or not candidate.get("quality_gate_passed"):
        raise ValueError("Baseline candidate has not passed the quality gate")
    errors = validate_file_manifest(candidate.get("artifacts", {}))
    if errors:
        raise RuntimeError(f"Baseline candidate artifacts failed verification: {errors[:10]}")
    note = reviewer_note.strip()
    if not note:
        raise ValueError("A reviewer note is required to freeze the baseline")
    destination = candidate_path.parent.parent / f"baseline_selection__{candidate['selection_id']}.json"
    payload = {
        **candidate, "candidate_manifest": str(candidate_path), "reviewer_note": note,
        "approved_at_utc": utc_now(), "status": "frozen",
        "decision": "approved_for_look", "manifest_path": str(destination),
    }
    atomic_write_json(payload, destination)
    return payload


def study_grid_from_baseline_selection(
    manifest_path: Path, *, filling_strategies: list[str] | None = None
) -> StudyGrid:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if payload.get("status") != "frozen":
        raise ValueError("Baseline-selection manifest is not frozen")
    if payload.get("protocol") != "ukb_glaucoma_binary_baseline_selection":
        raise ValueError("Unexpected baseline-selection protocol")
    errors = validate_file_manifest(payload.get("artifacts", {}))
    if errors:
        raise RuntimeError(f"Baseline-selection artifacts failed verification: {errors[:10]}")
    winner = payload["winner"]
    return StudyGrid(
        fusion_positions=[str(winner["fusion_position"])],
        seeds=[int(seed) for seed in winner["seeds"]],
        filling_strategies=list(
            filling_strategies or ["raw_zero", "normalized_mean", "paired_cgan"]
        ),
        classifier_profiles=[payload["classifier_profile"]],
    )


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
        artifact_paths.extend(sorted((runner.experiment_dir / "look").glob("*/factor_selection.json")))
        artifact_paths.extend(sorted((runner.experiment_dir / "look").glob("*/matrix_analysis.json")))
    identity = {
        "grid": asdict(grid),
        "experiment_ids": [runner.experiment_id for runner in runners],
        "artifacts": build_file_manifest(artifact_paths),
    }
    freeze_id = stable_hash(identity)[:12]
    payload = {
        **identity, "freeze_id": freeze_id, "frozen_at_utc": utc_now(),
        "status": "frozen",
    }
    destination = paths.runs_root / "freezes" / f"freeze__{freeze_id}" / "frozen_configuration_manifest.json"
    atomic_write_json(payload, destination)
    return {**payload, "manifest_path": str(destination)}
