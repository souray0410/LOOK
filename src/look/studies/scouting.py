from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from look.runtime.artifacts import build_file_manifest, validate_file_manifest
from look.data.cohort import IMAGE_COLUMNS, OBJECTIVE_EVIDENCE, SELF_REPORT_PAIR, SPLITS, _evidence_set, _age_bin, _match_controls, _standardized_differences, _stable_key
from look.runtime.config import ExperimentConfig, ExperimentSelection
from look.runtime.paths import ProjectPaths
from look.studies.pipeline import ExperimentRunner, PipelineOptions
from look.runtime.provenance import sha256, write_json_atomic
from look.runtime.state import stable_hash, utc_now


TASK_SCOUT_SEED = "ukb-record-task-scout-v1"
TASK_PROFILES: dict[str, dict[str, Any]] = {
    "glaucoma_high_confidence": {
        "positive_label": "glaucoma",
        "target": "glaucoma",
        "case_rule": "objective_or_concordant_self_report",
        "final_task_eligible": True,
    },
    "glaucoma_all_evidence": {
        "positive_label": "glaucoma",
        "target": "glaucoma",
        "case_rule": "all_prevalent_evidence",
        "final_task_eligible": True,
    },
    "glaucoma_objective_only": {
        "positive_label": "glaucoma",
        "target": "glaucoma",
        "case_rule": "objective_only",
        "final_task_eligible": False,
    },
    "any_target_eye_disease_high_confidence": {
        "positive_label": "target_eye_disease",
        "target": "any",
        "case_rule": "objective_or_concordant_self_report",
        "final_task_eligible": True,
    },
    "diabetic_eye_disease_all_evidence": {
        "positive_label": "diabetes_related_eye_disease",
        "target": "diabetes_related_eye_disease",
        "case_rule": "all_prevalent_evidence",
        "final_task_eligible": False,
    },
    "macular_degeneration_all_evidence": {
        "positive_label": "macular_degeneration",
        "target": "macular_degeneration",
        "case_rule": "all_prevalent_evidence",
        "final_task_eligible": False,
    },
    "any_target_eye_disease": {
        "positive_label": "target_eye_disease",
        "target": "any",
        "case_rule": "any_prevalent_target_including_target_comorbidity",
        "final_task_eligible": True,
    },
}


def _global_split(participant_id: object, seed: str) -> str:
    value = int(_stable_key(seed, "global-split", participant_id)[:16], 16) / 16**16
    if value < 0.70:
        return "train"
    if value < 0.85:
        return "validation"
    return "test"


def _is_high_confidence(evidence: object) -> bool:
    sources = _evidence_set(evidence)
    return bool(sources & OBJECTIVE_EVIDENCE) or SELF_REPORT_PAIR <= sources


def _is_objective(evidence: object) -> bool:
    return bool(_evidence_set(evidence) & OBJECTIVE_EVIDENCE)


def _select_cases(candidates: pd.DataFrame, profile: dict[str, Any]) -> pd.DataFrame:
    target = str(profile["target"])
    if target == "any":
        prevalent = candidates["prevalent_targets"].fillna("").astype(str).ne("")
        eligible_status = candidates["candidate_status"].isin(
            ["prevalent_case", "excluded_target_comorbidity"]
        )
        cases = candidates.loc[prevalent & eligible_status].copy()
    else:
        cases = candidates.loc[
            candidates["candidate_status"].eq("prevalent_case")
            & candidates["candidate_label"].eq(target)
        ].copy()
    rule = str(profile["case_rule"])
    if rule == "objective_or_concordant_self_report":
        cases = cases.loc[cases["evidence_sources"].map(_is_high_confidence)].copy()
    elif rule == "objective_only":
        cases = cases.loc[cases["evidence_sources"].map(_is_objective)].copy()
    return cases


def _select_incident(candidates: pd.DataFrame, profile: dict[str, Any]) -> pd.DataFrame:
    target = str(profile["target"])
    incident = candidates.loc[candidates["candidate_status"].eq("incident_case")].copy()
    if target != "any":
        incident = incident.loc[incident["candidate_label"].eq(target)].copy()
    return incident


def _label_rows(
    frame: pd.DataFrame,
    profile_id: str,
    positive_label: str,
    label_id: int,
) -> pd.DataFrame:
    result = frame.copy()
    result["label_id"] = int(label_id)
    result["label_name"] = "normal" if label_id == 0 else positive_label
    result["age_5year_bin"] = result["assessment_age"].map(_age_bin)
    result["task_profile"] = profile_id
    result["phenotype_profile"] = f"ukb_record_{profile_id}_binary_bilateral"
    result["reference_standard_type"] = "record_derived_clinical_phenotype"
    result["reference_source"] = (
        "UKB self-report, HES/ICD, diagnosis timing, and procedure records"
    )
    return result


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    frame.to_csv(partial, index=False)
    os.replace(partial, path)


def _write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def build_task_cohort_bank(
    candidate_csv: Path,
    bank_root: Path,
    *,
    seed: str = TASK_SCOUT_SEED,
    profile_ids: list[str] | None = None,
) -> dict[str, Any]:
    candidate_csv = Path(candidate_csv).resolve()
    bank_root = Path(bank_root).resolve()
    selected_profiles = profile_ids or list(TASK_PROFILES)
    unknown = set(selected_profiles) - set(TASK_PROFILES)
    if unknown:
        raise ValueError(f"Unknown task profiles: {sorted(unknown)}")
    source_hash = sha256(candidate_csv)
    manifest_path = bank_root / "task_bank_manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            previous.get("status") == "PASS"
            and previous.get("source_candidate_sha256") == source_hash
            and previous.get("seed") == seed
            and previous.get("profile_ids") == selected_profiles
            and not previous.get("artifact_errors")
            and not validate_file_manifest(previous.get("artifacts", {}))
        ):
            return previous

    candidates = pd.read_csv(candidate_csv, dtype={"participant_id": str, "sex": str})
    required = {
        "participant_id", "instance", "assessment_age", "sex", "assessment_centre",
        "candidate_status", "candidate_label", "prevalent_targets", "evidence_sources",
        *IMAGE_COLUMNS,
    }
    missing = required - set(candidates)
    if missing:
        raise ValueError(f"Task-scout candidates are missing columns: {sorted(missing)}")
    if candidates["participant_id"].duplicated().any():
        raise ValueError("Task-scout source must contain one bilateral visit per participant")
    candidates["split"] = candidates["participant_id"].map(
        lambda value: _global_split(value, seed)
    )
    controls = candidates.loc[
        candidates["candidate_status"].eq("strict_control")
        & candidates["candidate_label"].eq("normal")
    ].copy()
    outputs: list[Path] = []
    summaries: dict[str, Any] = {}

    split_manifest = bank_root / "participant_split_manifest.csv"
    _write_csv(candidates[["participant_id", "split"]], split_manifest)
    outputs.append(split_manifest)
    for profile_id in selected_profiles:
        profile = TASK_PROFILES[profile_id]
        positive_label = str(profile["positive_label"])
        cases = _label_rows(
            _select_cases(candidates, profile), profile_id, positive_label, 1
        )
        normal = _label_rows(controls, profile_id, positive_label, 0)
        incident = _label_rows(
            _select_incident(candidates, profile), profile_id, positive_label, 1
        )
        primary_parts: list[pd.DataFrame] = []
        matching: dict[str, Any] = {}
        for split in SPLITS:
            split_cases = cases.loc[cases["split"].eq(split)].copy()
            split_controls = normal.loc[normal["split"].eq(split)].copy()
            matched, matching[split] = _match_controls(
                split_cases, split_controls, f"{seed}:{profile_id}", split
            )
            primary_parts.extend((split_cases, matched))
        primary = pd.concat(primary_parts, ignore_index=True).sort_values(
            ["split", "label_id", "participant_id"], kind="stable"
        )
        natural = pd.concat((cases, normal), ignore_index=True).sort_values(
            ["split", "label_id", "participant_id"], kind="stable"
        )
        incident = incident.sort_values(["split", "participant_id"], kind="stable")
        profile_root = bank_root / profile_id
        for name, frame in (
            ("primary", primary), ("natural", natural), ("incident", incident)
        ):
            destination = profile_root / name / "reference_labels.csv"
            _write_csv(frame, destination)
            outputs.append(destination)
        split_counts = {
            split: {
                "cases": int(((primary.split == split) & (primary.label_id == 1)).sum()),
                "controls": int(((primary.split == split) & (primary.label_id == 0)).sum()),
            }
            for split in SPLITS
        }
        validation_cases = split_counts["validation"]["cases"]
        test_cases = split_counts["test"]["cases"]
        sample_size_eligible = (
            len(cases) >= 500 and validation_cases >= 75 and test_cases >= 75
        )
        summaries[profile_id] = {
            **profile,
            "prevalent_cases": int(len(cases)),
            "matched_primary_rows": int(len(primary)),
            "incident_cases": int(len(incident)),
            "split_counts": split_counts,
            "matching": matching,
            "standardized_differences": _standardized_differences(primary),
            "sample_size_eligible": sample_size_eligible,
            "final_task_eligible": bool(profile["final_task_eligible"])
            and sample_size_eligible,
            "primary_labels": str(profile_root / "primary" / "reference_labels.csv"),
            "natural_labels": str(profile_root / "natural" / "reference_labels.csv"),
        }
    artifacts = build_file_manifest(outputs)
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "purpose": "validation_only_task_scout_no_test_selection",
        "seed": seed,
        "profile_ids": selected_profiles,
        "source_candidate_csv": str(candidate_csv),
        "source_candidate_sha256": source_hash,
        "profiles": summaries,
        "artifacts": artifacts,
        "artifact_errors": [],
    }
    write_json_atomic(payload, manifest_path)
    return payload


def verify_task_cohort_bank(
    bank_root: Path,
    image_root: Path,
    *,
    check_images: bool = True,
) -> dict[str, Any]:
    bank_root = Path(bank_root).resolve()
    image_root = Path(image_root).resolve()
    manifest = json.loads((bank_root / "task_bank_manifest.json").read_text())
    errors: list[str] = []
    warnings: list[str] = []
    split_map = pd.read_csv(
        bank_root / "participant_split_manifest.csv", dtype={"participant_id": str}
    ).set_index("participant_id")["split"]
    referenced_images: set[str] = set()
    reports: dict[str, Any] = {}
    for profile_id, profile in manifest["profiles"].items():
        primary = pd.read_csv(profile["primary_labels"], dtype={"participant_id": str})
        natural = pd.read_csv(profile["natural_labels"], dtype={"participant_id": str})
        if primary["participant_id"].duplicated().any():
            errors.append(f"{profile_id}: duplicate primary participants")
        if not set(primary.participant_id) <= set(natural.participant_id):
            errors.append(f"{profile_id}: primary is not a natural-cohort subset")
        if not primary["split"].eq(primary["participant_id"].map(split_map)).all():
            errors.append(f"{profile_id}: primary split differs from global split")
        observed = dict(
            primary[["label_id", "label_name"]].drop_duplicates().sort_values("label_id").values
        )
        if observed.get(0) != "normal" or set(observed) != {0, 1}:
            errors.append(f"{profile_id}: invalid binary mapping {observed}")
        for split in SPLITS:
            counts = primary.loc[primary.split.eq(split), "label_id"].value_counts()
            if counts.get(0, 0) != counts.get(1, 0):
                errors.append(f"{profile_id}:{split}: primary cohort is not 1:1")
        smd = profile["standardized_differences"]
        for split, values in smd.items():
            if values["age_smd"] > 0.10 or values["sex_smd"] > 0.10:
                target = errors if profile["final_task_eligible"] else warnings
                target.append(f"{profile_id}:{split}: demographic SMD exceeds 0.10")
            if values["maximum_centre_proportion_difference"] > 0.10:
                target = errors if profile["final_task_eligible"] else warnings
                target.append(f"{profile_id}:{split}: centre imbalance exceeds 0.10")
        for column in IMAGE_COLUMNS:
            referenced_images.update(primary[column].astype(str))
        reports[profile_id] = {
            "rows": int(len(primary)),
            "cases": int((primary.label_id == 1).sum()),
            "final_task_eligible": bool(profile["final_task_eligible"]),
            "split_counts": profile["split_counts"],
        }
    missing_images = []
    if check_images:
        missing_images = [
            relative for relative in sorted(referenced_images)
            if not (image_root / relative).is_file()
        ]
        if missing_images:
            errors.append(f"{len(missing_images)} unique image paths are missing")
    report = {
        "status": "PASS" if not errors else "FAIL",
        "purpose": manifest["purpose"],
        "profiles": reports,
        "unique_referenced_images": len(referenced_images),
        "missing_images": missing_images[:20],
        "warnings": warnings,
        "errors": errors,
    }
    write_json_atomic(report, bank_root / "task_bank_verification.json")
    if errors:
        raise RuntimeError("Task cohort bank verification failed: " + "; ".join(errors[:10]))
    return report


def run_task_scout(
    paths: ProjectPaths,
    *,
    profile_ids: list[str],
    fusion_positions: list[str],
    seed: int,
    epochs: int,
    patience: int,
    gpu_devices: tuple[int, ...],
    execute: bool,
) -> dict[str, Any]:
    bank_root = paths.dataset_root / "cohorts" / "task_scout"
    bank = json.loads((bank_root / "task_bank_manifest.json").read_text())
    unknown = set(profile_ids) - set(bank["profiles"])
    if unknown:
        raise ValueError(f"Unknown task profiles: {sorted(unknown)}")
    identity = {
        "protocol": "ukb_record_task_scout_v1",
        "profiles": profile_ids,
        "fusion_positions": fusion_positions,
        "seed": seed,
        "epochs": epochs,
        "patience": patience,
        "gpu_devices": list(gpu_devices),
        "selection_data": "validation_only",
        "sealed_test_access": False,
    }
    scout_id = stable_hash(identity)[:12]
    scout_root = paths.runs_root / "task_scout" / f"scout__{scout_id}"
    plan = {**identity, "scout_id": scout_id, "configuration_count": len(profile_ids) * len(fusion_positions)}
    write_json_atomic(plan, scout_root / "plan.json")
    if not execute:
        return {**plan, "status": "dry_run"}
    rows: list[dict[str, Any]] = []
    total = plan["configuration_count"]
    current = 0
    for profile_id in profile_ids:
        profile = bank["profiles"][profile_id]
        labels = Path(profile["primary_labels"])
        natural = Path(profile["natural_labels"])
        for fusion_position in fusion_positions:
            current += 1
            config = ExperimentConfig(
                image_root=paths.image_root,
                labels_csv=labels,
                natural_labels_csv=natural,
                preprocess_cache_root=paths.preprocess_cache_root,
                output_root=paths.runs_root,
                cache_root=paths.cache_root,
                label_profile=f"ukb_record_{profile_id}_binary_bilateral",
                class_names=["normal", str(profile["positive_label"])],
                seeds=[seed],
                fusion_positions=[fusion_position],
                filling_strategies=["normalized_mean"],
                epochs=epochs,
                patience=patience,
                warmup_epochs=min(2, max(0, epochs - 1)),
                effective_batch_size=128,
                micro_batch_size=64,
                num_workers=8,
                pretrained_lr=1e-4,
                new_layer_lr=1e-3,
                classifier_dropout=0.2,
                world_size=len(gpu_devices),
            )
            selection = ExperimentSelection(
                fusion_position=fusion_position,
                seed=seed,
                filling_strategy="normalized_mean",
                run_label=f"task-scout-{profile_id}",
            )
            options = PipelineOptions(
                fit_look=False,
                evaluate_random_missing=False,
                evaluate_missing_baselines=False,
                phase="validation",
                resume=True,
                restart=False,
                check_all_image_paths=False,
                bootstrap_iterations=200,
                gpu_devices=gpu_devices,
            )
            write_json_atomic(
                {
                    **plan,
                    "status": "running",
                    "completed": len(rows),
                    "current": current,
                    "total": total,
                    "task_profile": profile_id,
                    "fusion_position": fusion_position,
                    "updated_at_utc": utc_now(),
                },
                scout_root / "progress.json",
            )
            runner = ExperimentRunner(
                config, selection, options, torch.device("cuda:0")
            )
            result = runner.run()
            metrics = result["validation"]["complete"]
            rows.append(
                {
                    "task_profile": profile_id,
                    "positive_label": profile["positive_label"],
                    "fusion_position": fusion_position,
                    "prevalent_cases": profile["prevalent_cases"],
                    "sample_size_eligible": profile["sample_size_eligible"],
                    "final_task_eligible": profile["final_task_eligible"],
                    "experiment_id": runner.experiment_id,
                    "backbone_id": result.get("checkpoint", {}).get("backbone_id"),
                    "best_epoch": result.get("checkpoint", {}).get("epoch"),
                    **metrics,
                }
            )
            rows.sort(
                key=lambda row: (
                    not bool(row["final_task_eligible"]),
                    -float(row["macro_auroc_ovr"]),
                    -float(row["macro_f1"]),
                )
            )
            write_json_atomic(rows, scout_root / "leaderboard.json")
            _write_csv(pd.DataFrame(rows), scout_root / "leaderboard.csv")
            del result, runner
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    final = {
        **plan,
        "status": "complete",
        "completed": len(rows),
        "completed_at_utc": utc_now(),
        "decision": "manual_task_review_required",
        "automatic_task_freeze": False,
        "leaderboard": str(scout_root / "leaderboard.csv"),
    }
    write_json_atomic(final, scout_root / "progress.json")
    return final


def summarize_task_scout(
    scout_root: Path,
    bank_root: Path,
) -> dict[str, Any]:
    """Create a cautious data/task usability report from validation-only scout results."""
    scout_root = Path(scout_root).resolve()
    bank_root = Path(bank_root).resolve()
    plan = json.loads((scout_root / "plan.json").read_text(encoding="utf-8"))
    progress = json.loads((scout_root / "progress.json").read_text(encoding="utf-8"))
    rows = json.loads((scout_root / "leaderboard.json").read_text(encoding="utf-8"))
    bank = json.loads((bank_root / "task_bank_manifest.json").read_text(encoding="utf-8"))

    ranked: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        auroc = float(row["macro_auroc_ovr"])
        if auroc >= 0.75:
            signal_band = "strong_preliminary_signal"
        elif auroc >= 0.65:
            signal_band = "promising_preliminary_signal"
        elif auroc >= 0.55:
            signal_band = "weak_preliminary_signal"
        else:
            signal_band = "inconclusive_near_chance"
        profile = bank["profiles"][row["task_profile"]]
        ranked.append(
            {
                "rank": rank,
                **row,
                "case_rule": profile["case_rule"],
                "split_counts": profile["split_counts"],
                "signal_band": signal_band,
            }
        )

    report_status = "complete" if progress.get("status") == "complete" else "preliminary"
    report = {
        "schema_version": 1,
        "status": report_status,
        "purpose": "internal_ukb_cfp_oct_task_usability_audit",
        "scout_id": plan["scout_id"],
        "completed_configurations": len(ranked),
        "planned_configurations": plan["configuration_count"],
        "selection_data": "validation_only",
        "sealed_test_access": False,
        "automatic_task_selection": False,
        "ranked_profiles": ranked,
        "interpretation_contract": [
            "Signal bands summarize one-seed short-budget validation results only.",
            "A larger cohort is not preferred when phenotype validity is materially weaker.",
            "No profile is promoted without unimodal, missing-modality, fusion-stage, and multi-seed review.",
            "Record-derived phenotypes are not expert image-grading gold standards.",
        ],
        "generated_at_utc": utc_now(),
    }
    write_json_atomic(report, scout_root / "task_usability_summary.json")

    lines = [
        "# UK Biobank CFP/OCT Task Usability Summary",
        "",
        f"Status: **{report_status}** ({len(ranked)}/{plan['configuration_count']} candidates complete).",
        "",
        "This is a validation-only internal audit of record-derived phenotype definitions. "
        "It is not a test result or an expert-label accuracy claim.",
        "",
        "| Rank | Task profile | Cases | Eligible | AUROC | Macro-F1 | Sensitivity | Specificity | Interpretation |",
        "|---:|---|---:|:---:|---:|---:|---:|---:|---|",
    ]
    for row in ranked:
        sensitivity = row.get("sensitivity_per_class", [None, None])[1]
        specificity = row.get("specificity_per_class", [None, None])[1]
        lines.append(
            f"| {row['rank']} | `{row['task_profile']}` | {row['prevalent_cases']} | "
            f"{str(bool(row['final_task_eligible']))} | {float(row['macro_auroc_ovr']):.4f} | "
            f"{float(row['macro_f1']):.4f} | {float(sensitivity):.4f} | "
            f"{float(specificity):.4f} | `{row['signal_band']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- Prefer a defensible quality-volume trade-off, not the largest cohort or highest number alone.",
            "- Require formal unimodal, missing-modality, seven-stage fusion and three-seed checks before LOOK.",
            "- Keep all test splits sealed until the task, baseline and LOOK configuration are frozen.",
            "- Report record-derived phenotype limitations explicitly in any manuscript.",
            "",
        ]
    )
    _write_text("\n".join(lines), scout_root / "task_usability_summary.md")
    return report
