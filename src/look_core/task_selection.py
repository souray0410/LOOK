from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import ProjectPaths
from .reproducibility import sha256, write_json_atomic
from .state import utc_now


SELECTED_TASK_MANIFEST = Path("cohorts/task_selection/selected_task_manifest.json")


def selected_task_manifest_path(paths: ProjectPaths) -> Path:
    return paths.dataset_root / SELECTED_TASK_MANIFEST


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_scout_root(paths: ProjectPaths, scout_id: str | None) -> Path:
    task_root = paths.runs_root / "task_scout"
    if scout_id:
        return task_root / f"scout__{scout_id}"
    summaries = sorted(
        task_root.glob("scout__*/task_usability_summary.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not summaries:
        raise FileNotFoundError(f"No completed task-scout summary under {task_root}")
    return summaries[-1].parent


def select_formal_task(
    paths: ProjectPaths,
    *,
    profile_id: str,
    reviewer_note: str,
    scout_id: str | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    """Select a verified validation-only scout profile for formal baseline search."""
    if execute and not reviewer_note.strip():
        raise ValueError("An explicit reviewer note is required to select a formal task")

    bank_root = paths.dataset_root / "cohorts" / "task_scout"
    bank_manifest_path = bank_root / "task_bank_manifest.json"
    verification_path = bank_root / "task_bank_verification.json"
    bank = _read_json(bank_manifest_path)
    verification = _read_json(verification_path)
    if bank.get("status") != "PASS" or verification.get("status") != "PASS":
        raise RuntimeError("The task cohort bank has not passed verification")
    profile = bank.get("profiles", {}).get(profile_id)
    verified_profile = verification.get("profiles", {}).get(profile_id)
    if not profile or not verified_profile:
        raise ValueError(f"Unknown or unverified task profile: {profile_id}")
    if not profile.get("final_task_eligible") or not verified_profile.get(
        "final_task_eligible"
    ):
        raise ValueError(f"Task profile is exploratory-only: {profile_id}")

    scout_root = _resolve_scout_root(paths, scout_id)
    summary_path = scout_root / "task_usability_summary.json"
    summary = _read_json(summary_path)
    if summary.get("status") != "complete":
        raise RuntimeError("Task selection requires a completed scout")
    if summary.get("selection_data") != "validation_only" or summary.get(
        "sealed_test_access"
    ) is not False:
        raise RuntimeError("Task selection evidence must be validation-only with sealed test")
    selected_metrics = next(
        (
            row
            for row in summary.get("ranked_profiles", [])
            if row.get("task_profile") == profile_id
        ),
        None,
    )
    if not selected_metrics:
        raise ValueError(f"Selected profile is absent from scout: {profile_id}")
    if not selected_metrics.get("final_task_eligible"):
        raise ValueError(f"Scout marks profile exploratory-only: {profile_id}")

    primary_labels = Path(profile["primary_labels"]).resolve()
    natural_labels = Path(profile["natural_labels"]).resolve()
    for label_path in (primary_labels, natural_labels):
        if not label_path.is_file():
            raise FileNotFoundError(label_path)

    payload = {
        "schema_version": 1,
        "status": "selected_for_formal_baseline",
        "selected_task_profile": profile_id,
        "phenotype_profile": f"ukb_record_{profile_id}_binary_bilateral",
        "reference_standard_type": "record_derived_clinical_phenotype",
        "selection_data": "validation_only",
        "sealed_test_access": False,
        "scout_id": summary["scout_id"],
        "scout_rank": selected_metrics.get("rank"),
        "prevalent_cases": profile["prevalent_cases"],
        "split_counts": profile["split_counts"],
        "primary_labels": str(primary_labels),
        "primary_labels_sha256": sha256(primary_labels),
        "natural_labels": str(natural_labels),
        "natural_labels_sha256": sha256(natural_labels),
        "scout_validation_metrics": {
            key: selected_metrics.get(key)
            for key in (
                "best_epoch",
                "macro_auroc_ovr",
                "macro_auprc_ovr",
                "macro_f1",
                "balanced_accuracy",
                "sensitivity_per_class",
                "specificity_per_class",
                "ece_15",
            )
        },
        "selection_rationale": (
            "Highest eligible validation AUROC and Macro-F1 in the fixed short-budget "
            "task scout, with adequate disease-specific sample size. Formal baseline "
            "qualification remains required."
        ),
        "reviewer_note": reviewer_note.strip(),
        "baseline_qualification_required": True,
        "approved_for_look": False,
        "source_manifests": {
            "task_bank_manifest": str(bank_manifest_path.resolve()),
            "task_bank_manifest_sha256": sha256(bank_manifest_path),
            "task_bank_verification": str(verification_path.resolve()),
            "task_bank_verification_sha256": sha256(verification_path),
            "task_scout_summary": str(summary_path.resolve()),
            "task_scout_summary_sha256": sha256(summary_path),
        },
        "selected_at_utc": utc_now(),
    }
    if execute:
        write_json_atomic(payload, selected_task_manifest_path(paths))
    else:
        payload["status"] = "dry_run"
    return payload


def validate_selected_task(paths: ProjectPaths) -> dict[str, Any]:
    manifest_path = selected_task_manifest_path(paths)
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "selected_for_formal_baseline":
        raise RuntimeError("No task has been selected for formal baseline execution")
    expected = (
        ("primary_labels", paths.labels_csv),
        ("natural_labels", paths.natural_labels_csv),
    )
    for key, configured_path in expected:
        selected_path = Path(manifest[key]).resolve()
        configured_path = configured_path.resolve()
        if selected_path != configured_path:
            raise RuntimeError(
                f"Configured {key} does not match selected task: "
                f"{configured_path} != {selected_path}"
            )
        if not configured_path.is_file():
            raise FileNotFoundError(configured_path)
        if sha256(configured_path) != manifest[f"{key}_sha256"]:
            raise RuntimeError(f"Selected task artifact hash changed: {configured_path}")
    if manifest.get("selection_data") != "validation_only":
        raise RuntimeError("Formal task was not selected from validation-only evidence")
    if manifest.get("sealed_test_access") is not False:
        raise RuntimeError("Formal task selection accessed sealed test data")
    return manifest
