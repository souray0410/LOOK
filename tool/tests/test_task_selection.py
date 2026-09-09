from __future__ import annotations

import json
from pathlib import Path

import pytest

from look_core.paths import ProjectPaths
from look_core.task_selection import select_formal_task, validate_selected_task


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(tmp_path: Path) -> ProjectPaths:
    bank = tmp_path / "dataset/cohorts/task_scout"
    profile = bank / "glaucoma_all_evidence"
    primary = profile / "primary/reference_labels.csv"
    natural = profile / "natural/reference_labels.csv"
    for path in (primary, natural):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("participant_id,label_id\n1,1\n2,0\n", encoding="utf-8")
    profile_payload = {
        "prevalent_cases": 925,
        "final_task_eligible": True,
        "split_counts": {
            "train": {"cases": 632, "controls": 632},
            "validation": {"cases": 148, "controls": 148},
            "test": {"cases": 145, "controls": 145},
        },
        "primary_labels": str(primary),
        "natural_labels": str(natural),
    }
    _write_json(
        bank / "task_bank_manifest.json",
        {"status": "PASS", "profiles": {"glaucoma_all_evidence": profile_payload}},
    )
    _write_json(
        bank / "task_bank_verification.json",
        {
            "status": "PASS",
            "profiles": {
                "glaucoma_all_evidence": {"final_task_eligible": True, "cases": 925}
            },
        },
    )
    summary = {
        "status": "complete",
        "selection_data": "validation_only",
        "sealed_test_access": False,
        "scout_id": "example",
        "ranked_profiles": [
            {
                "rank": 1,
                "task_profile": "glaucoma_all_evidence",
                "final_task_eligible": True,
                "best_epoch": 12,
                "macro_auroc_ovr": 0.6948,
                "macro_auprc_ovr": 0.6731,
                "macro_f1": 0.6402,
                "balanced_accuracy": 0.6453,
                "sensitivity_per_class": [0.7635, 0.5270],
                "specificity_per_class": [0.5270, 0.7635],
                "ece_15": 0.0496,
            }
        ],
    }
    _write_json(
        tmp_path / "runs/task_scout/scout__example/task_usability_summary.json",
        summary,
    )
    return ProjectPaths(
        project_root=tmp_path,
        data_root=tmp_path,
        dataset_root=tmp_path / "dataset",
        image_root=tmp_path / "images",
        cohort_root=tmp_path / "dataset/cohorts/standalone",
        labels_csv=primary,
        natural_labels_csv=natural,
        preprocess_cache_root=tmp_path / "cache/preprocessed_pairs",
        cache_root=tmp_path / "cache",
        runs_root=tmp_path / "runs",
        tool_root=tmp_path / "tool",
        pipeline_root=tmp_path / "pipeline",
    )


def test_formal_task_selection_is_explicit_and_hash_checked(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    selected = select_formal_task(
        paths,
        profile_id="glaucoma_all_evidence",
        scout_id="example",
        reviewer_note="Selected after reviewing the validation-only task audit.",
        execute=True,
    )
    assert selected["status"] == "selected_for_formal_baseline"
    assert selected["sealed_test_access"] is False
    assert selected["approved_for_look"] is False
    assert validate_selected_task(paths)["selected_task_profile"] == (
        "glaucoma_all_evidence"
    )
    paths.labels_csv.write_text("participant_id,label_id\n1,0\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact hash changed"):
        validate_selected_task(paths)


def test_formal_task_selection_requires_reviewer_note(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reviewer note"):
        select_formal_task(
            _fixture(tmp_path),
            profile_id="glaucoma_all_evidence",
            scout_id="example",
            reviewer_note="",
            execute=True,
        )
