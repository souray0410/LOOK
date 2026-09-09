from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from look.runtime.state import atomic_write_json, utc_now


def evidence_stratified_validation(
    labels_csv: Path,
    labels: np.ndarray,
    probabilities: np.ndarray,
    participant_ids: np.ndarray,
    *,
    require_complete_split: bool = True,
) -> dict[str, Any]:
    """Report image predictability by record-evidence strength.

    This diagnostic never changes labels, selects a checkpoint, or filters a
    validation/test set.
    """
    frame = pd.read_csv(
        labels_csv,
        dtype={"participant_id": str},
        usecols=["participant_id", "split", "label_id", "label_name", "evidence_sources"],
    )
    frame = frame.loc[frame["split"] == "validation"].set_index("participant_id")
    ids = [str(value) for value in participant_ids]
    valid_coverage = (
        set(ids) == set(frame.index)
        if require_complete_split
        else set(ids).issubset(set(frame.index))
    )
    if len(ids) != len(set(ids)) or not valid_coverage:
        raise ValueError("Validation predictions and phenotype rows are not one-to-one")
    aligned = frame.loc[ids].reset_index()
    expected = aligned["label_id"].astype(int).to_numpy()
    if not np.array_equal(expected, labels.astype(int)):
        raise ValueError("Validation predictions are not aligned to phenotype labels")
    predicted = probabilities.argmax(axis=1)
    true_probability = probabilities[np.arange(len(labels)), labels.astype(int)]
    rows = []
    for index, row in aligned.iterrows():
        raw_sources = "" if pd.isna(row["evidence_sources"]) else str(row["evidence_sources"])
        sources = {item for item in raw_sources.split(";") if item}
        if int(row["label_id"]) == 0:
            evidence_group = "strict_normal_control"
        elif sources == {"6148"}:
            evidence_group = "assessment_self_report_only"
        else:
            evidence_group = "corroborated_or_non_screen_record"
        rows.append({
            "label_id": int(row["label_id"]),
            "label_name": str(row["label_name"]),
            "evidence_group": evidence_group,
            "correct": int(predicted[index] == labels[index]),
            "true_class_probability": float(true_probability[index]),
            "confidence": float(probabilities[index].max()),
        })
    diagnostic = pd.DataFrame(rows)
    groups = []
    for (label_id, label_name, evidence_group), group in diagnostic.groupby(
        ["label_id", "label_name", "evidence_group"], sort=True
    ):
        groups.append({
            "label_id": int(label_id),
            "label_name": str(label_name),
            "evidence_group": str(evidence_group),
            "participants": int(len(group)),
            "class_recall": float(group["correct"].mean()),
            "mean_true_class_probability": float(group["true_class_probability"].mean()),
            "mean_confidence": float(group["confidence"].mean()),
        })
    return {
        "purpose": "diagnostic_only_no_model_or_label_selection",
        "prediction_rows": len(ids),
        "complete_split_required": require_complete_split,
        "complete_split_rows": len(frame),
        "unique_participants": len(set(ids)),
        "label_mismatches": 0,
        "groups": groups,
    }


def _sample_key(row: pd.Series, seed: str) -> str:
    value = f"{seed}:{row['participant_id']}:{row['instance']}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _image_record(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"), dtype=np.float32)
            return {
                "readable": True,
                "width": int(image.width),
                "height": int(image.height),
                "mean": float(array.mean()),
                "std": float(array.std()),
                "near_blank": bool(array.std() < 2.0),
            }
    except (OSError, UnidentifiedImageError, ValueError) as error:
        return {"readable": False, "error": repr(error), "near_blank": True}


def generate_baseline_quality_audit(
    labels_csv: Path,
    image_root: Path,
    winner: dict[str, Any],
    winner_rows: list[dict[str, Any]],
    output_path: Path,
    *,
    samples_per_class: int = 32,
    seed: str = "look-quality-audit-v1",
) -> Path:
    frame = pd.read_csv(labels_csv, dtype={"participant_id": str})
    key_columns = ["participant_id"]
    sampled_frames = []
    for (_, _), group in frame.groupby(["split", "label_name"], sort=True):
        group = group.copy()
        group["_sample_key"] = group.apply(_sample_key, axis=1, seed=seed)
        sampled_frames.append(
            group.sort_values("_sample_key", kind="stable").head(samples_per_class)
        )
    sampled = pd.concat(sampled_frames, ignore_index=True)
    image_records: dict[str, list[dict[str, Any]]] = {
        "left_cfp": [], "left_oct": [], "right_cfp": [], "right_oct": []
    }
    for row in sampled.itertuples(index=False):
        for modality, column in (
            ("left_cfp", "left_fundus_path"),
            ("left_oct", "left_oct_path"),
            ("right_cfp", "right_fundus_path"),
            ("right_oct", "right_oct_path"),
        ):
            relative = str(getattr(row, column))
            record = _image_record(Path(image_root) / relative)
            image_records[modality].append(
                {
                    "participant_id": str(row.participant_id),
                    "split": str(row.split),
                    "label_name": str(row.label_name),
                    "relative_path": relative,
                    **record,
                }
            )
    image_summary = {}
    for modality, records in image_records.items():
        readable = [record for record in records if record["readable"]]
        image_summary[modality] = {
            "sampled": len(records),
            "unreadable": len(records) - len(readable),
            "near_blank": sum(bool(record["near_blank"]) for record in records),
            "mean_intensity": float(np.mean([record["mean"] for record in readable]))
            if readable else None,
            "mean_within_image_std": float(np.mean([record["std"] for record in readable]))
            if readable else None,
        }
    payload = {
        "status": "REVIEW_REQUIRED",
        "created_at_utc": utc_now(),
        "reason": "baseline_compound_quality_gate_failed",
        "winner": winner,
        "per_seed_model_evidence": winner_rows,
        "label_audit": {
            "rows": int(len(frame)),
            "participants": int(frame["participant_id"].nunique()),
            "split_class_counts": {
                f"{split}:{label}": int(count)
                for (split, label), count in frame.groupby(["split", "label_name"]).size().items()
            },
            "duplicate_participant_rows": int(frame.duplicated(key_columns).sum()),
            "participant_split_leakage": int(
                (frame.groupby("participant_id")["split"].nunique() > 1).sum()
            ),
            "participants_with_multiple_labels": int(
                (frame.groupby("participant_id")["label_name"].nunique() > 1).sum()
            ),
            "reference_standard": "record_derived_clinical_phenotype",
        },
        "image_quality_summary": image_summary,
        "sampled_image_records": image_records,
        "review_order": [
            "phenotype_source_timing_and_record_noise",
            "per_class_confusion_and_error_distribution",
            "image_readability_and_modality_visibility",
            "preprocessing_and_ImageNet_weight_mapping",
            "only_then_consider_model_changes",
        ],
    }
    atomic_write_json(payload, output_path)
    return output_path
