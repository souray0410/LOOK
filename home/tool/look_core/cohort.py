from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .reproducibility import sha256, write_json_atomic


FOUR_CLASS_MAPPING = {
    "normal": 0,
    "diabetes_related_eye_disease": 1,
    "glaucoma": 2,
    "macular_degeneration": 3,
}
COHORT_NAME = "ukb_retinal_4class_weak"
DEFAULT_SAMPLING_SEED = "ukb-retinal-4class-balanced-v1"
SPLITS = ("train", "validation", "test")


def _counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        split: {
            name: int(
                ((frame["split"] == split) & (frame["label_name"] == name)).sum()
            )
            for name in FOUR_CLASS_MAPPING
        }
        for split in SPLITS
    }


def _stable_sample_key(row: pd.Series, seed: str) -> str:
    identity = ":".join(
        (
            seed,
            str(row["participant_id"]),
            str(row["instance"]),
            str(row.get("array", "0")),
            str(row["eye"]),
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _validate_source(frame: pd.DataFrame) -> None:
    required = {
        "participant_id",
        "instance",
        "eye",
        "fundus_path",
        "oct_path",
        "label_id",
        "label_name",
        "split",
        "reference_source",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Source reference table is missing columns: {sorted(missing)}")
    if set(frame["split"]) != set(SPLITS):
        raise ValueError(f"Expected splits {SPLITS}, got {sorted(frame['split'].unique())}")
    if frame.groupby("participant_id")["split"].nunique().max() != 1:
        raise ValueError("Participant leakage exists in the source table")
    keys = ["participant_id", "instance", "eye"]
    if "array" in frame.columns:
        keys.insert(2, "array")
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate eye-visit rows exist in the source table")


def derive_four_class_cohorts(
    source_labels_csv: Path,
    cohort_root: Path,
    *,
    sampling_seed: str = DEFAULT_SAMPLING_SEED,
) -> dict[str, Any]:
    source_labels_csv = Path(source_labels_csv).resolve()
    cohort_root = Path(cohort_root).resolve()
    source_hash = sha256(source_labels_csv)
    manifest_path = cohort_root / "cohort_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        outputs = [Path(item["path"]) for item in existing.get("outputs", [])]
        if (
            existing.get("status") == "PASS"
            and existing.get("source_labels_sha256") == source_hash
            and existing.get("sampling_seed") == sampling_seed
            and outputs
            and all(path.is_file() and sha256(path) == item["sha256"] for path, item in zip(outputs, existing["outputs"]))
        ):
            return existing

    frame = pd.read_csv(source_labels_csv, dtype={"participant_id": str})
    _validate_source(frame)
    natural = frame.loc[frame["label_name"].isin(FOUR_CLASS_MAPPING)].copy()
    natural["label_id"] = natural["label_name"].map(FOUR_CLASS_MAPPING).astype(int)
    natural["reference_standard_type"] = "laterality_aware_doctor_informed_self_report"

    selected = []
    for split in SPLITS:
        split_frame = natural.loc[natural["split"] == split].copy()
        diseases = split_frame.loc[split_frame["label_name"] != "normal"]
        normal = split_frame.loc[split_frame["label_name"] == "normal"].copy()
        target = int((split_frame["label_name"] == "glaucoma").sum())
        if target <= 0 or len(normal) < target:
            raise ValueError(f"Cannot balance {split}: normal={len(normal)}, target={target}")
        normal["_sample_key"] = normal.apply(_stable_sample_key, axis=1, seed=sampling_seed)
        normal = normal.sort_values("_sample_key", kind="stable").head(target).drop(columns="_sample_key")
        selected.append(pd.concat((diseases, normal), ignore_index=True))
    balanced = pd.concat(selected, ignore_index=True)

    sort_columns = ["split", "label_id", "participant_id", "instance", "eye"]
    natural = natural.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    balanced = balanced.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    natural_counts = _counts(natural)
    balanced_counts = _counts(balanced)
    for split in SPLITS:
        if balanced_counts[split]["normal"] != balanced_counts[split]["glaucoma"]:
            raise AssertionError(f"Normal target mismatch in {split}")

    mapping = {
        "dataset": "UKB paired CFP-central-OCT four-class weak-reference cohort",
        "classes": [
            {"id": class_id, "name": name}
            for name, class_id in FOUR_CLASS_MAPPING.items()
        ],
        "reference_standard_type": (
            "Doctor-informed participant self-report with eye laterality; "
            "not expert image grading."
        ),
        "excluded_primary_class": "cataract",
        "sampling": (
            "All disease rows retained; Normal deterministically capped to the "
            "Glaucoma row count within each pre-existing participant-level split."
        ),
        "sampling_seed": sampling_seed,
    }
    cohort_root.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for name, cohort in (("balanced", balanced), ("natural", natural)):
        directory = cohort_root / name
        directory.mkdir(parents=True, exist_ok=True)
        labels_path = directory / "reference_labels.csv"
        temporary = labels_path.with_suffix(".csv.partial")
        cohort.to_csv(temporary, index=False)
        os.replace(temporary, labels_path)
        write_json_atomic(mapping, directory / "class_mapping.json")
        outputs.extend((labels_path, directory / "class_mapping.json"))

    flow = {
        "source_rows": int(len(frame)),
        "natural_rows": int(len(natural)),
        "balanced_rows": int(len(balanced)),
        "excluded_cataract_rows": int((frame["label_name"] == "cataract").sum()),
        "natural_split_class_counts": natural_counts,
        "balanced_split_class_counts": balanced_counts,
    }
    write_json_atomic(flow, cohort_root / "cohort_flow.json")
    outputs.append(cohort_root / "cohort_flow.json")
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "cohort": COHORT_NAME,
        "source_labels_csv": str(source_labels_csv),
        "source_labels_sha256": source_hash,
        "sampling_seed": sampling_seed,
        "reference_standard": "weak_reference_self_report",
        "flow": flow,
        "outputs": [
            {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in outputs
        ],
    }
    write_json_atomic(manifest, manifest_path)
    return manifest


def verify_four_class_cohorts(
    cohort_root: Path,
    image_root: Path,
    *,
    check_images: bool = True,
) -> dict[str, Any]:
    cohort_root = Path(cohort_root).resolve()
    image_root = Path(image_root).resolve()
    manifest_path = cohort_root / "cohort_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reports: dict[str, Any] = {}
    errors: list[str] = []
    for output in manifest.get("outputs", []):
        path = Path(output["path"])
        if not path.is_file():
            errors.append(f"manifest output missing: {path}")
        elif path.stat().st_size != int(output["bytes"]) or sha256(path) != output["sha256"]:
            errors.append(f"manifest output changed: {path}")
    source = Path(manifest.get("source_labels_csv", ""))
    if not source.is_file() or sha256(source) != manifest.get("source_labels_sha256"):
        errors.append("source reference table is missing or changed")
    frames: dict[str, pd.DataFrame] = {}
    for cohort_name in ("balanced", "natural"):
        labels_path = cohort_root / cohort_name / "reference_labels.csv"
        frame = pd.read_csv(labels_path, dtype={"participant_id": str})
        frames[cohort_name] = frame
        _validate_source(frame)
        observed = dict(
            frame[["label_name", "label_id"]]
            .drop_duplicates()
            .sort_values("label_id")
            .itertuples(index=False, name=None)
        )
        if observed != FOUR_CLASS_MAPPING:
            errors.append(f"{cohort_name}: class mapping differs: {observed}")
        if "cataract" in set(frame["label_name"]):
            errors.append(f"{cohort_name}: cataract was not excluded")
        missing_images = 0
        if check_images:
            for column in ("fundus_path", "oct_path"):
                missing_images += sum(
                    not (image_root / relative).is_file() for relative in frame[column]
                )
        counts = _counts(frame)
        expected_counts = manifest.get("flow", {}).get(
            f"{cohort_name}_split_class_counts"
        )
        if expected_counts is not None and counts != expected_counts:
            errors.append(f"{cohort_name}: split/class counts changed")
        if cohort_name == "balanced":
            for split in SPLITS:
                values = list(counts[split].values())
                if max(values) / min(values) > 1.9:
                    errors.append(f"{split}: balance ratio exceeds 1.9:1")
                if counts[split]["normal"] != counts[split]["glaucoma"]:
                    errors.append(f"{split}: Normal is not capped to Glaucoma")
        if missing_images:
            errors.append(f"{cohort_name}: {missing_images} referenced images are missing")
        reports[cohort_name] = {
            "rows": int(len(frame)),
            "participants": int(frame["participant_id"].nunique()),
            "split_class_counts": counts,
            "participant_split_leakage": int(
                (frame.groupby("participant_id")["split"].nunique() > 1).sum()
            ),
            "duplicate_eye_visit_rows": int(
                frame.duplicated(
                    [
                        "participant_id",
                        "instance",
                        *(["array"] if "array" in frame.columns else []),
                        "eye",
                    ]
                ).sum()
            ),
            "missing_images": missing_images,
            "labels_sha256": sha256(labels_path),
        }
        if reports[cohort_name]["duplicate_eye_visit_rows"]:
            errors.append(f"{cohort_name}: duplicate eye-visit rows exist")
    identity_columns = ["participant_id", "instance", "eye"]
    if "array" in frames["natural"].columns:
        identity_columns.insert(2, "array")
    balanced_keys = set(map(tuple, frames["balanced"][identity_columns].to_numpy()))
    natural_keys = set(map(tuple, frames["natural"][identity_columns].to_numpy()))
    if not balanced_keys <= natural_keys:
        errors.append("balanced cohort is not a strict subset of the natural cohort")
    report = {
        "status": "PASS" if not errors else "FAIL",
        "reference_standard": "weak_reference_self_report",
        "image_root": str(image_root),
        "cohort_root": str(cohort_root),
        "cohorts": reports,
        "errors": errors,
    }
    write_json_atomic(report, cohort_root / "verification.json")
    if errors:
        raise RuntimeError("Four-class cohort verification failed: " + "; ".join(errors[:10]))
    return report
