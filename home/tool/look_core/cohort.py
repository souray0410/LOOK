from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .phenotype import PHENOTYPE_PROFILE
from .reproducibility import sha256, write_json_atomic


FOUR_CLASS_MAPPING = {
    "normal": 0,
    "diabetes_related_eye_disease": 1,
    "glaucoma": 2,
    "macular_degeneration": 3,
}
COHORT_NAME = PHENOTYPE_PROFILE
DEFAULT_SAMPLING_SEED = "ukb-record-prevalent-bilateral-v1"
SPLITS = ("train", "validation", "test")
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
IMAGE_COLUMNS = (
    "left_fundus_path",
    "left_oct_path",
    "right_fundus_path",
    "right_oct_path",
)
ANALYSIS_ADEQUACY_GATES = {
    "balanced_participants": 2000,
    "disease_participants_per_class": 400,
    "training_participants_per_disease_class": 250,
    "validation_participants_per_class": 60,
    "internal_test_participants_per_class": 60,
}


def _stable_key(seed: str, *values: object) -> str:
    identity = ":".join((seed, *(str(value) for value in values)))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _split_exact(frame: pd.DataFrame, seed: str) -> pd.DataFrame:
    """Assign exact deterministic 70/15/15 splits within each label."""
    chunks = []
    for label, group in frame.groupby("label_name", sort=True):
        group = group.copy()
        group["_split_key"] = [
            _stable_key(seed, "split", label, participant)
            for participant in group["participant_id"]
        ]
        group = group.sort_values("_split_key", kind="stable").reset_index(drop=True)
        count = len(group)
        train_count = int(math.floor(count * SPLIT_FRACTIONS[0]))
        validation_count = int(math.floor(count * SPLIT_FRACTIONS[1]))
        split = (
            ["train"] * train_count
            + ["validation"] * validation_count
            + ["test"] * (count - train_count - validation_count)
        )
        group["split"] = split
        chunks.append(group.drop(columns="_split_key"))
    return pd.concat(chunks, ignore_index=True)


def _age_bin(value: object) -> str:
    try:
        age = float(value)
    except (TypeError, ValueError):
        return "unknown"
    return str(int(age // 5) * 5)


def _allocate_quotas(strata: Counter[tuple[str, str]], target: int) -> dict[tuple[str, str], int]:
    total = sum(strata.values())
    if target <= 0 or total <= 0:
        return {}
    exact = {key: target * count / total for key, count in strata.items()}
    quotas = {key: int(math.floor(value)) for key, value in exact.items()}
    remainder = target - sum(quotas.values())
    order = sorted(exact, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[:remainder]:
        quotas[key] += 1
    return quotas


def _matched_controls(split_frame: pd.DataFrame, seed: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    diseases = split_frame.loc[split_frame["label_name"] != "normal"].copy()
    controls = split_frame.loc[split_frame["label_name"] == "normal"].copy()
    disease_counts = diseases["label_name"].value_counts()
    target = int(disease_counts.max())
    if target <= 0 or len(controls) < target:
        raise ValueError(f"Insufficient disease or control samples: target={target}, controls={len(controls)}")

    diseases["_stratum"] = list(zip(diseases["age_5year_bin"], diseases["sex"].fillna("unknown")))
    controls["_stratum"] = list(zip(controls["age_5year_bin"], controls["sex"].fillna("unknown")))
    quotas = _allocate_quotas(Counter(diseases["_stratum"]), target)
    selected_indices: list[int] = []
    shortfall = 0
    for stratum, quota in sorted(quotas.items()):
        candidates = controls.loc[controls["_stratum"] == stratum].copy()
        candidates["_match_key"] = [
            _stable_key(seed, "normal", split_frame["split"].iloc[0], participant)
            for participant in candidates["participant_id"]
        ]
        chosen = candidates.sort_values("_match_key", kind="stable").head(quota)
        selected_indices.extend(chosen.index.tolist())
        shortfall += max(0, quota - len(chosen))

    if shortfall:
        remaining = controls.drop(index=selected_indices).copy()
        remaining["_match_key"] = [
            _stable_key(seed, "normal-fallback", split_frame["split"].iloc[0], participant)
            for participant in remaining["participant_id"]
        ]
        selected_indices.extend(
            remaining.sort_values("_match_key", kind="stable").head(shortfall).index.tolist()
        )
    selected_controls = controls.loc[selected_indices].drop(columns="_stratum")
    report = {
        "target_controls": target,
        "selected_controls": int(len(selected_controls)),
        "exact_stratum_matches": int(target - shortfall),
        "fallback_matches": int(shortfall),
        "disease_class_counts": {key: int(value) for key, value in disease_counts.items()},
    }
    return selected_controls, report


def _prepare_rows(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "participant_id", "instance", "assessment_date", "assessment_age", "sex",
        "candidate_status", "candidate_label", "reference_standard_type",
        *IMAGE_COLUMNS,
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Phenotype candidates are missing columns: {sorted(missing)}")
    if candidates["participant_id"].duplicated().any():
        raise ValueError("Phenotype candidates must contain exactly one visit per participant")
    eligible = candidates.loc[
        candidates["candidate_status"].isin({"prevalent_case", "strict_control"})
    ].copy()
    eligible = eligible.rename(columns={"candidate_label": "label_name"})
    eligible["label_id"] = eligible["label_name"].map(FOUR_CLASS_MAPPING)
    if eligible["label_id"].isna().any():
        raise ValueError("Eligible row has an unsupported label")
    eligible["label_id"] = eligible["label_id"].astype(int)
    eligible["age_5year_bin"] = eligible["assessment_age"].map(_age_bin)
    eligible["reference_source"] = "UKB self-report, HES/ICD, diagnosis timing, and procedure records"
    incident = candidates.loc[candidates["candidate_status"] == "incident_case"].copy()
    incident = incident.rename(columns={"candidate_label": "label_name"})
    incident["label_id"] = incident["label_name"].map(FOUR_CLASS_MAPPING).astype(int)
    incident["age_5year_bin"] = incident["assessment_age"].map(_age_bin)
    incident["reference_source"] = "post-imaging incident record-derived phenotype"
    return eligible, incident


def _counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        split: {
            label: int(((frame["split"] == split) & (frame["label_name"] == label)).sum())
            for label in FOUR_CLASS_MAPPING
        }
        for split in SPLITS
    }


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _cohort_characteristics(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for cohort_name, frame in tables.items():
        for (split, label), group in frame.groupby(["split", "label_name"], sort=True):
            ages = pd.to_numeric(group["assessment_age"], errors="coerce")
            sex = group["sex"] if "sex" in group else pd.Series("unknown", index=group.index)
            sex_counts = sex.fillna("unknown").astype(str).value_counts()
            centres = group["assessment_centre"] if "assessment_centre" in group else pd.Series(dtype=object)
            records.append({
                "cohort": cohort_name,
                "split": split,
                "label_name": label,
                "participants": int(len(group)),
                "age_available": int(ages.notna().sum()),
                "age_mean": float(ages.mean()) if ages.notna().any() else None,
                "age_std": float(ages.std(ddof=1)) if ages.notna().sum() > 1 else None,
                "age_median": float(ages.median()) if ages.notna().any() else None,
                "age_q1": float(ages.quantile(0.25)) if ages.notna().any() else None,
                "age_q3": float(ages.quantile(0.75)) if ages.notna().any() else None,
                "sex_0": int(sex_counts.get("0", 0)),
                "sex_1": int(sex_counts.get("1", 0)),
                "sex_unknown": int(sex_counts.get("unknown", 0)),
                "assessment_centres": int(centres.nunique(dropna=True)),
            })
    return pd.DataFrame.from_records(records)


def _evidence_counts(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cohort_name, frame in tables.items():
        cohort_counts: dict[str, Any] = {}
        for label, group in frame.groupby("label_name", sort=True):
            counts: Counter[str] = Counter()
            evidence = group["evidence_sources"] if "evidence_sources" in group else pd.Series(dtype=object)
            for value in evidence.fillna(""):
                counts.update(item for item in str(value).split(";") if item)
            cohort_counts[label] = dict(sorted(counts.items()))
        result[cohort_name] = cohort_counts
    return result


def _wilson_half_width(sample_size: int, rate: float, z: float = 1.96) -> float | None:
    if sample_size <= 0:
        return None
    denominator = 1.0 + z * z / sample_size
    radius = z * math.sqrt(
        rate * (1.0 - rate) / sample_size + z * z / (4.0 * sample_size * sample_size)
    ) / denominator
    return float(radius)


def _analysis_readiness(balanced: pd.DataFrame) -> dict[str, Any]:
    split_counts = _counts(balanced)
    total_counts = {
        label: int((balanced["label_name"] == label).sum())
        for label in FOUR_CLASS_MAPPING
    }
    disease_labels = [label for label in FOUR_CLASS_MAPPING if label != "normal"]
    observed = {
        "balanced_participants": int(len(balanced)),
        "disease_participants_per_class": min(total_counts[label] for label in disease_labels),
        "training_participants_per_disease_class": min(
            split_counts["train"][label] for label in disease_labels
        ),
        "validation_participants_per_class": min(split_counts["validation"].values()),
        "internal_test_participants_per_class": min(split_counts["test"].values()),
    }
    gates = {
        name: {
            "observed": int(observed[name]),
            "minimum": int(minimum),
            "passed": bool(observed[name] >= minimum),
        }
        for name, minimum in ANALYSIS_ADEQUACY_GATES.items()
    }
    method_ready = all(item["passed"] for item in gates.values())
    precision = {
        label: {
            "n": int(sample_size),
            "wilson_95_half_width_at_rate_0_50": _wilson_half_width(sample_size, 0.50),
            "wilson_95_half_width_at_rate_0_70": _wilson_half_width(sample_size, 0.70),
        }
        for label, sample_size in split_counts["test"].items()
    }
    return {
        "status": "CONDITIONAL_PASS" if method_ready else "NOT_READY",
        "method_development_and_internal_testing_ready": method_ready,
        "clinical_deployment_validation_ready": False,
        "gates": gates,
        "balanced_total_class_counts": total_counts,
        "internal_test_precision": precision,
        "interpretation": (
            "Protocol-defined minimums support controlled method development and paired internal "
            "testing; they are not TMI or MIA acceptance rules. Per-class uncertainty must be "
            "reported. Expert image grading and independent external testing are unavailable, so "
            "clinical deployment claims are out of scope."
        ),
    }


def derive_four_class_cohorts(
    candidate_csv: Path,
    cohort_root: Path,
    *,
    sampling_seed: str = DEFAULT_SAMPLING_SEED,
) -> dict[str, Any]:
    candidate_csv = Path(candidate_csv).resolve()
    cohort_root = Path(cohort_root).resolve()
    source_hash = sha256(candidate_csv)
    manifest_path = cohort_root / "cohort_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        outputs = existing.get("outputs", [])
        if (
            existing.get("schema_version") == 3
            and existing.get("status") == "PASS"
            and existing.get("source_candidate_sha256") == source_hash
            and existing.get("sampling_seed") == sampling_seed
            and outputs
            and all(Path(item["path"]).is_file() and sha256(Path(item["path"])) == item["sha256"] for item in outputs)
        ):
            return existing

    candidates = pd.read_csv(candidate_csv, dtype={"participant_id": str, "sex": str})
    eligible, incident = _prepare_rows(candidates)
    natural = _split_exact(eligible, sampling_seed)
    incident = _split_exact(incident, f"{sampling_seed}:incident") if not incident.empty else incident.assign(split=pd.Series(dtype=str))

    balanced_parts = []
    matching: dict[str, Any] = {}
    for split in SPLITS:
        split_frame = natural.loc[natural["split"] == split].copy()
        controls, matching[split] = _matched_controls(split_frame, sampling_seed)
        balanced_parts.append(pd.concat((split_frame.loc[split_frame["label_name"] != "normal"], controls), ignore_index=True))
    balanced = pd.concat(balanced_parts, ignore_index=True)

    sort_columns = ["split", "label_id", "participant_id"]
    natural = natural.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    balanced = balanced.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    incident = incident.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    mapping = {
        "dataset": "UKB bilateral CFP plus central OCT participant-level four-class cohort",
        "phenotype_profile": PHENOTYPE_PROFILE,
        "classes": [{"id": value, "name": key} for key, value in FOUR_CLASS_MAPPING.items()],
        "reference_standard_type": "record_derived_clinical_phenotype",
        "unit_of_analysis": "participant at earliest complete bilateral imaging visit",
        "sampling_seed": sampling_seed,
    }
    cohort_tables = {"balanced": balanced, "natural": natural, "incident": incident}
    outputs: list[Path] = []
    for name, frame in cohort_tables.items():
        directory = cohort_root / name
        labels = directory / "reference_labels.csv"
        classes = directory / "class_mapping.json"
        _write_table(frame, labels)
        write_json_atomic(mapping, classes)
        outputs.extend((labels, classes))

    flow = {
        "candidate_rows": int(len(candidates)),
        "status_counts": {key: int(value) for key, value in candidates["candidate_status"].value_counts().items()},
        "natural_rows": int(len(natural)),
        "balanced_rows": int(len(balanced)),
        "incident_rows": int(len(incident)),
        "natural_split_class_counts": _counts(natural),
        "balanced_split_class_counts": _counts(balanced),
        "normal_matching": matching,
    }
    flow_path = cohort_root / "cohort_flow.json"
    write_json_atomic(flow, flow_path)
    outputs.append(flow_path)
    characteristics_path = cohort_root / "cohort_characteristics.csv"
    _write_table(_cohort_characteristics(cohort_tables), characteristics_path)
    evidence_path = cohort_root / "evidence_source_counts.json"
    write_json_atomic(_evidence_counts(cohort_tables), evidence_path)
    outputs.extend((characteristics_path, evidence_path))
    manifest = {
        "schema_version": 3,
        "status": "PASS",
        "cohort": COHORT_NAME,
        "source_candidate_csv": str(candidate_csv),
        "source_candidate_sha256": source_hash,
        "sampling_seed": sampling_seed,
        "reference_standard": "record_derived_clinical_phenotype",
        "flow": flow,
        "outputs": [
            {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in outputs
        ],
    }
    write_json_atomic(manifest, manifest_path)
    return manifest


def _validate_table(
    frame: pd.DataFrame, cohort_name: str, *, require_all_classes: bool = True
) -> list[str]:
    errors = []
    required = {
        "participant_id", "instance", "label_id", "label_name", "split",
        "reference_standard_type", "candidate_status", *IMAGE_COLUMNS,
    }
    missing = required - set(frame.columns)
    if missing:
        return [f"{cohort_name}: missing columns {sorted(missing)}"]
    if frame["participant_id"].duplicated().any():
        errors.append(f"{cohort_name}: duplicate participants")
    if require_all_classes and set(frame["split"]) != set(SPLITS):
        errors.append(f"{cohort_name}: incomplete split set")
    observed = dict(frame[["label_name", "label_id"]].drop_duplicates().values)
    expected = FOUR_CLASS_MAPPING if require_all_classes else {
        name: FOUR_CLASS_MAPPING[name] for name in observed
    }
    if observed != expected:
        errors.append(f"{cohort_name}: class mapping differs: {observed}")
    if set(frame["reference_standard_type"]) != {"record_derived_clinical_phenotype"}:
        errors.append(f"{cohort_name}: incorrect reference-standard type")
    return errors


def verify_four_class_cohorts(
    cohort_root: Path,
    image_root: Path,
    *,
    check_images: bool = True,
) -> dict[str, Any]:
    cohort_root = Path(cohort_root).resolve()
    image_root = Path(image_root).resolve()
    manifest = json.loads((cohort_root / "cohort_manifest.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    for output in manifest.get("outputs", []):
        path = Path(output["path"])
        if not path.is_file() or path.stat().st_size != int(output["bytes"]) or sha256(path) != output["sha256"]:
            errors.append(f"manifest output missing or changed: {path}")
    source = Path(manifest.get("source_candidate_csv", ""))
    if not source.is_file() or sha256(source) != manifest.get("source_candidate_sha256"):
        errors.append("phenotype candidate source is missing or changed")

    frames: dict[str, pd.DataFrame] = {}
    reports: dict[str, Any] = {}
    for cohort_name in ("balanced", "natural", "incident"):
        labels_path = cohort_root / cohort_name / "reference_labels.csv"
        frame = pd.read_csv(labels_path, dtype={"participant_id": str, "sex": str})
        frames[cohort_name] = frame
        if cohort_name != "incident" or not frame.empty:
            errors.extend(
                _validate_table(
                    frame, cohort_name, require_all_classes=cohort_name != "incident"
                )
            )
        missing_images = 0
        if check_images:
            for column in IMAGE_COLUMNS:
                missing_images += sum(not (image_root / relative).is_file() for relative in frame[column])
        if missing_images:
            errors.append(f"{cohort_name}: {missing_images} referenced images are missing")
        counts = _counts(frame) if not frame.empty else {}
        if cohort_name == "balanced":
            for split in SPLITS:
                values = [value for value in counts[split].values() if value > 0]
                if len(values) != 4 or max(values) / min(values) > 2.5:
                    errors.append(f"{split}: balanced class ratio exceeds 2.5:1 or a class is absent")
        reports[cohort_name] = {
            "rows": int(len(frame)),
            "participants": int(frame["participant_id"].nunique()),
            "split_class_counts": counts,
            "missing_images": missing_images,
            "labels_sha256": sha256(labels_path),
        }

    balanced_ids = set(frames["balanced"]["participant_id"])
    natural_ids = set(frames["natural"]["participant_id"])
    incident_ids = set(frames["incident"]["participant_id"])
    if not balanced_ids <= natural_ids:
        errors.append("balanced cohort is not a subset of natural cohort")
    if natural_ids & incident_ids:
        errors.append("prevalent/control and incident cohorts overlap")
    if set(frames["balanced"].loc[frames["balanced"]["label_name"] == "normal", "candidate_status"]) != {"strict_control"}:
        errors.append("balanced Normal rows are not strict controls")

    readiness = _analysis_readiness(frames["balanced"])
    write_json_atomic(readiness, cohort_root / "analysis_readiness.json")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "reference_standard": "record_derived_clinical_phenotype",
        "unit_of_analysis": "participant_bilateral_visit",
        "image_root": str(image_root),
        "cohort_root": str(cohort_root),
        "cohorts": reports,
        "analysis_readiness": readiness,
        "errors": errors,
    }
    write_json_atomic(report, cohort_root / "verification.json")
    if errors:
        raise RuntimeError("Four-class cohort verification failed: " + "; ".join(errors[:10]))
    return report
