from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .reproducibility import sha256, write_json_atomic


BINARY_CLASS_MAPPING = {"normal": 0, "glaucoma": 1}
COHORT_NAME = "ukb_record_glaucoma_binary_bilateral"
DEFAULT_SAMPLING_SEED = "ukb-record-glaucoma-bilateral-v1"
SPLITS = ("train", "validation", "test")
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
IMAGE_COLUMNS = (
    "left_fundus_path",
    "left_oct_path",
    "right_fundus_path",
    "right_oct_path",
)
OBJECTIVE_EVIDENCE = {"41270", "41271", "41272", "glaucoma_treatment"}
SELF_REPORT_PAIR = {"6148", "20002"}
ANALYSIS_ADEQUACY_GATES = {
    "high_confidence_cases": 500,
    "training_cases": 350,
    "validation_cases": 100,
    "internal_test_cases": 100,
}


def _stable_key(seed: str, *values: object) -> str:
    identity = ":".join((seed, *(str(value) for value in values)))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _split_exact(frame: pd.DataFrame, seed: str, group_column: str = "label_name") -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for label, group in frame.groupby(group_column, sort=True, dropna=False):
        ordered = group.copy()
        ordered["_split_key"] = [
            _stable_key(seed, "split", label, participant)
            for participant in ordered["participant_id"]
        ]
        ordered = ordered.sort_values("_split_key", kind="stable").reset_index(drop=True)
        count = len(ordered)
        train_count = int(math.floor(count * SPLIT_FRACTIONS[0]))
        validation_count = int(math.floor(count * SPLIT_FRACTIONS[1]))
        ordered["split"] = (
            ["train"] * train_count
            + ["validation"] * validation_count
            + ["test"] * (count - train_count - validation_count)
        )
        chunks.append(ordered.drop(columns="_split_key"))
    if not chunks:
        return frame.assign(split=pd.Series(dtype=str))
    return pd.concat(chunks, ignore_index=True)


def _age(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _age_bin(value: object) -> str:
    age = _age(value)
    return "unknown" if not np.isfinite(age) else str(int(age // 5) * 5)


def _evidence_set(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {item for item in str(value).split(";") if item}


def is_high_confidence_glaucoma(evidence_sources: object) -> bool:
    evidence = _evidence_set(evidence_sources)
    return bool(evidence & OBJECTIVE_EVIDENCE) or SELF_REPORT_PAIR <= evidence


def _prepare_rows(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        "participant_id",
        "instance",
        "assessment_date",
        "assessment_age",
        "sex",
        "assessment_centre",
        "candidate_status",
        "candidate_label",
        "reference_standard_type",
        "evidence_sources",
        *IMAGE_COLUMNS,
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Phenotype candidates are missing columns: {sorted(missing)}")
    if candidates["participant_id"].duplicated().any():
        raise ValueError("Phenotype candidates must contain one bilateral visit per participant")

    glaucoma = candidates.loc[
        (candidates["candidate_status"] == "prevalent_case")
        & (candidates["candidate_label"] == "glaucoma")
    ].copy()
    glaucoma["high_confidence"] = glaucoma["evidence_sources"].map(
        is_high_confidence_glaucoma
    )
    controls = candidates.loc[
        (candidates["candidate_status"] == "strict_control")
        & (candidates["candidate_label"] == "normal")
    ].copy()
    controls["high_confidence"] = True
    incident = candidates.loc[
        (candidates["candidate_status"] == "incident_case")
        & (candidates["candidate_label"] == "glaucoma")
    ].copy()
    incident["high_confidence"] = False

    for frame in (glaucoma, controls, incident):
        frame["label_name"] = np.where(
            frame["candidate_label"].eq("glaucoma"), "glaucoma", "normal"
        )
        frame["label_id"] = frame["label_name"].map(BINARY_CLASS_MAPPING).astype(int)
        frame["age_5year_bin"] = frame["assessment_age"].map(_age_bin)
        frame["reference_source"] = (
            "UKB self-report, HES/ICD, diagnosis timing, and procedure records"
        )
        frame["reference_standard_type"] = "record_derived_glaucoma_phenotype"
        frame["phenotype_profile"] = COHORT_NAME
    return glaucoma, controls, incident


def _match_controls(
    cases: pd.DataFrame,
    controls: pd.DataFrame,
    seed: str,
    split: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    available = controls.copy()
    selected: list[int] = []
    tiers: Counter[str] = Counter()
    ordered_cases = cases.assign(
        _case_key=[_stable_key(seed, "case", split, value) for value in cases.participant_id]
    ).sort_values("_case_key", kind="stable")
    for case in ordered_cases.itertuples():
        case_sex = str(case.sex)
        case_age_bin = str(case.age_5year_bin)
        case_centre = str(case.assessment_centre)
        pools = (
            (
                "age_sex_centre",
                available.loc[
                    available["sex"].astype(str).eq(case_sex)
                    & available["age_5year_bin"].astype(str).eq(case_age_bin)
                    & available["assessment_centre"].astype(str).eq(case_centre)
                ],
            ),
            (
                "age_sex",
                available.loc[
                    available["sex"].astype(str).eq(case_sex)
                    & available["age_5year_bin"].astype(str).eq(case_age_bin)
                ],
            ),
            ("sex_nearest_age", available.loc[available["sex"].astype(str).eq(case_sex)]),
            ("nearest_age", available),
        )
        for tier, pool in pools:
            if pool.empty:
                continue
            ranked = pool.copy()
            ranked["_age_distance"] = (
                pd.to_numeric(ranked["assessment_age"], errors="coerce") - _age(case.assessment_age)
            ).abs().fillna(float("inf"))
            ranked["_centre_penalty"] = (
                ~ranked["assessment_centre"].astype(str).eq(case_centre)
            ).astype(int)
            ranked["_match_key"] = [
                _stable_key(seed, "control", split, case.participant_id, value)
                for value in ranked["participant_id"]
            ]
            index = ranked.sort_values(
                ["_centre_penalty", "_age_distance", "_match_key"], kind="stable"
            ).index[0]
            selected.append(index)
            available = available.drop(index=index)
            tiers[tier] += 1
            break
        else:
            raise RuntimeError(f"No control remained for case {case.participant_id}")
    result = controls.loc[selected].copy()
    return result, {
        "cases": int(len(cases)),
        "controls": int(len(result)),
        "matching_tiers": dict(sorted(tiers.items())),
    }


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        split: {
            name: int(((frame["split"] == split) & (frame["label_name"] == name)).sum())
            for name in BINARY_CLASS_MAPPING
        }
        for split in SPLITS
    }


def _standardized_differences(primary: pd.DataFrame) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for split in SPLITS:
        frame = primary.loc[primary["split"] == split]
        cases = frame.loc[frame.label_id == 1]
        controls = frame.loc[frame.label_id == 0]
        case_age = pd.to_numeric(cases.assessment_age, errors="coerce")
        control_age = pd.to_numeric(controls.assessment_age, errors="coerce")
        pooled = math.sqrt((float(case_age.var(ddof=1)) + float(control_age.var(ddof=1))) / 2)
        age_smd = abs(float(case_age.mean() - control_age.mean())) / pooled if pooled else 0.0
        case_sex = float(pd.to_numeric(cases.sex, errors="coerce").mean())
        control_sex = float(pd.to_numeric(controls.sex, errors="coerce").mean())
        p = (case_sex + control_sex) / 2
        sex_smd = abs(case_sex - control_sex) / math.sqrt(max(2 * p * (1 - p), 1e-12))
        centres = sorted(
            set(cases.assessment_centre.astype(str))
            | set(controls.assessment_centre.astype(str))
        )
        centre_gap = max(
            (
                abs(
                    float(cases.assessment_centre.astype(str).eq(value).mean())
                    - float(controls.assessment_centre.astype(str).eq(value).mean())
                )
                for value in centres
            ),
            default=0.0,
        )
        report[split] = {
            "age_smd": age_smd,
            "sex_smd": sex_smd,
            "maximum_centre_proportion_difference": centre_gap,
        }
    return report


def _analysis_readiness(primary: pd.DataFrame) -> dict[str, Any]:
    counts = _counts(primary)
    observed = {
        "high_confidence_cases": int((primary.label_id == 1).sum()),
        "training_cases": counts["train"]["glaucoma"],
        "validation_cases": counts["validation"]["glaucoma"],
        "internal_test_cases": counts["test"]["glaucoma"],
    }
    gates = {
        name: {
            "observed": observed[name],
            "minimum": minimum,
            "passed": observed[name] >= minimum,
        }
        for name, minimum in ANALYSIS_ADEQUACY_GATES.items()
    }
    ready = all(item["passed"] for item in gates.values())
    return {
        "status": "CONDITIONAL_PASS" if ready else "NOT_READY",
        "method_development_and_internal_testing_ready": ready,
        "clinical_deployment_validation_ready": False,
        "gates": gates,
        "interpretation": (
            "The cohort supports controlled internal LOOK development. Labels are record-derived, "
            "and independent expert-graded external validation is unavailable."
        ),
    }


def derive_glaucoma_cohorts(
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
            existing.get("schema_version") == 4
            and existing.get("status") == "PASS"
            and existing.get("source_candidate_sha256") == source_hash
            and existing.get("sampling_seed") == sampling_seed
            and outputs
            and all(
                Path(item["path"]).is_file()
                and sha256(Path(item["path"])) == item["sha256"]
                for item in outputs
            )
        ):
            return existing

    candidates = pd.read_csv(candidate_csv, dtype={"participant_id": str, "sex": str})
    glaucoma, controls, incident = _prepare_rows(candidates)
    high = _split_exact(glaucoma.loc[glaucoma.high_confidence].copy(), sampling_seed)
    low = _split_exact(
        glaucoma.loc[~glaucoma.high_confidence].copy(), f"{sampling_seed}:low-confidence"
    )
    controls = _split_exact(controls, sampling_seed)
    glaucoma = pd.concat((high, low), ignore_index=True)
    incident = _split_exact(incident, f"{sampling_seed}:incident")

    primary_parts: list[pd.DataFrame] = []
    matching: dict[str, Any] = {}
    for split in SPLITS:
        split_cases = high.loc[high.split == split].copy()
        split_controls = controls.loc[controls.split == split].copy()
        matched, matching[split] = _match_controls(
            split_cases, split_controls, sampling_seed, split
        )
        primary_parts.append(pd.concat((split_cases, matched), ignore_index=True))
    primary = pd.concat(primary_parts, ignore_index=True)
    natural = pd.concat((glaucoma, controls), ignore_index=True)

    sort_columns = ["split", "label_id", "participant_id"]
    primary = primary.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    natural = natural.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    incident = incident.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    tables = {"primary": primary, "natural": natural, "incident": incident}
    mapping = {
        "dataset": "UKB bilateral CFP plus central OCT glaucoma benchmark",
        "phenotype_profile": COHORT_NAME,
        "classes": [{"id": value, "name": key} for key, value in BINARY_CLASS_MAPPING.items()],
        "reference_standard_type": "record_derived_glaucoma_phenotype",
        "unit_of_analysis": "participant at earliest complete bilateral imaging visit",
        "primary_case_rule": (
            "pre-imaging HES ICD, glaucoma-specific treatment/procedure, or concordant "
            "6148 and 20002 self-report"
        ),
        "sampling_seed": sampling_seed,
    }
    outputs: list[Path] = []
    for name, frame in tables.items():
        labels = cohort_root / name / "reference_labels.csv"
        classes = cohort_root / name / "class_mapping.json"
        _write_table(frame, labels)
        write_json_atomic(mapping, classes)
        outputs.extend((labels, classes))

    flow = {
        "candidate_rows": int(len(candidates)),
        "all_prevalent_glaucoma": int(len(glaucoma)),
        "high_confidence_glaucoma": int(len(high)),
        "single_source_glaucoma": int(len(glaucoma) - len(high)),
        "strict_controls": int(len(controls)),
        "incident_glaucoma": int(len(incident)),
        "primary_rows": int(len(primary)),
        "primary_split_class_counts": _counts(primary),
        "natural_split_class_counts": _counts(natural),
        "matching": matching,
        "standardized_differences": _standardized_differences(primary),
        "evidence_source_counts": dict(sorted(Counter(high.evidence_sources.fillna("")).items())),
    }
    flow_path = cohort_root / "cohort_flow.json"
    write_json_atomic(flow, flow_path)
    outputs.append(flow_path)
    manifest = {
        "schema_version": 4,
        "status": "PASS",
        "cohort": COHORT_NAME,
        "source_candidate_csv": str(candidate_csv),
        "source_candidate_sha256": source_hash,
        "sampling_seed": sampling_seed,
        "reference_standard": "record_derived_glaucoma_phenotype",
        "flow": flow,
        "outputs": [
            {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in outputs
        ],
    }
    write_json_atomic(manifest, manifest_path)
    return manifest


def _validate_table(
    frame: pd.DataFrame,
    name: str,
    *,
    require_both_classes: bool = True,
    require_complete_splits: bool = True,
) -> list[str]:
    required = {
        "participant_id",
        "instance",
        "label_id",
        "label_name",
        "split",
        "reference_standard_type",
        "candidate_status",
        *IMAGE_COLUMNS,
    }
    missing = required - set(frame.columns)
    if missing:
        return [f"{name}: missing columns {sorted(missing)}"]
    errors: list[str] = []
    if frame.participant_id.duplicated().any():
        errors.append(f"{name}: duplicate participants")
    if require_complete_splits and set(frame.split) != set(SPLITS):
        errors.append(f"{name}: incomplete split set")
    observed = dict(
        frame[["label_name", "label_id"]].drop_duplicates().sort_values("label_id").values
    )
    expected = (
        BINARY_CLASS_MAPPING
        if require_both_classes
        else {label: BINARY_CLASS_MAPPING[label] for label in observed}
    )
    if observed != expected:
        errors.append(f"{name}: class mapping differs: {observed}")
    return errors


def verify_glaucoma_cohorts(
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
        if (
            not path.is_file()
            or path.stat().st_size != int(output["bytes"])
            or sha256(path) != output["sha256"]
        ):
            errors.append(f"manifest output missing or changed: {path}")
    source = Path(manifest.get("source_candidate_csv", ""))
    if not source.is_file() or sha256(source) != manifest.get("source_candidate_sha256"):
        errors.append("phenotype candidate source is missing or changed")

    frames: dict[str, pd.DataFrame] = {}
    reports: dict[str, Any] = {}
    for name in ("primary", "natural", "incident"):
        path = cohort_root / name / "reference_labels.csv"
        frame = pd.read_csv(path, dtype={"participant_id": str, "sex": str})
        frames[name] = frame
        if name != "incident" or not frame.empty:
            errors.extend(
                _validate_table(
                    frame,
                    name,
                    require_both_classes=name != "incident",
                    require_complete_splits=name != "incident",
                )
            )
        missing_images = 0
        if check_images:
            for column in IMAGE_COLUMNS:
                missing_images += sum(not (image_root / relative).is_file() for relative in frame[column])
        if missing_images:
            errors.append(f"{name}: {missing_images} referenced images are missing")
        reports[name] = {
            "rows": int(len(frame)),
            "participants": int(frame.participant_id.nunique()),
            "split_class_counts": _counts(frame) if not frame.empty else {},
            "missing_images": missing_images,
            "labels_sha256": sha256(path),
        }

    primary = frames["primary"]
    natural = frames["natural"]
    if not set(primary.participant_id) <= set(natural.participant_id):
        errors.append("primary cohort is not a subset of the natural cohort")
    shared_splits = primary[["participant_id", "split"]].merge(
        natural[["participant_id", "split"]],
        on="participant_id",
        suffixes=("_primary", "_natural"),
    )
    if not shared_splits["split_primary"].eq(shared_splits["split_natural"]).all():
        errors.append("primary and natural cohorts assign shared participants to different splits")
    if set(natural.participant_id) & set(frames["incident"].participant_id):
        errors.append("prevalent/control and incident cohorts overlap")
    for split in SPLITS:
        split_counts = primary.loc[primary.split == split, "label_name"].value_counts()
        if split_counts.get("normal", 0) != split_counts.get("glaucoma", 0):
            errors.append(f"{split}: primary cohort is not 1:1 matched")
    cases = primary.loc[primary.label_name == "glaucoma"]
    if not cases.high_confidence.astype(bool).all():
        errors.append("primary cohort contains non-high-confidence glaucoma cases")
    if set(primary.loc[primary.label_name == "normal", "candidate_status"]) != {"strict_control"}:
        errors.append("primary controls are not strict controls")

    matching = _standardized_differences(primary)
    for split, values in matching.items():
        if values["age_smd"] > 0.10:
            errors.append(f"{split}: age SMD exceeds 0.10")
        if values["sex_smd"] > 0.10:
            errors.append(f"{split}: sex SMD exceeds 0.10")
        if values["maximum_centre_proportion_difference"] > 0.10:
            errors.append(f"{split}: centre proportion difference exceeds 0.10")

    readiness = _analysis_readiness(primary)
    write_json_atomic(readiness, cohort_root / "analysis_readiness.json")
    report = {
        "status": "PASS" if not errors else "FAIL",
        "reference_standard": "record_derived_glaucoma_phenotype",
        "unit_of_analysis": "participant_bilateral_visit",
        "image_root": str(image_root),
        "cohort_root": str(cohort_root),
        "cohorts": reports,
        "matching_balance": matching,
        "analysis_readiness": readiness,
        "errors": errors,
    }
    write_json_atomic(report, cohort_root / "verification.json")
    if errors:
        raise RuntimeError("Glaucoma cohort verification failed: " + "; ".join(errors[:10]))
    return report
