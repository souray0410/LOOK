from __future__ import annotations

import csv
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from look.runtime.provenance import sha256, write_json_atomic


PHENOTYPE_PROFILE = "ukb_record_eye_phenotype_candidates"
TARGETS = ("diabetes_related_eye_disease", "glaucoma", "macular_degeneration")
SELECTED_FIELD_IDS = {
    "31", "53", "54", "21003",
    "6148", "5890", "6119", "5912", "5441", "5419", "5934",
    "5326", "5327",
    "20002", "20009", "20004", "20011",
    "41270", "41280", "41271", "41281", "41272", "41282",
}


def base_field(column: str) -> str:
    return column.split("-", 1)[0]


def _first_path(value: str) -> str:
    paths = [item for item in str(value).split(";") if item]
    if len(paths) != 1:
        raise ValueError(f"Expected one exported path, got {len(paths)}: {value}")
    return paths[0]


def build_paired_visit_manifest(source_csv: Path, destination: Path) -> dict[str, Any]:
    """Collapse eye rows to one earliest complete bilateral visit per participant."""
    frame = pd.read_csv(source_csv, dtype={"participant_id": str})
    fundus_column = "fundus_path" if "fundus_path" in frame else "fundus_paths"
    oct_column = "oct_path" if "oct_path" in frame else "oct_paths"
    required = ["participant_id", "instance", "eye", fundus_column, oct_column]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Paired source is missing columns: {sorted(missing)}")
    frame = frame.loc[:, required].copy()
    frame["instance"] = frame["instance"].astype(int)
    frame["fundus_path"] = frame[fundus_column].map(_first_path)
    frame["oct_path"] = frame[oct_column].map(_first_path)
    frame = frame.sort_values(["participant_id", "instance", "eye"], kind="stable")

    records: list[dict[str, Any]] = []
    visits = 0
    for participant, group in frame.groupby("participant_id", sort=False):
        chosen = None
        for instance, visit in group.groupby("instance", sort=True):
            by_eye = {str(row.eye): row for row in visit.itertuples(index=False)}
            if set(by_eye) == {"left", "right"}:
                chosen = (int(instance), by_eye)
                break
        if chosen is None:
            continue
        visits += 1
        instance, by_eye = chosen
        records.append({
            "participant_id": participant,
            "instance": instance,
            "left_fundus_path": by_eye["left"].fundus_path,
            "left_oct_path": by_eye["left"].oct_path,
            "right_fundus_path": by_eye["right"].fundus_path,
            "right_oct_path": by_eye["right"].oct_path,
        })
    result = pd.DataFrame.from_records(records).sort_values("participant_id", kind="stable")
    if result.empty or result["participant_id"].duplicated().any():
        raise ValueError("Bilateral visit manifest is empty or contains duplicate participants")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    result.to_csv(temporary, index=False)
    os.replace(temporary, destination)
    return {
        "source": str(source_csv),
        "source_rows": int(len(frame)),
        "complete_bilateral_participants": int(len(result)),
        "selected_visits": visits,
        "destination": str(destination),
        "sha256": sha256(destination),
    }


def extract_selected_table(
    source: Path,
    destination: Path,
    participant_ids: set[str],
    *,
    chunksize: int = 1000,
) -> dict[str, Any]:
    header = pd.read_csv(source, nrows=0).columns.tolist()
    if not header or header[0] != "eid":
        raise ValueError(f"First column is not eid in {source}")
    columns = ["eid", *[column for column in header[1:] if base_field(column) in SELECTED_FIELD_IDS]]
    field_ids = {base_field(column) for column in columns[1:]}
    if not {"31", "53", "21003", "6148", "20002", "41270", "41280"} <= field_ids:
        raise ValueError(f"Required phenotype fields are absent from {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    matched = 0
    wrote_header = False
    for chunk in pd.read_csv(source, usecols=columns, dtype=str, chunksize=chunksize, low_memory=False):
        selected = chunk.loc[chunk["eid"].isin(participant_ids)]
        if selected.empty:
            continue
        selected.to_csv(temporary, mode="a", header=not wrote_header, index=False)
        wrote_header = True
        matched += len(selected)
    if not wrote_header:
        raise ValueError(f"No paired participants were found in {source}")
    os.replace(temporary, destination)
    return {
        "source": str(source),
        "source_bytes": source.stat().st_size,
        "destination": str(destination),
        "destination_bytes": destination.stat().st_size,
        "columns": len(columns),
        "field_ids": sorted(field_ids, key=int),
        "matched_participants": matched,
        "missing_participants": len(participant_ids) - matched,
        "manual_retinal_309xx_present": any(30905 <= int(field) <= 30943 for field in field_ids),
        "sha256": sha256(destination),
    }


def _values(row: Mapping[str, Any], field: str, instance: int | None = None) -> list[str]:
    prefix = f"{field}-{instance}." if instance is not None else f"{field}-"
    return [
        str(value).strip()
        for column, value in row.items()
        if column.startswith(prefix) and pd.notna(value) and str(value).strip()
    ]


def _single(row: Mapping[str, Any], field: str, instance: int) -> str:
    values = _values(row, field, instance)
    return values[0] if values else ""


def _normalize_code(value: str) -> str:
    return value.upper().replace(".", "").replace(" ", "")


def _dated_codes(row: Mapping[str, Any], code_field: str, date_field: str) -> list[tuple[str, str]]:
    result = []
    prefix = f"{code_field}-"
    for column, value in row.items():
        if not column.startswith(prefix) or pd.isna(value) or not str(value).strip():
            continue
        suffix = column[len(prefix):]
        date = row.get(f"{date_field}-{suffix}", "")
        result.append((_normalize_code(str(value)), "" if pd.isna(date) else str(date).strip()))
    return result


def _paired_codes(row: Mapping[str, Any], code_field: str, detail_field: str, instance: int) -> list[tuple[str, str]]:
    result = []
    prefix = f"{code_field}-{instance}."
    for column, value in row.items():
        if not column.startswith(prefix) or pd.isna(value) or not str(value).strip():
            continue
        array = column[len(prefix):]
        detail = row.get(f"{detail_field}-{instance}.{array}", "")
        result.append((_normalize_code(str(value)), "" if pd.isna(detail) else str(detail).strip()))
    return result


def _longitudinal_codes(
    row: Mapping[str, Any], code_field: str, detail_field: str | None = None
) -> list[tuple[int, str, str]]:
    result = []
    prefix = f"{code_field}-"
    for column, value in row.items():
        if not column.startswith(prefix) or pd.isna(value) or not str(value).strip():
            continue
        suffix = column[len(prefix):]
        try:
            instance = int(suffix.split(".", 1)[0])
        except ValueError:
            continue
        detail = row.get(f"{detail_field}-{suffix}", "") if detail_field else ""
        result.append(
            (instance, _normalize_code(str(value)), "" if pd.isna(detail) else str(detail).strip())
        )
    return result


def _date_relation(event_date: str, assessment_date: str) -> str:
    if not event_date or not assessment_date:
        return "unknown"
    try:
        event = pd.Timestamp(event_date)
        assessment = pd.Timestamp(assessment_date)
    except (TypeError, ValueError):
        return "unknown"
    return "prevalent" if event <= assessment else "incident"


def _age_or_report_relation(
    diagnosis_age: str,
    imaging_age: str,
    report_date: str,
    imaging_date: str,
) -> str:
    try:
        if diagnosis_age and imaging_age:
            return "prevalent" if float(diagnosis_age) <= float(imaging_age) else "incident"
    except ValueError:
        pass
    return _date_relation(report_date, imaging_date)


def _target_for_code(code: str, coding: str) -> str | None:
    if coding == "icd10":
        if code.startswith("H353"):
            return "macular_degeneration"
        if code.startswith(("H401", "H408", "H409")):
            return "glaucoma"
        if code.startswith("H360") or (len(code) >= 4 and code[:3] in {"E10", "E11", "E12", "E13", "E14"} and code[3] == "3"):
            return "diabetes_related_eye_disease"
    if coding == "icd9":
        if code.startswith("3625"):
            return "macular_degeneration"
        if code.startswith(("3651", "3658", "3659")):
            return "glaucoma"
        if code.startswith(("3620", "2505")):
            return "diabetes_related_eye_disease"
    return None


def classify_participant_visit(row: Mapping[str, Any], visit: Mapping[str, Any]) -> dict[str, Any]:
    instance = int(visit["instance"])
    assessment_date = _single(row, "53", instance)
    assessment_age = _single(row, "21003", instance)
    sex = _single(row, "31", 0)
    centre = _single(row, "54", instance)
    screen_codes = {_normalize_code(value) for value in _values(row, "6148", instance)}
    interview_records = _paired_codes(row, "20002", "20009", instance)
    interview_codes = {code for code, _ in interview_records}

    evidence: dict[str, list[str]] = {target: [] for target in TARGETS}
    incident: dict[str, list[str]] = {target: [] for target in TARGETS}
    undated: dict[str, list[str]] = {target: [] for target in TARGETS}
    screen_map = {
        "1": "diabetes_related_eye_disease",
        "2": "glaucoma",
        "5": "macular_degeneration",
    }
    interview_map = {
        "1276": "diabetes_related_eye_disease",
        "1277": "glaucoma",
        "1528": "macular_degeneration",
    }
    for report_instance, code, _ in _longitudinal_codes(row, "6148"):
        target = screen_map.get(code)
        if target is None:
            continue
        report_date = _single(row, "53", report_instance)
        relation = "prevalent" if report_instance == instance else _date_relation(report_date, assessment_date)
        item = f"6148:{code}:instance{report_instance}:date={report_date or 'missing'}"
        if relation == "prevalent":
            evidence[target].append(item)
        elif relation == "incident":
            incident[target].append(item)
        else:
            undated[target].append(item)
    for report_instance, code, age in _longitudinal_codes(row, "20002", "20009"):
        target = interview_map.get(code)
        if target is None:
            continue
        report_date = _single(row, "53", report_instance)
        relation = _age_or_report_relation(age, assessment_age, report_date, assessment_date)
        item = f"20002:{code}:diagnosis_age={age or 'missing'}:instance{report_instance}"
        if relation == "prevalent":
            evidence[target].append(item)
        elif relation == "incident":
            incident[target].append(item)
        else:
            undated[target].append(item)

    for coding, code_field, date_field in (
        ("icd10", "41270", "41280"),
        ("icd9", "41271", "41281"),
    ):
        for code, date in _dated_codes(row, code_field, date_field):
            target = _target_for_code(code, coding)
            if target is None:
                continue
            item = f"{code_field}:{code}:{date or 'date_missing'}"
            relation = _date_relation(date, assessment_date)
            if relation == "prevalent":
                evidence[target].append(item)
            elif relation == "incident":
                incident[target].append(item)
            else:
                undated[target].append(item)

    glaucoma_surgery = any(
        value in {"2", "3", "4"}
        for field in ("5326", "5327")
        for value in _values(row, field, instance)
    )
    for report_instance, code, age in _longitudinal_codes(row, "20004", "20011"):
        if code != "1436":
            continue
        report_date = _single(row, "53", report_instance)
        relation = _age_or_report_relation(age, assessment_age, report_date, assessment_date)
        item = f"20004:{code}:operation_age={age or 'missing'}:instance{report_instance}"
        if relation == "prevalent":
            evidence["glaucoma"].append(item)
        elif relation == "incident":
            incident["glaucoma"].append(item)
        else:
            undated["glaucoma"].append(item)
    for code, date in _dated_codes(row, "41272", "41282"):
        if code != "C601":
            continue
        relation = _date_relation(date, assessment_date)
        if relation == "prevalent":
            glaucoma_surgery = True
        elif relation == "incident":
            incident["glaucoma"].append(f"41272:{code}:{date}")
        else:
            undated["glaucoma"].append(f"41272:{code}:date_missing")
    if glaucoma_surgery:
        evidence["glaucoma"].append("glaucoma_treatment:conditional_field_at_imaging")

    prevalent_targets = {target for target, items in evidence.items() if items}
    incident_targets = {target for target, items in incident.items() if items}
    undated_targets = {target for target, items in undated.items() if items}
    lifetime_targets = prevalent_targets | incident_targets | undated_targets
    competing_self_report = bool(screen_codes & {"3", "4", "6"}) or bool(
        interview_codes & {"1278", "1279", "1281", "1282"}
    )
    competing_icd = False
    for code, date in _dated_codes(row, "41270", "41280"):
        if _date_relation(date, assessment_date) != "incident" and code.startswith(("H25", "H26", "H33", "H34", "S05")):
            competing_icd = True
            break
    competing = competing_self_report or competing_icd
    explicit_normal = "-7" in screen_codes

    status, label = "excluded_uncertain", ""
    if not assessment_date:
        status = "excluded_missing_assessment_date"
    elif len(lifetime_targets) > 1:
        status = "excluded_target_comorbidity"
    elif competing:
        status = "excluded_competing_eye_condition"
    elif len(prevalent_targets) == 1:
        status, label = "prevalent_case", next(iter(prevalent_targets))
    elif undated_targets:
        status = "excluded_undated_target_evidence"
    elif not prevalent_targets and len(incident_targets) == 1:
        status, label = "incident_case", next(iter(incident_targets))
    elif explicit_normal and not lifetime_targets:
        status, label = "strict_control", "normal"

    return {
        **visit,
        "assessment_date": assessment_date,
        "assessment_age": assessment_age,
        "sex": sex,
        "assessment_centre": centre,
        "candidate_status": status,
        "candidate_label": label,
        "prevalent_targets": ";".join(sorted(prevalent_targets)),
        "incident_targets": ";".join(sorted(incident_targets)),
        "undated_targets": ";".join(sorted(undated_targets)),
        "evidence_sources": ";".join(sorted({item.split(":", 1)[0] for items in evidence.values() for item in items})),
        "evidence_detail": "|".join(item for target in TARGETS for item in evidence[target]),
        "incident_evidence_detail": "|".join(item for target in TARGETS for item in incident[target]),
        "undated_evidence_detail": "|".join(item for target in TARGETS for item in undated[target]),
        "competing_eye_condition": str(competing).lower(),
        "reference_standard_type": "record_derived_clinical_phenotype",
        "phenotype_profile": PHENOTYPE_PROFILE,
    }


def build_candidate_table(
    matched_phenotype_csv: Path,
    paired_visit_csv: Path,
    destination: Path,
    *,
    chunksize: int = 500,
) -> dict[str, Any]:
    visits = pd.read_csv(paired_visit_csv, dtype={"participant_id": str})
    if visits["participant_id"].duplicated().any():
        raise ValueError("Paired visit table must contain one row per participant")
    visit_map = {str(row.participant_id): row._asdict() for row in visits.itertuples(index=False)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    writer: csv.DictWriter | None = None
    counts: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    processed: set[str] = set()
    with temporary.open("x", newline="", encoding="utf-8") as output_handle:
        for chunk in pd.read_csv(matched_phenotype_csv, dtype=str, chunksize=chunksize, low_memory=False):
            for raw in chunk.to_dict(orient="records"):
                eid = str(raw["eid"])
                visit = visit_map.get(eid)
                if visit is None:
                    continue
                result = classify_participant_visit(raw, visit)
                if writer is None:
                    writer = csv.DictWriter(output_handle, fieldnames=list(result))
                    writer.writeheader()
                writer.writerow(result)
                processed.add(eid)
                counts[result["candidate_status"]] += 1
                if result["candidate_label"]:
                    labels[result["candidate_label"]] += 1
    if writer is None or processed != set(visit_map):
        raise ValueError(f"Phenotype coverage mismatch: processed={len(processed)}, expected={len(visit_map)}")
    os.replace(temporary, destination)
    report = {
        "profile": PHENOTYPE_PROFILE,
        "paired_participants": len(visit_map),
        "processed_participants": len(processed),
        "status_counts": dict(sorted(counts.items())),
        "label_counts": dict(sorted(labels.items())),
        "destination": str(destination),
        "sha256": sha256(destination),
    }
    write_json_atomic(report, destination.with_suffix(".audit.json"))
    return report
