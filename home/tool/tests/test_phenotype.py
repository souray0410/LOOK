import pandas as pd

from look_core.phenotype import (
    build_candidate_table,
    classify_participant_visit,
    extract_selected_table,
)


VISIT = {
    "participant_id": "1001",
    "instance": 0,
    "left_fundus_path": "left_cfp.png",
    "left_oct_path": "left_oct.png",
    "right_fundus_path": "right_cfp.png",
    "right_oct_path": "right_oct.png",
}


def _base_row():
    return {
        "eid": "1001",
        "53-0.0": "2010-01-01",
        "21003-0.0": "60",
        "31-0.0": "1",
        "54-0.0": "11001",
        "6148-0.0": "-7",
    }


def test_icd_timing_separates_prevalent_and_incident_disease():
    prevalent = _base_row() | {"41270-0.0": "E11.3", "41280-0.0": "2009-01-01"}
    result = classify_participant_visit(prevalent, VISIT)
    assert (result["candidate_status"], result["candidate_label"]) == (
        "prevalent_case", "diabetes_related_eye_disease"
    )

    incident = _base_row() | {"41270-0.0": "H40.10", "41280-0.0": "2012-01-01"}
    result = classify_participant_visit(incident, VISIT)
    assert (result["candidate_status"], result["candidate_label"]) == (
        "incident_case", "glaucoma"
    )


def test_undated_target_and_lifetime_target_comorbidity_are_excluded():
    undated = _base_row() | {"41270-0.0": "H35.3"}
    assert classify_participant_visit(undated, VISIT)["candidate_status"] == "excluded_undated_target_evidence"

    comorbid = _base_row() | {
        "41270-0.0": "H36.0", "41280-0.0": "2009-01-01",
        "41270-0.1": "H40.9", "41280-0.1": "2013-01-01",
    }
    assert classify_participant_visit(comorbid, VISIT)["candidate_status"] == "excluded_target_comorbidity"


def test_self_report_age_and_competing_eye_disease_are_auditable():
    amd = _base_row() | {"20002-0.0": "1528", "20009-0.0": "52"}
    result = classify_participant_visit(amd, VISIT)
    assert result["candidate_label"] == "macular_degeneration"
    assert "diagnosis_age=52" in result["evidence_detail"]

    cataract = _base_row() | {"20002-0.0": "1278", "20009-0.0": "55"}
    assert classify_participant_visit(cataract, VISIT)["candidate_status"] == "excluded_competing_eye_condition"


def test_later_record_uses_diagnosis_age_before_report_date():
    diagnosed_before_imaging = _base_row() | {
        "53-1.0": "2014-01-01", "20002-1.0": "1277", "20009-1.0": "55",
    }
    result = classify_participant_visit(diagnosed_before_imaging, VISIT)
    assert (result["candidate_status"], result["candidate_label"]) == (
        "prevalent_case", "glaucoma"
    )

    diagnosed_after_imaging = _base_row() | {
        "53-1.0": "2014-01-01", "20002-1.0": "1277", "20009-1.0": "63",
    }
    result = classify_participant_visit(diagnosed_after_imaging, VISIT)
    assert (result["candidate_status"], result["candidate_label"]) == (
        "incident_case", "glaucoma"
    )


def test_selected_field_extraction_and_candidate_output_are_atomic(tmp_path):
    source = tmp_path / "ukb.csv"
    pd.DataFrame([
        _base_row(),
        _base_row() | {"eid": "1002", "6148-0.0": "2"},
    ]).assign(**{
        "20002-0.0": "", "41270-0.0": "", "41280-0.0": "",
    }).to_csv(source, index=False)
    matched = tmp_path / "matched.csv"
    report = extract_selected_table(source, matched, {"1001"}, chunksize=1)
    assert report["matched_participants"] == 1
    assert matched.is_file() and not matched.with_suffix(".csv.partial").exists()

    visits = tmp_path / "visits.csv"
    pd.DataFrame([VISIT]).to_csv(visits, index=False)
    candidates = tmp_path / "candidates.csv"
    result = build_candidate_table(matched, visits, candidates, chunksize=1)
    assert result["processed_participants"] == 1
    row = pd.read_csv(candidates).iloc[0]
    assert row["candidate_status"] == "strict_control"
    assert row["reference_standard_type"] == "record_derived_clinical_phenotype"
