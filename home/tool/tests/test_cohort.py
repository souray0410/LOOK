import pandas as pd
from PIL import Image

from look_core.cohort import (
    BINARY_CLASS_MAPPING,
    derive_glaucoma_cohorts,
    is_high_confidence_glaucoma,
    verify_glaucoma_cohorts,
)
from look_core.task_scout import build_task_cohort_bank, verify_task_cohort_bank


def _row(participant, status, label, evidence, offset):
    return {
        "participant_id": str(participant),
        "instance": 0,
        "left_fundus_path": "image.png",
        "left_oct_path": "image.png",
        "right_fundus_path": "image.png",
        "right_oct_path": "image.png",
        "assessment_date": "2010-01-01",
        "assessment_age": 55 + (offset % 2) * 5,
        "sex": str(offset % 2),
        "assessment_centre": str(11020 + offset % 2),
        "candidate_status": status,
        "candidate_label": label,
        "prevalent_targets": label if status == "prevalent_case" else "",
        "incident_targets": label if status == "incident_case" else "",
        "undated_targets": "",
        "evidence_sources": evidence,
        "evidence_detail": "{}",
        "incident_evidence_detail": "{}",
        "undated_evidence_detail": "{}",
        "competing_eye_condition": False,
        "reference_standard_type": "record_derived_clinical_phenotype",
        "phenotype_profile": "candidate",
    }


def test_high_confidence_rule_uses_objective_or_concordant_sources():
    assert is_high_confidence_glaucoma("41270")
    assert is_high_confidence_glaucoma("glaucoma_treatment")
    assert is_high_confidence_glaucoma("20002;6148")
    assert not is_high_confidence_glaucoma("6148")
    assert not is_high_confidence_glaucoma("20002")


def test_glaucoma_cohort_is_deterministic_matched_and_verified(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (8, 8)).save(image_root / "image.png")
    rows = []
    for index in range(20):
        evidence = "20002;6148" if index % 2 else "41270"
        rows.append(_row(1000 + index, "prevalent_case", "glaucoma", evidence, index))
    for index in range(4):
        rows.append(_row(2000 + index, "prevalent_case", "glaucoma", "6148", index))
    for index in range(80):
        rows.append(_row(3000 + index, "strict_control", "normal", "", index))
    for index in range(6):
        rows.append(_row(4000 + index, "incident_case", "glaucoma", "", index))
    source = tmp_path / "candidates.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    cohort_root = tmp_path / "cohort"

    first = derive_glaucoma_cohorts(source, cohort_root)
    second = derive_glaucoma_cohorts(source, cohort_root)
    assert first == second
    assert first["flow"]["high_confidence_glaucoma"] == 20
    assert first["flow"]["single_source_glaucoma"] == 4

    primary = pd.read_csv(cohort_root / "primary" / "reference_labels.csv")
    assert set(primary.label_name) == set(BINARY_CLASS_MAPPING)
    assert len(primary) == 40
    assert primary.participant_id.nunique() == 40
    for split in ("train", "validation", "test"):
        counts = primary.loc[primary.split == split, "label_name"].value_counts()
        assert counts["normal"] == counts["glaucoma"]
    report = verify_glaucoma_cohorts(cohort_root, image_root, check_images=True)
    assert report["status"] == "PASS"
    assert report["unit_of_analysis"] == "participant_bilateral_visit"


def test_task_scout_uses_one_global_split_and_matched_controls(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (8, 8)).save(image_root / "image.png")
    rows = []
    for index in range(60):
        rows.append(
            _row(
                1000 + index,
                "prevalent_case",
                "glaucoma",
                "20002;6148",
                index,
            )
        )
    for index in range(180):
        rows.append(_row(3000 + index, "strict_control", "normal", "", index))
    source = tmp_path / "candidates.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    bank_root = tmp_path / "task_scout"
    manifest = build_task_cohort_bank(
        source,
        bank_root,
        profile_ids=["glaucoma_high_confidence"],
    )
    assert manifest["profiles"]["glaucoma_high_confidence"]["prevalent_cases"] == 60
    report = verify_task_cohort_bank(bank_root, image_root, check_images=True)
    assert report["status"] == "PASS"
    primary = pd.read_csv(
        bank_root / "glaucoma_high_confidence/primary/reference_labels.csv",
        dtype={"participant_id": str},
    )
    split_map = pd.read_csv(
        bank_root / "participant_split_manifest.csv", dtype={"participant_id": str}
    ).set_index("participant_id")["split"]
    assert primary["split"].eq(primary["participant_id"].map(split_map)).all()
