from pathlib import Path

import pandas as pd

from look_core.cohort import FOUR_CLASS_MAPPING, derive_four_class_cohorts, verify_four_class_cohorts


def _source(tmp_path: Path) -> tuple[Path, Path]:
    image_root = tmp_path / "images"
    image_root.mkdir()
    rows = []
    participant = 1000
    counts = {
        "normal": 60,
        "diabetes_related_eye_disease": 20,
        "glaucoma": 30,
        "macular_degeneration": 25,
    }
    for label_name, count in counts.items():
        for offset in range(count):
            paths = {}
            for eye in ("left", "right"):
                for modality in ("fundus", "oct"):
                    name = f"{participant}_{eye}_{modality}.png"
                    (image_root / name).write_bytes(b"fixture")
                    paths[f"{eye}_{modality}_path"] = name
            rows.append({
                "participant_id": str(participant), "instance": 0,
                "assessment_date": "2010-01-01", "assessment_age": 45 + offset % 30,
                "sex": str(offset % 2), "candidate_label": label_name,
                "candidate_status": "strict_control" if label_name == "normal" else "prevalent_case",
                "reference_standard_type": "record_derived_clinical_phenotype",
                **paths,
            })
            participant += 1
    for offset in range(8):
        paths = {}
        for eye in ("left", "right"):
            for modality in ("fundus", "oct"):
                name = f"{participant}_{eye}_{modality}.png"
                (image_root / name).write_bytes(b"fixture")
                paths[f"{eye}_{modality}_path"] = name
        rows.append({
            "participant_id": str(participant), "instance": 0,
            "assessment_date": "2010-01-01", "assessment_age": 60,
            "sex": str(offset % 2), "candidate_label": "glaucoma",
            "candidate_status": "incident_case",
            "reference_standard_type": "record_derived_clinical_phenotype",
            **paths,
        })
        participant += 1
    source = tmp_path / "record_phenotype_candidates.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    return source, image_root


def test_record_derived_cohorts_are_deterministic_and_non_destructive(tmp_path):
    source, image_root = _source(tmp_path)
    before = source.read_bytes()
    cohort_root = tmp_path / "cohorts"
    first = derive_four_class_cohorts(source, cohort_root)
    second = derive_four_class_cohorts(source, cohort_root)
    assert first == second
    assert source.read_bytes() == before
    balanced = pd.read_csv(cohort_root / "balanced/reference_labels.csv", dtype={"participant_id": str})
    natural = pd.read_csv(cohort_root / "natural/reference_labels.csv", dtype={"participant_id": str})
    incident = pd.read_csv(cohort_root / "incident/reference_labels.csv", dtype={"participant_id": str})
    assert set(balanced.label_name) == set(FOUR_CLASS_MAPPING)
    assert balanced.participant_id.is_unique
    assert natural.participant_id.is_unique
    assert set(balanced.participant_id) <= set(natural.participant_id)
    assert set(incident.participant_id).isdisjoint(natural.participant_id)
    for split in ("train", "validation", "test"):
        counts = balanced.loc[balanced.split == split, "label_name"].value_counts()
        assert counts["normal"] == counts.max()
        assert counts.max() / counts.min() <= 2.5
    report = verify_four_class_cohorts(cohort_root, image_root, check_images=True)
    assert report["status"] == "PASS"
    assert report["unit_of_analysis"] == "participant_bilateral_visit"
    assert report["analysis_readiness"]["status"] == "NOT_READY"
    assert (cohort_root / "analysis_readiness.json").is_file()
