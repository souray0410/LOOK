import numpy as np
import pandas as pd

from look_core.quality_audit import evidence_stratified_validation


def test_evidence_stratification_preserves_prediction_alignment(tmp_path):
    labels_csv = tmp_path / "labels.csv"
    pd.DataFrame([
        {
            "participant_id": "normal",
            "split": "validation",
            "label_id": 0,
            "label_name": "normal",
            "evidence_sources": "",
        },
        {
            "participant_id": "self-report",
            "split": "validation",
            "label_id": 1,
            "label_name": "diabetes_related_eye_disease",
            "evidence_sources": "6148",
        },
        {
            "participant_id": "record",
            "split": "validation",
            "label_id": 2,
            "label_name": "glaucoma",
            "evidence_sources": "20002;6148",
        },
    ]).to_csv(labels_csv, index=False)
    probabilities = np.asarray([
        [0.8, 0.1, 0.1, 0.0],
        [0.2, 0.6, 0.1, 0.1],
        [0.1, 0.1, 0.7, 0.1],
    ])
    report = evidence_stratified_validation(
        labels_csv,
        np.asarray([0, 1, 2]),
        probabilities,
        np.asarray(["normal", "self-report", "record"]),
    )
    assert report["label_mismatches"] == 0
    assert {row["evidence_group"] for row in report["groups"]} == {
        "strict_normal_control",
        "assessment_self_report_only",
        "corroborated_or_non_screen_record",
    }
