from pathlib import Path

import pandas as pd

from look_core.cohort import FOUR_CLASS_MAPPING, derive_four_class_cohorts, verify_four_class_cohorts


def _source(tmp_path: Path) -> tuple[Path, Path]:
    image_root = tmp_path / "images"
    image_root.mkdir()
    rows = []
    participant = 1000
    source_counts = {
        "normal": 8,
        "diabetes_related_eye_disease": 3,
        "glaucoma": 4,
        "macular_degeneration": 5,
        "cataract": 2,
    }
    for split in ("train", "validation", "test"):
        for label_name, count in source_counts.items():
            for _ in range(count):
                fundus = f"{participant}_cfp.png"
                oct_image = f"{participant}_oct.png"
                (image_root / fundus).write_bytes(b"cfp")
                (image_root / oct_image).write_bytes(b"oct")
                rows.append({
                    "participant_id": str(participant), "instance": 0, "array": 0,
                    "eye": "left", "fundus_path": fundus, "oct_path": oct_image,
                    "label_id": 0, "label_name": label_name, "split": split,
                    "reference_source": "weak_self_report",
                })
                participant += 1
    source = tmp_path / "reference_labels.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    return source, image_root


def test_balanced_and_natural_cohorts_are_deterministic_and_non_destructive(tmp_path):
    source, image_root = _source(tmp_path)
    source_before = source.read_bytes()
    cohort_root = tmp_path / "cohorts"
    first = derive_four_class_cohorts(source, cohort_root)
    second = derive_four_class_cohorts(source, cohort_root)
    assert first == second
    assert source.read_bytes() == source_before
    balanced = pd.read_csv(cohort_root / "balanced" / "reference_labels.csv")
    natural = pd.read_csv(cohort_root / "natural" / "reference_labels.csv")
    assert set(balanced.label_name) == set(FOUR_CLASS_MAPPING)
    assert set(natural.label_name) == set(FOUR_CLASS_MAPPING)
    assert "cataract" not in set(natural.label_name)
    for split in ("train", "validation", "test"):
        current = balanced[balanced.split == split].label_name.value_counts()
        assert current["normal"] == current["glaucoma"] == 4
        assert current.max() / current.min() <= 1.9
        natural_split = natural[natural.split == split].label_name.value_counts()
        assert natural_split["normal"] == 8
    report = verify_four_class_cohorts(cohort_root, image_root, check_images=True)
    assert report["status"] == "PASS"
    assert all(value["participant_split_leakage"] == 0 for value in report["cohorts"].values())
