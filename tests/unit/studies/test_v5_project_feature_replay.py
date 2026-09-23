import json
from pathlib import Path

from look.runtime.state import file_sha256
from look.studies import v5_project_feature_replay as replay


class Dataset:
    def __init__(self):
        self.participant_ids = ["a", "b"]
        self.counts = [2, 1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def parent(tmp_path, name, track):
    root = tmp_path / name
    spec = {"test_used": False, "track": track, "training": {},
            "train_manifest": "train", "train_manifest_sha256": "same",
            "development_manifest": "dev", "development_manifest_sha256": "same"}
    write(root / "spec.json", spec)
    write(root / "source_accepted.json", {})
    manifest = {"schema": "look_selected_native_v1", "test_access": False,
                "complete_training_resume": False,
                "files": {"spec.json": file_sha256(root / "spec.json"),
                          "source_accepted.json": file_sha256(root / "source_accepted.json")}}
    write(root / "selected_artifact.json", manifest)
    return root, file_sha256(root / "selected_artifact.json")


def test_historical_loader_is_explicit_and_records_ordered_identity(tmp_path, monkeypatch):
    first, first_sha = parent(tmp_path, "first", "cfp_2d")
    second, second_sha = parent(tmp_path, "second", "oct_bscan_2d")
    run = tmp_path / "run"
    write(run / "spec.json", {"schema": "look_project_case_v1", "test_access": False,
        "seed": 3416, "parents": {"first": {"path": str(first), "manifest_sha256": first_sha},
                                    "second": {"path": str(second), "manifest_sha256": second_sha}}})
    monkeypatch.setattr(replay, "from_parent_specs", lambda *a, **k: Dataset())
    datasets, identity = replay.datasets(run, object(), seed=3416)
    assert set(datasets) == {"train", "development"}
    assert identity["roles"]["train"]["participants"] == 2
    assert identity["roles"]["train"]["eyes"] == 3
    assert identity["test_access"] is False
