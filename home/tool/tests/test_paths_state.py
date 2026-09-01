import json
import os
from pathlib import Path

import pytest

from look_core.paths import ProjectPaths
from look_core.state import PipelineState, atomic_write_json, atomic_write_text, quarantine, stable_hash


def test_project_paths_allow_environment_overrides(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    (root / "project.json").write_text(
        json.dumps({
            "deployment_defaults": {"data_root": str(tmp_path / "data")},
            "directories": {"tool": "tool", "pipeline": "pipeline", "dataset": "dataset", "cache": "cache", "runs": "runs"},
        }),
        encoding="utf-8",
    )
    override = tmp_path / "override"
    monkeypatch.setenv("LOOK_RUNS_ROOT", str(override))
    paths = ProjectPaths.load(root)
    assert paths.runs_root == override
    assert paths.dataset_root == tmp_path / "data/dataset"


def test_atomic_json_never_leaves_partial_file(tmp_path):
    destination = tmp_path / "result.json"
    atomic_write_json({"status": "complete"}, destination)
    assert json.loads(destination.read_text()) == {"status": "complete"}
    assert not list(tmp_path.glob("*.partial"))
    atomic_write_text("verified\n", tmp_path / "report.txt")
    assert (tmp_path / "report.txt").read_text() == "verified\n"


def test_pipeline_state_recovers_dead_local_lock(tmp_path):
    state_root = tmp_path / "state"
    lock_dir = state_root / "stage"
    lock_dir.mkdir(parents=True)
    (lock_dir / "stage.lock").write_text(
        json.dumps({"pid": 999_999_999, "host": os.uname().nodename}), encoding="utf-8"
    )
    output = tmp_path / "output.json"
    atomic_write_json({"ok": True}, output)
    with PipelineState(state_root, "stage", {"value": 1}) as state:
        state.checkpoint(item=3)
        state.complete([output])
    payload = json.loads((lock_dir / "state.json").read_text())
    assert payload["status"] == "completed"
    assert payload["progress"] == {"item": 3}
    assert not (lock_dir / "stage.lock").exists()
    state = PipelineState(state_root, "stage", {"value": 1})
    assert state.completed_outputs_valid([output])
    output.write_text("truncated", encoding="utf-8")
    assert not state.completed_outputs_valid([output])


def test_active_lock_is_not_overwritten(tmp_path):
    with PipelineState(tmp_path, "stage", {}) as _state:
        with pytest.raises(RuntimeError, match="locked"):
            with PipelineState(tmp_path, "stage", {}):
                pass


def test_quarantine_preserves_bad_artifact(tmp_path):
    artifact = tmp_path / "broken.bin"
    artifact.write_bytes(b"truncated")
    destination = quarantine(artifact, tmp_path / "quarantine", "hash mismatch")
    assert destination.read_bytes() == b"truncated"
    assert not artifact.exists()
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})
