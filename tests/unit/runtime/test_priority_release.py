import json
from pathlib import Path

import pytest

from look.runtime import priority_release as module
from look.runtime.state import file_sha256
from look.studies.family_search_protocol import ARMS


def _feed(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    tasks = []
    for arm in ARMS:
        spec = tmp_path / f"{arm}.json"
        spec.write_text(json.dumps({"arm": arm}))
        run = tmp_path / f"run-{arm}"
        run.mkdir()
        (run / "spec.json").write_text(spec.read_text())
        (run / "accepted.json").write_text(json.dumps({"state": "accepted"}))
        tasks.append({"execution": "look_family_search", "test_access": False,
                      "arm": arm, "spec": str(spec),
                      "spec_sha256": file_sha256(spec), "run_dir": str(run)})
    feed = tmp_path / "feed.json"
    feed.write_text(json.dumps({"schema": "look_family_search_feed_v1",
                                "test_access": False, "tasks": tasks}))
    return feed


def test_builds_deterministic_core_release(tmp_path, monkeypatch):
    feed = _feed(tmp_path)
    monkeypatch.setattr(module, "validate", lambda spec: spec)
    checked = []
    monkeypatch.setattr(module, "verify_case", lambda run, spec: checked.append(spec["arm"]))
    output = tmp_path / "release.json"
    first = module.build(feed, output)
    before = output.read_bytes()
    second = module.build(feed, output)
    assert first == second and output.read_bytes() == before
    assert checked == ["residual_rrr", "pca_free_mean"] * 2
    assert [row["arm"] for row in first["requirements"]] == ["residual_rrr", "pca_free_mean"]
    assert first["released_search_modes"] == ["best_forward", "greedy"]


def test_fails_closed_for_changed_spec_or_acceptance(tmp_path, monkeypatch):
    feed = _feed(tmp_path)
    monkeypatch.setattr(module, "validate", lambda spec: spec)
    monkeypatch.setattr(module, "verify_case", lambda run, spec: None)
    output = tmp_path / "release.json"
    receipt = module.build(feed, output)
    Path(receipt["requirements"][0]["spec"]).write_text('{"changed": true}')
    with pytest.raises(ValueError, match="specification changed"):
        module.build(feed, output)


def test_requires_all_four_registered_arms_and_refuses_overwrite(tmp_path, monkeypatch):
    feed = _feed(tmp_path)
    data = json.loads(feed.read_text())
    data["tasks"].pop()
    feed.write_text(json.dumps(data))
    monkeypatch.setattr(module, "validate", lambda spec: spec)
    with pytest.raises(ValueError, match="Exactly four"):
        module.build(feed, tmp_path / "release.json")

    feed = _feed(tmp_path / "second")
    monkeypatch.setattr(module, "verify_case", lambda run, spec: None)
    output = tmp_path / "existing.json"
    output.write_text('{}')
    with pytest.raises(ValueError, match="different identity"):
        module.build(feed, output)
