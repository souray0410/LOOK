import json
from pathlib import Path

import pytest

from look.runtime import feature_replay_dispatch as replay
from look.runtime import project_dispatch
from look.runtime.state import file_sha256, stable_hash


SITES = [f"site_{index}" for index in range(9)]


def write(path, value):
    Path(path).write_text(json.dumps(value))


def fixture(tmp_path):
    script = tmp_path / "run.sh"; script.write_text("#!/bin/sh\nexit 0\n")
    output = tmp_path / "output"; output.mkdir()
    package = tmp_path / "package.json"
    write(package, {"package_id": "replay", "dispatch_authorized": False,
                    "preservation": {"test_access": False},
                    "execution": {"output": str(output)}})
    policy = tmp_path / "policy.json"
    write(policy, {"schema": "look_weekly_delivery_v1", "package_id": "week",
                   "first_seed": 3416, "test_access": False,
                   "required_cases": [{"run_dir": "missing", "spec": "missing", "spec_sha256": "x"}],
                   "release_receipt": str(tmp_path / "missing.json"),
                   "unresolved_requirements": ["still-running"]})
    spec = {"schema": "look_feature_replay_dispatch_v1", "package_id": "replay",
            "seed": 3417, "test_access": False, "production_cutover": False,
            "phases": ["reference", "check"], "run_phase_script": str(script),
            "run_phase_script_sha256": file_sha256(script), "package_manifest": str(package),
            "package_manifest_sha256": file_sha256(package), "weekly_delivery_policy": str(policy),
            "weekly_delivery_policy_sha256": file_sha256(policy),
            "expected_counts": {"train": 1264, "development": 296}, "expected_sites": SITES}
    spec_path = tmp_path / "spec.json"; write(spec_path, spec)
    return spec_path, spec, output


def test_feed_is_held_by_existing_whole_weekly_gate(tmp_path):
    spec_path, spec, output = fixture(tmp_path)
    task = {"id": "replay", "spec": str(spec_path), "spec_sha256": file_sha256(spec_path),
            "run_dir": str(output)}
    feed = tmp_path / "feed.json"
    write(feed, {"schema": "look_feature_replay_feed_v1", "test_access": False, "tasks": [task]})
    config = {"project_feed": str(tmp_path / "none"), "native_feed": str(tmp_path / "none2"),
              "feature_replay_feeds": [str(feed)], "weekly_delivery_policy": spec["weekly_delivery_policy"],
              "output": str(tmp_path / "admission")}
    assert project_dispatch.work(config) == [dict(task, execution="look_feature_replay")]
    assert project_dispatch.admissible_work(config, object()) == []
    admission = json.loads((tmp_path / "admission/api_admission.json").read_text())
    assert admission["rejected"] == [{"run": str(output), "reason": "waiting_whole_weekly_delivery", "seed": 3417}]


def test_execution_fails_before_launch_while_weekly_gate_closed(tmp_path, monkeypatch):
    spec_path, _, output = fixture(tmp_path)
    monkeypatch.setenv("LOOK_ROTATION_CLAIM", str(tmp_path / "claim.json"))
    monkeypatch.setattr(replay.subprocess, "run", lambda *a, **k: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="whole-weekly"):
        replay.execute(spec_path, output)


def test_verifier_requires_bitwise_equal_and_pinned_receipts(tmp_path):
    _, spec, output = fixture(tmp_path)
    common = {"schema": "look_node_feature_replay_v1", "test_access": False,
              "source_checkpoint_sha256": "source", "original_bank_sha256": "bank",
              "sites": SITES, "counts": spec["expected_counts"], "batches": {"train": 79, "development": 19}}
    write(output / "reference_receipt.json", dict(common, state="reference_exported"))
    write(output / "check_receipt.json", dict(common, state="accepted", production_cutover=False,
                                                all_site_values_bitwise_equal=True))
    accepted = {"schema": "look_feature_replay_dispatch_acceptance_v1", "state": "accepted",
                "identity": stable_hash(spec), "test_access": False,
                "reference_receipt_sha256": file_sha256(output / "reference_receipt.json"),
                "check_receipt_sha256": file_sha256(output / "check_receipt.json")}
    write(output / "accepted.json", accepted)
    assert replay.verify_case(output, spec)["state"] == "accepted"
    check = json.loads((output / "check_receipt.json").read_text())
    check["all_site_values_bitwise_equal"] = False; write(output / "check_receipt.json", check)
    with pytest.raises(ValueError, match="V5 receipt rejected"):
        replay.verify_case(output, spec)


def test_feature_replay_never_enters_priority_preemption(tmp_path, monkeypatch):
    task = {"execution": "look_feature_replay"}
    monkeypatch.setattr(project_dispatch, "admissible_work", lambda *a: [task])
    assert project_dispatch.priority_work({}, object()) == []
