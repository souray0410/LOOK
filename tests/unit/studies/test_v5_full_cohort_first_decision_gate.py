import json
from pathlib import Path

import pytest

from look.runtime.state import file_sha256, stable_hash
from look.studies.v5_full_cohort_first_decision_gate import contract_identity, execute
from look.studies.v5_project_full_replay import FRAMEWORK, MODELS
from look.studies.v5_project_sharded_first_decision import main as first_decision_main


def put(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


def fixture(tmp_path, *, feed=False, mismatch=False):
    first = tmp_path / "first"
    identity = {"framework_commit": FRAMEWORK, "models_commit": MODELS, "test_access": False}
    put(first / "identity.json", identity)
    accepted = {"schema": "look_formal_v5_first_prefix_receipt_v1", "state": "accepted",
                "identity_sha256": stable_hash(identity), "test_access": False}
    put(first / "accepted.json", accepted)
    artifacts = []
    for index, name in enumerate(("train_moments", "development_predictions", "next_decision")):
        current = put(first / f"{name}.json", {"value": index + int(mismatch and index == 2)})
        reference = put(tmp_path / "reference" / f"{name}.json", {"value": index})
        artifacts.append({"name": name, "current_relative_path": current.name,
                          "reference_path": str(reference), "reference_sha256": file_sha256(reference),
                          "comparator": "canonical_json"})
    contract = {"schema": "look_v5_full_cohort_first_decision_comparison_contract_v1",
                "framework_commit": FRAMEWORK, "mhd_models_commit": MODELS,
                "first_prefix_acceptance_sha256": file_sha256(first / "accepted.json"),
                "artifacts": artifacts, "test_access": False}
    contract_path = put(tmp_path / "contract.json", contract)
    if feed:
        feed_path = tmp_path / "feed.json"
        put(feed_path, {"schema": "look_formal_v5_dual_pin_feed_v1", "framework_commit": FRAMEWORK,
                        "mhd_models_commit": MODELS,
                        "comparison_contract_identity_sha256": contract_identity(contract),
                        "test_access": False})
        contract["eligible_dual_pin_feed"] = {"path": str(feed_path), "sha256": file_sha256(feed_path)}
        put(contract_path, contract)
    return first, contract_path


def test_equal_assets_without_feed_stop_fail_closed(tmp_path):
    first, contract = fixture(tmp_path)
    result = execute(first_prefix=first, contract_path=contract, output=tmp_path / "out")
    assert result["state"] == "blocked_missing_eligible_dual_pin_feed"
    assert result["dispatcher_submission_performed"] is False
    assert all(row["equal"] for row in result["comparisons"])


def test_mismatch_never_hands_off(tmp_path):
    first, contract = fixture(tmp_path, mismatch=True)
    result = execute(first_prefix=first, contract_path=contract, output=tmp_path / "out")
    assert result["state"] == "blocked_comparison_mismatch"
    assert result["eligible_dual_pin_feed"] is None


def test_changed_reference_is_rejected(tmp_path):
    first, contract = fixture(tmp_path)
    value = json.loads(contract.read_text())
    Path(value["artifacts"][0]["reference_path"]).write_text("{}\n")
    with pytest.raises(ValueError, match="reference changed"):
        execute(first_prefix=first, contract_path=contract, output=tmp_path / "out")


def test_existing_eligible_dual_pin_feed_emits_handoff_only(tmp_path):
    first, contract = fixture(tmp_path, feed=True)
    result = execute(first_prefix=first, contract_path=contract, output=tmp_path / "out")
    assert result["state"] == "eligible_handoff"
    assert result["dispatcher_submission_performed"] is False
    assert (tmp_path / "out" / "dispatcher_handoff.json").is_file()


def test_first_decision_requires_complete_comparison_wiring():
    args = ["--source-run", "source", "--checkpoint", "checkpoint", "--cache", "cache",
            "--pca", "pca", "--output", "output", "--gpu-budget-bytes", "1",
            "--comparison-contract", "contract"]
    with pytest.raises(ValueError, match="configured together"):
        first_decision_main(args)
