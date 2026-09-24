"""Fail-closed handoff gate after the formal-V5 first prefix decision.

The gate never creates a scientific feed or submits work.  It compares the
three artifacts required by the locked migration protocol and exposes an
already-existing dual-pin feed only when every pinned identity matches.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from look.runtime.provenance import write_json_atomic
from look.runtime.state import file_sha256, stable_hash
from look.studies.v5_project_full_replay import FRAMEWORK, MODELS

CONTRACT_SCHEMA = "look_v5_full_cohort_first_decision_comparison_contract_v1"
FEED_SCHEMA = "look_formal_v5_dual_pin_feed_v1"
RECEIPT_SCHEMA = "look_v5_full_cohort_first_decision_comparison_receipt_v1"
REQUIRED = ("train_moments", "development_predictions", "next_decision")


def contract_identity(contract: dict) -> str:
    """Identity of scientific comparisons, excluding the optional feed binding."""
    return stable_hash({key: value for key, value in contract.items()
                        if key != "eligible_dual_pin_feed"})


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _regular_file(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Pinned regular file required: {path}")
    return path.resolve()


def _under(root: Path, relative: str) -> Path:
    raw = root / relative
    candidate = raw.resolve()
    if not candidate.is_relative_to(root) or raw.is_symlink():
        raise ValueError("Current artifact escapes the accepted first-prefix root")
    return _regular_file(candidate)


def _canonical_json_sha(path: Path) -> str:
    return stable_hash(json.loads(path.read_text(encoding="utf-8")))


def execute(*, first_prefix: Path, contract_path: Path, output: Path) -> dict:
    first_prefix = first_prefix.resolve()
    contract_path = _regular_file(contract_path)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    accepted_path = _regular_file(first_prefix / "accepted.json")
    identity_path = _regular_file(first_prefix / "identity.json")
    accepted, identity, contract = _read(accepted_path), _read(identity_path), _read(contract_path)
    if (accepted.get("schema") != "look_formal_v5_first_prefix_receipt_v1"
            or accepted.get("state") != "accepted"
            or accepted.get("test_access") is not False):
        raise ValueError("Accepted formal-V5 first-prefix receipt required")
    if stable_hash(identity) != accepted.get("identity_sha256"):
        raise ValueError("First-prefix identity/receipt mismatch")
    if (identity.get("framework_commit") != FRAMEWORK
            or identity.get("models_commit") != MODELS
            or identity.get("test_access") is not False):
        raise ValueError("First-prefix dual-pin identity mismatch")
    if (contract.get("schema") != CONTRACT_SCHEMA
            or contract.get("framework_commit") != FRAMEWORK
            or contract.get("mhd_models_commit") != MODELS
            or contract.get("test_access") is not False):
        raise ValueError("Pinned comparison contract mismatch")
    if contract.get("first_prefix_acceptance_sha256") != file_sha256(accepted_path):
        raise ValueError("Comparison contract does not bind this first-prefix receipt")

    rows = contract.get("artifacts")
    if not isinstance(rows, list) or sorted(row.get("name") for row in rows) != sorted(REQUIRED):
        raise ValueError("Exactly the three registered comparison artifacts are required")
    comparisons = []
    for row in rows:
        if set(row) != {"name", "current_relative_path", "reference_path", "reference_sha256", "comparator"}:
            raise ValueError("Comparison artifact contract has unexpected fields")
        if row["comparator"] != "canonical_json":
            raise ValueError("Only the reviewed canonical-JSON comparison is accepted")
        current = _under(first_prefix, row["current_relative_path"])
        reference = _regular_file(Path(row["reference_path"]))
        reference_sha = file_sha256(reference)
        if reference_sha != row["reference_sha256"]:
            raise ValueError(f"Pinned V4 reference changed: {row['name']}")
        current_value_sha = _canonical_json_sha(current)
        reference_value_sha = _canonical_json_sha(reference)
        comparisons.append({
            "name": row["name"],
            "current_path": str(current),
            "current_sha256": file_sha256(current),
            "reference_path": str(reference),
            "reference_sha256": reference_sha,
            "canonical_value_sha256": current_value_sha,
            "equal": current_value_sha == reference_value_sha,
        })

    equal = all(row["equal"] for row in comparisons)
    feed_info = contract.get("eligible_dual_pin_feed")
    state, handoff = "blocked_comparison_mismatch", None
    if equal and feed_info is None:
        state = "blocked_missing_eligible_dual_pin_feed"
    elif equal:
        if set(feed_info) != {"path", "sha256"}:
            raise ValueError("Dual-pin feed binding is malformed")
        feed_path = _regular_file(Path(feed_info["path"]))
        if file_sha256(feed_path) != feed_info["sha256"]:
            raise ValueError("Dual-pin feed bytes changed")
        feed = _read(feed_path)
        if (feed.get("schema") != FEED_SCHEMA
                or feed.get("framework_commit") != FRAMEWORK
                or feed.get("mhd_models_commit") != MODELS
                or feed.get("comparison_contract_identity_sha256") != contract_identity(contract)
                or feed.get("test_access") is not False):
            raise ValueError("Feed is not eligible for this comparison contract")
        state = "eligible_handoff"
        handoff = {"path": str(feed_path), "sha256": file_sha256(feed_path)}

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "state": state,
        "framework_commit": FRAMEWORK,
        "mhd_models_commit": MODELS,
        "first_prefix_acceptance_sha256": file_sha256(accepted_path),
        "comparison_contract_path": str(contract_path),
        "comparison_contract_sha256": file_sha256(contract_path),
        "comparisons": comparisons,
        "eligible_dual_pin_feed": handoff,
        "dispatcher_submission_performed": False,
        "gpu_requested": False,
        "test_access": False,
    }
    write_json_atomic(receipt, output / "comparison_receipt.json")
    if handoff is not None:
        write_json_atomic({
            "schema": "look_v5_existing_dispatcher_handoff_v1",
            "comparison_receipt_sha256": file_sha256(output / "comparison_receipt.json"),
            "feed": handoff,
            "test_access": False,
        }, output / "dispatcher_handoff.json")
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-prefix", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    receipt = execute(first_prefix=args.first_prefix, contract_path=args.contract, output=args.output)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["state"] == "eligible_handoff" else 75


if __name__ == "__main__":
    raise SystemExit(main())
