"""Prepare an isolated LOOK-only role-policy transition.

This module never installs a role policy and never submits or cancels Slurm
jobs.  It creates content-addressable candidate inputs for the account control
plane.  Models must build its later proposal from the resulting LOOK policy.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_exclusive_json(path: str | Path, value: dict[str, Any]) -> None:
    """Create one immutable JSON file without following a final symlink."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(target, flags, 0o444)
    try:
        payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def build_dedicated_journal(
    *, job: dict[str, Any], claim_path: str | Path, expected_claim_sha256: str
) -> dict[str, Any]:
    required = {"job_id", "name", "command", "submit_time", "state", "user", "account",
                "req_tres", "replaces_job_id", "cancelled_retry_job_id",
                "replacement_receipt", "replacement_receipt_sha256"}
    if set(job) != required:
        raise ValueError(f"Job identity fields must be exactly {sorted(required)}")
    if not str(job["job_id"]).isdigit() or job["state"] not in {"PENDING", "RUNNING"}:
        raise ValueError("The dedicated journal requires one live Slurm job")
    claim = Path(claim_path).resolve()
    if sha256(claim) != expected_claim_sha256:
        raise ValueError("Claim identity changed")
    replacement = Path(job["replacement_receipt"]).resolve()
    if sha256(replacement) != job["replacement_receipt_sha256"]:
        raise ValueError("Replacement receipt identity changed")
    if not all(str(job[k]).isdigit() for k in ("replaces_job_id", "cancelled_retry_job_id")):
        raise ValueError("Replacement lineage is incomplete")
    return {
        "schema": "look_workflow_requests_v1",
        "requests": [
            {
                "job_id": str(job["job_id"]),
                "name": job["name"],
                "command": job["command"],
                "time": job["submit_time"],
                "state": "submitted",
                "v5_claim": str(claim),
                "v5_claim_sha256": expected_claim_sha256,
                "slurm_user": job["user"], "slurm_account": job["account"],
                "slurm_req_tres": job["req_tres"],
                "replaces_job_id": str(job["replaces_job_id"]),
                "cancelled_retry_job_id": str(job["cancelled_retry_job_id"]),
                "replacement_receipt": str(replacement),
                "replacement_receipt_sha256": job["replacement_receipt_sha256"],
            }
        ],
    }


def build_look_only_policy(
    *, current: dict[str, Any], dedicated_journal: str | Path
) -> dict[str, Any]:
    if current.get("schema") != "research_gpu_roles_v1":
        raise ValueError("Unknown role-policy schema")
    projects = current.get("projects")
    if not isinstance(projects, dict) or "LOOK" not in projects:
        raise ValueError("LOOK role is missing")
    result = copy.deepcopy(current)
    journals = result["projects"]["LOOK"].get("request_journals")
    if not isinstance(journals, list) or not all(isinstance(x, str) for x in journals):
        raise ValueError("Malformed LOOK request journals")
    path = str(Path(dedicated_journal).resolve())
    if path in journals:
        raise ValueError("Dedicated journal is already registered")
    journals.append(path)
    return result


def verify_look_only_delta(
    current: dict[str, Any], candidate: dict[str, Any], journal_path: str | Path
) -> None:
    expected = build_look_only_policy(current=current, dedicated_journal=journal_path)
    if candidate != expected:
        raise ValueError("Candidate changes more than the LOOK journal registration")


def build_pending_monitor_replacement(
    *, finalizer: dict[str, Any], current_policy_sha256: str, look_policy_sha256: str
) -> dict[str, Any]:
    """Describe the required immutable replacement for a still-pending monitor.

    The existing job must be cancelled and replaced by the control plane; its
    script must never be edited in place.  A later Models transition requires a
    newly frozen replacement that adds the exact Models policy SHA.
    """
    required = {"job_id", "state", "dependency", "command", "source_sha256"}
    if set(finalizer) != required or finalizer["state"] != "PENDING":
        raise ValueError("The old finalizer must have one exact PENDING identity")
    if (str(finalizer["job_id"]) != "52430491" or
            finalizer["dependency"] != "afterany:52429877(unfulfilled)"):
        raise ValueError("Malformed finalizer identity")
    if current_policy_sha256 == look_policy_sha256:
        raise ValueError("LOOK transition must change the policy identity")
    return {
        "schema": "look_v5_pending_monitor_replacement_v1",
        "old_finalizer": finalizer,
        "replacement_mode": "cancel_pending_then_submit_one_afterany_under_account_lock",
        "allowed_policy_sha256": [current_policy_sha256, look_policy_sha256],
        "models_policy_sha256": None,
        "models_transition_gate": "replace_pending_monitor_again_with_exact_models_sha_before_models_policy_install",
        "hot_edit_allowed": False,
        "old_source_sha256": finalizer["source_sha256"],
        "transaction": ["submit_replacement_held", "verify_exact_held_identity",
                        "cancel_old_pending", "verify_old_terminal", "release_replacement"],
        "stage2_running_rule": "wait_for_stage1_monitor_terminal_before_policy_install",
        "test_access": False,
    }


def verify_monitor_policy(policy_path: str | Path, binding: dict[str, Any]) -> str:
    digest = sha256(policy_path)
    allowed = binding.get("allowed_policy_sha256")
    if binding.get("schema") != "look_v5_pending_monitor_replacement_v1" or not isinstance(allowed, list):
        raise ValueError("Unknown monitor transition binding")
    if digest not in allowed:
        raise ValueError("Role policy is not authorized for this immutable monitor")
    return digest


def build_v21_chain_manager(*, v20_source: str | Path, expected_sha256: str,
                            dedicated_journal: str | Path, allowed_policy_sha256: list[str]) -> str:
    """Derive v21 while preserving v20 science and redirecting all successors."""
    source = Path(v20_source)
    if sha256(source) != expected_sha256:
        raise ValueError("v20 chain manager identity changed")
    text = source.read_text()
    old_journal = "JOURNAL=ROOT/'look_forward_20260916/lease_recovery_20260917/dispatcher/requests.json'"
    old_allow = "{'afde7998b16d39332818f2b1f6ad87401b7ff42f653666f85a922065cc1b3b72','efc451893fbeedf4fb2e22aa07a090910e3530e8e8595e33fb3aebc5e791ec02'}"
    if text.count(old_journal) != 1 or text.count(old_allow) != 1:
        raise ValueError("v20 replacement anchors changed")
    absolute = str(Path(dedicated_journal).resolve())
    allowed = "{" + ",".join(repr(x) for x in allowed_policy_sha256) + "}"
    result = text.replace(old_journal, f"JOURNAL=pathlib.Path({absolute!r})").replace(old_allow, allowed)
    if "lease_recovery_20260917/dispatcher/requests.json" in result:
        raise ValueError("Legacy journal remains in v21")
    return result


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    current_path = Path(args.current_policy).resolve()
    if sha256(current_path) != args.expected_current_sha256:
        raise ValueError("Current policy SHA changed")
    job = read_json(args.job_identity)
    journal_path = Path(args.output) / "look_v5_requests.json"
    journal = build_dedicated_journal(
        job=job,
        claim_path=args.claim,
        expected_claim_sha256=args.expected_claim_sha256,
    )
    write_exclusive_json(journal_path, journal)
    current = read_json(current_path)
    candidate = build_look_only_policy(current=current, dedicated_journal=journal_path)
    policy_path = Path(args.output) / "role_policy.look_only.proposed.json"
    write_exclusive_json(policy_path, candidate)
    verify_look_only_delta(current, candidate, journal_path)
    receipt = {
        "schema": "look_v5_role_policy_transition_v1",
        "state": "candidate_not_installed",
        "current_policy": str(current_path),
        "current_policy_sha256": args.expected_current_sha256,
        "dedicated_journal": str(journal_path.resolve()),
        "dedicated_journal_sha256": sha256(journal_path),
        "look_policy": str(policy_path.resolve()),
        "look_policy_sha256": sha256(policy_path),
        "job_id": str(job["job_id"]),
        "models_next_previous_policy_sha256": sha256(policy_path),
        "test_access": False,
    }
    if args.finalizer_identity:
        monitor = build_pending_monitor_replacement(
            finalizer=read_json(args.finalizer_identity),
            current_policy_sha256=args.expected_current_sha256,
            look_policy_sha256=receipt["look_policy_sha256"],
        )
        monitor_path = Path(args.output) / "pending_monitor_replacement.json"
        write_exclusive_json(monitor_path, monitor)
        receipt["pending_monitor_replacement"] = str(monitor_path.resolve())
        receipt["pending_monitor_replacement_sha256"] = sha256(monitor_path)
    write_exclusive_json(Path(args.output) / "transition_receipt.json", receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--current-policy", required=True)
    result.add_argument("--expected-current-sha256", required=True)
    result.add_argument("--job-identity", required=True)
    result.add_argument("--claim", required=True)
    result.add_argument("--expected-claim-sha256", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--finalizer-identity")
    return result


def main() -> None:
    print(json.dumps(prepare(parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
