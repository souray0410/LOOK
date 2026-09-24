"""Crash-resumable held-job replacement for a pending LOOK finalizer.

The caller supplies a concrete scheduler adapter. This module keeps the shared
account lock continuously from the first live observation through replacement
release verification and records every irreversible boundary before the action
whose acknowledgement could be lost.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable


PHASES = (
    "prepared",
    "held_submitted",
    "held_verified",
    "cancel_requested",
    "old_cancelled",
    "artifacts_committed",
    "release_requested",
    "release_verified",
)


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def spec_sha256(spec: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(spec)).hexdigest()


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o444)
    try:
        payload = _canonical(value)
        if os.write(fd, payload) != len(payload):
            raise OSError("short receipt write")
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_dir(path.parent)


@contextmanager
def exclusive_existing_lock(path: str | Path):
    lock = Path(path)
    flags = os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(lock, flags)
    try:
        before = os.fstat(fd)
        current = os.stat(lock, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != (
            current.st_dev,
            current.st_ino,
        ):
            raise RuntimeError("Shared account lock identity changed")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = os.stat(lock, follow_symlinks=False)
        if (before.st_dev, before.st_ino) != (current.st_dev, current.st_ino):
            raise RuntimeError("Shared account lock replaced while waiting")
        yield fd
    finally:
        os.close(fd)


def _hex_digest(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"Malformed {field}")
    return value


def validate_spec(spec: dict[str, Any]) -> None:
    required = {
        "schema",
        "mode",
        "transaction_id",
        "old_job_id",
        "gpu_job_id",
        "dependency",
        "old_command",
        "gpu_command",
        "replacement_command",
        "replacement_source_sha256",
        "binding_sha256",
        "expected_user",
        "expected_account",
        "expected_monitor_req_tres",
        "expected_gpu_req_tres",
        "shared_lock",
        "max_account_gpus",
        "test_access",
    }
    if set(spec) != required or spec.get("schema") != "look_v5_monitor_handover_spec_v2":
        raise ValueError(f"Transaction spec fields must be exactly {sorted(required)}")
    if spec["mode"] not in {"production", "shadow"}:
        raise ValueError("Unknown handover mode")
    if spec["test_access"] is not False or spec["max_account_gpus"] != 24:
        raise ValueError("Safety envelope changed")
    if spec["mode"] == "production":
        if str(spec["old_job_id"]) != "52430491" or str(spec["gpu_job_id"]) != "52429877":
            raise ValueError("Production chain identity changed")
        if spec["dependency"] != "afterany:52429877(unfulfilled)":
            raise ValueError("Production dependency changed")
    elif spec["dependency"] != f"afterany:{spec['gpu_job_id']}(unfulfilled)":
        raise ValueError("Shadow dependency does not bind its predecessor")
    for field in ("replacement_source_sha256", "binding_sha256"):
        _hex_digest(spec[field], field)
    for field in ("old_command", "gpu_command", "replacement_command", "shared_lock"):
        value = spec[field]
        if not isinstance(value, str) or not value.startswith("/"):
            raise ValueError(f"{field} must be an absolute path")


def _read_receipts(root: Path, digest: str) -> list[dict[str, Any]]:
    found = []
    for index, phase in enumerate(PHASES):
        path = root / f"{index:02d}-{phase}.json"
        if not path.exists():
            if any(
                (root / f"{later:02d}-{PHASES[later]}.json").exists()
                for later in range(index + 1, len(PHASES))
            ):
                raise ValueError("Non-contiguous phase receipts")
            break
        value = json.loads(path.read_text())
        if value.get("phase") != phase or value.get("spec_sha256") != digest:
            raise ValueError("Phase receipt identity changed")
        found.append(value)
    return found


def _record(root: Path, phase: str, digest: str, **evidence: Any) -> None:
    write_exclusive(
        root / f"{PHASES.index(phase):02d}-{phase}.json",
        {
            "schema": "look_v5_monitor_handover_phase_v2",
            "phase": phase,
            "spec_sha256": digest,
            **evidence,
        },
    )


def _assert_old_pending(observed: dict[str, Any], spec: dict[str, Any]) -> None:
    expected = {
        "job_id": str(spec["old_job_id"]),
        "state": "PENDING",
        "dependency": spec["dependency"],
        "user": spec["expected_user"],
        "account": spec["expected_account"],
        "req_tres": spec["expected_monitor_req_tres"],
        "command": spec["old_command"],
    }
    if any(str(observed.get(key)) != str(value) for key, value in expected.items()):
        raise ValueError("Old monitor is not the exact replaceable job")
    if observed.get("has_step") is not False:
        raise ValueError("Old monitor has an active or ambiguous step")


def _assert_gpu_preserved(observed: dict[str, Any], spec: dict[str, Any]) -> None:
    expected = {
        "job_id": str(spec["gpu_job_id"]),
        "user": spec["expected_user"],
        "account": spec["expected_account"],
        "req_tres": spec["expected_gpu_req_tres"],
        "command": spec["gpu_command"],
    }
    if observed.get("state") not in {"PENDING", "RUNNING"} or any(
        str(observed.get(key)) != str(value) for key, value in expected.items()
    ):
        raise ValueError("GPU predecessor identity or live state changed")


def _assert_held(observed: dict[str, Any], spec: dict[str, Any], job_id: str) -> None:
    expected = {
        "job_id": job_id,
        "state": "PENDING",
        "reason": "JobHeldUser",
        "dependency": spec["dependency"],
        "user": spec["expected_user"],
        "account": spec["expected_account"],
        "req_tres": spec["expected_monitor_req_tres"],
        "command": spec["replacement_command"],
    }
    if any(str(observed.get(key)) != str(value) for key, value in expected.items()):
        raise ValueError("Replacement is not the exact held job")
    if observed.get("has_step") is not False:
        raise ValueError("Held replacement unexpectedly has a step")


def _assert_released(observed: dict[str, Any], spec: dict[str, Any], job_id: str) -> None:
    expected = {
        "job_id": job_id,
        "user": spec["expected_user"],
        "account": spec["expected_account"],
        "req_tres": spec["expected_monitor_req_tres"],
        "command": spec["replacement_command"],
    }
    state = observed.get("state")
    if state == "PENDING":
        if observed.get("reason") == "JobHeldUser" or observed.get("dependency") not in {
            spec["dependency"],
            "(null)",
        }:
            raise ValueError("Replacement release was not observed")
    elif state == "RUNNING":
        if observed.get("dependency") != "(null)":
            raise ValueError("Running replacement retains an unexpected dependency")
    elif state == "COMPLETED":
        if observed.get("dependency") != "(null)" or observed.get("exit_code") != "0:0":
            raise ValueError("Completed replacement lacks successful Slurm evidence")
    else:
        raise ValueError("Replacement release was not observed")
    if any(str(observed.get(key)) != str(value) for key, value in expected.items()):
        raise ValueError("Released replacement identity changed")
    if state != "RUNNING" and observed.get("has_step") is not False:
        raise ValueError("Released dependency monitor unexpectedly has a step")


def _validate_artifact_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    required = {
        "policy_sha256",
        "journal_sha256",
        "claim_sha256",
        "provenance_sha256",
    }
    if set(evidence) != required:
        raise ValueError("Artifact validation did not return exact SHA evidence")
    for field in required:
        _hex_digest(evidence[field], field)
    return evidence


def run_handover(
    receipt_dir: str | Path,
    *,
    lock_path: str | Path,
    spec: dict[str, Any],
    observe: Callable[[str], dict[str, Any]],
    find_held: Callable[[dict[str, Any]], Iterable[str]],
    submit_held: Callable[[dict[str, Any]], str],
    cancel: Callable[[str], None],
    preflight_artifacts: Callable[[dict[str, Any]], dict[str, str]],
    commit_artifacts: Callable[[dict[str, Any], str], dict[str, Any]],
    release: Callable[[str], None],
    account_gpu_count: Callable[[], int],
) -> dict[str, Any]:
    """Replace one dependency monitor without touching its live GPU predecessor."""
    validate_spec(spec)
    if str(Path(lock_path).resolve()) != str(Path(spec["shared_lock"]).resolve()):
        raise ValueError("Caller lock does not match the immutable handover spec")
    root, digest = Path(receipt_dir), spec_sha256(spec)
    with exclusive_existing_lock(lock_path):
        root.mkdir(parents=True, exist_ok=True)
        receipts = _read_receipts(root, digest)
        phase = receipts[-1]["phase"] if receipts else None
        new_job = next(
            (
                str(receipt["new_job_id"])
                for receipt in reversed(receipts)
                if receipt.get("new_job_id") is not None
            ),
            None,
        )
        if phase is None:
            old = observe(str(spec["old_job_id"]))
            gpu = observe(str(spec["gpu_job_id"]))
            _assert_old_pending(old, spec)
            _assert_gpu_preserved(gpu, spec)
            gpu_count = account_gpu_count()
            if not 0 <= gpu_count <= spec["max_account_gpus"]:
                raise ValueError("Live running-plus-pending GPU count exceeds the approved ceiling")
            preflight = _validate_artifact_evidence(preflight_artifacts(spec))
            _record(root, "prepared", digest, old_job=old, gpu_job=gpu,
                    account_running_plus_pending_gpus=gpu_count,
                    artifact_preflight=preflight, spec=spec)
            phase = "prepared"
        if phase == "prepared":
            candidates = [str(value) for value in find_held(spec)]
            if len(candidates) > 1:
                raise ValueError("Multiple held replacements match the immutable spec")
            new_job = candidates[0] if candidates else str(submit_held(spec))
            if not new_job.isdigit():
                raise ValueError("Held submission returned a malformed job id")
            _record(root, "held_submitted", digest, new_job_id=new_job,
                    recovered_existing=bool(candidates))
            phase = "held_submitted"
        if phase == "held_submitted":
            assert new_job is not None
            held = observe(new_job)
            _assert_held(held, spec, new_job)
            _record(root, "held_verified", digest, new_job_id=new_job, held_job=held)
            phase = "held_verified"
        if phase == "held_verified":
            _assert_old_pending(observe(str(spec["old_job_id"])), spec)
            _assert_gpu_preserved(observe(str(spec["gpu_job_id"])), spec)
            preflight = _validate_artifact_evidence(preflight_artifacts(spec))
            _record(
                root,
                "cancel_requested",
                digest,
                new_job_id=new_job,
                artifact_preflight=preflight,
            )
            phase = "cancel_requested"
        if phase == "cancel_requested":
            old = observe(str(spec["old_job_id"]))
            if old.get("state") == "PENDING":
                _assert_old_pending(old, spec)
                _validate_artifact_evidence(preflight_artifacts(spec))
                cancel(str(spec["old_job_id"]))
                old = observe(str(spec["old_job_id"]))
            if old.get("state") != "CANCELLED" or old.get("has_step") is not False:
                raise ValueError("Old monitor did not reach exact cancelled-without-step state")
            _record(root, "old_cancelled", digest, new_job_id=new_job, old_job=old)
            phase = "old_cancelled"
        if phase == "old_cancelled":
            assert new_job is not None
            evidence = commit_artifacts(spec, new_job)
            _validate_artifact_evidence(evidence)
            _record(root, "artifacts_committed", digest, new_job_id=new_job, **evidence)
            phase = "artifacts_committed"
        if phase == "artifacts_committed":
            assert new_job is not None
            _assert_held(observe(new_job), spec, new_job)
            _record(root, "release_requested", digest, new_job_id=new_job)
            phase = "release_requested"
        if phase == "release_requested":
            assert new_job is not None
            held_receipt = next(
                receipt
                for receipt in _read_receipts(root, digest)
                if receipt["phase"] == "held_verified"
            )
            _assert_held(held_receipt["held_job"], spec, new_job)
            observed = observe(new_job)
            if observed.get("state") == "PENDING" and observed.get("reason") == "JobHeldUser":
                _assert_held(observed, spec, new_job)
                release(new_job)
                observed = observe(new_job)
            _assert_released(observed, spec, new_job)
            _record(root, "release_verified", digest, new_job_id=new_job,
                    released_job=observed)
        return _read_receipts(root, digest)[-1]


def stage2_action(monitor_state: str) -> str:
    if monitor_state == "RUNNING":
        return "wait_for_stage1_monitor_terminal"
    if monitor_state == "PENDING":
        return "replace_pending_monitor_with_exact_next_policy_sha"
    if monitor_state in {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT"}:
        return "fresh_ownership_audit_before_next_policy_install"
    raise ValueError("Unknown monitor state")
