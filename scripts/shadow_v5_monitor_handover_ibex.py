#!/usr/bin/env python3
"""Exercise the real Ibex adapter with isolated zero-GPU Slurm jobs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from look.runtime.v5_monitor_handover import run_handover, write_exclusive
from look.runtime.v5_monitor_ibex import (
    ArtifactInstaller,
    PRODUCTION_ROOT,
    SlurmAdapter,
    build_handover_spec,
    build_policy_binding,
    sha256,
)
from look.runtime.v5_role_policy_transition import write_exclusive_bytes, write_exclusive_json


def submit(script: Path, *arguments: str) -> str:
    output = subprocess.check_output(
        ["sbatch", "--parsable", *arguments, str(script)], text=True
    ).strip().split(";", 1)[0]
    if not output.isdigit():
        raise ValueError("Malformed shadow Slurm job id")
    return output


def wait(slurm: SlurmAdapter, job: str, states: set[str], seconds: float = 60) -> dict:
    deadline = time.monotonic() + seconds
    latest = None
    while time.monotonic() <= deadline:
        latest = slurm.observe(job)
        if latest.get("state") in states:
            return latest
        time.sleep(0.5)
    raise TimeoutError(f"Shadow job did not reach {states}: {latest}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if root.exists() or root == PRODUCTION_ROOT or PRODUCTION_ROOT in root.parents:
        raise ValueError("Shadow root must be new and outside the production operation root")
    root.mkdir(parents=True)
    lock = root / "account.lock"; lock.touch(mode=0o600)
    parent_script = root / "parent.sh"
    monitor_script = root / "monitor.sh"
    write_exclusive_bytes(parent_script, b"#!/bin/bash\nsleep 180\n", mode=0o555)
    write_exclusive_bytes(
        monitor_script,
        f"#!/bin/bash\nprintf '%s\\n' \"$SLURM_JOB_ID\" > {str(root / 'replacement_ran.txt')!r}\n".encode(),
        mode=0o555,
    )
    slurm = SlurmAdapter(user=os.environ["USER"], account="pi-mengy", wait_seconds=30)
    parent = submit(
        parent_script,
        "--hold", "--job-name=lookv21_shadow_parent", "--account=pi-mengy", "--partition=batch",
        "--time=00:05:00", "--cpus-per-task=1", "--mem=128M",
        f"--output={root}/parent-%j.log",
    )
    old = submit(
        monitor_script,
        f"--dependency=afterany:{parent}", "--job-name=lookv21_shadow_old",
        "--account=pi-mengy", "--partition=batch", "--time=00:05:00",
        "--cpus-per-task=2", "--mem=4G", f"--output={root}/old-%j.log",
    )
    try:
        parent_row, old_row = slurm.observe(parent), slurm.observe(old)
        claim = root / "claim.json"
        write_exclusive(claim, {"schema": "shadow_claim_v1", "job_id": parent, "test_access": False})
        journal = root / "look_v5_requests.json"
        write_exclusive_json(
            journal,
            {"schema": "look_workflow_requests_v1", "requests": [
                {"job_id": parent, "state": "submitted", "v5_claim_sha256": sha256(claim)}
            ]},
        )
        base = root / "role_policy.base.json"
        base_value = {
            "schema": "research_gpu_roles_v1", "account_ceiling": 24, "native_max_gpus": 0,
            "projects": {"LOOK": {"reserved_gpus": 7, "request_journals": []}},
        }
        write_exclusive_json(base, base_value)
        candidate = root / "role_policy.look_only.json"
        candidate_value = json.loads(json.dumps(base_value))
        candidate_value["projects"]["LOOK"]["request_journals"].append(str(journal))
        write_exclusive_json(candidate, candidate_value)
        active = root / "role_policy.active.json"
        write_exclusive_bytes(active, base.read_bytes())
        provenance = root / "provenance.json"
        write_exclusive_json(provenance, {"schema": "look_v5_shadow_provenance_v1", "test_access": False})
        binding_value = build_policy_binding(
            active_policy=active, base_policy_snapshot=base, candidate_policy=candidate,
            dedicated_journal=journal, claim=claim, provenance=provenance,
            replacement_command=monitor_script, gpu_job_id=parent,
        )
        binding = root / "binding.json"; write_exclusive(binding, binding_value)
        spec = build_handover_spec(
            old_job=old_row, gpu_job=parent_row, replacement_command=monitor_script,
            binding_path=binding, shared_lock=lock, mode="shadow",
        )
        spec_path = root / "handover_spec.json"; write_exclusive(spec_path, spec)
        for label, row in {"parent": parent_row, "old_monitor": old_row}.items():
            if "gres/gpu" in str(row.get("req_tres", "")):
                raise RuntimeError(f"{label} unexpectedly requested a GPU: {row}")

        installer = ArtifactInstaller(binding, expected_sha256=sha256(binding))
        receipt_root = root / "receipts"
        common = {
            "lock_path": lock,
            "spec": spec,
            "observe": slurm.observe,
            "find_held": slurm.find_held,
            "submit_held": lambda value: slurm.submit_held(value, claim, root),
            "cancel": slurm.cancel,
            "preflight_artifacts": installer.preflight,
            "account_gpu_count": slurm.account_gpu_count,
        }

        policy_ack_fault = False

        def commit_then_drop_ack(value: dict, new_job_id: str) -> dict:
            nonlocal policy_ack_fault
            evidence = installer.commit(value, new_job_id)
            if not policy_ack_fault:
                policy_ack_fault = True
                raise RuntimeError("injected lost policy-install acknowledgement")
            return evidence

        try:
            run_handover(
                receipt_root,
                commit_artifacts=commit_then_drop_ack,
                release=slurm.release,
                **common,
            )
        except RuntimeError as error:
            if str(error) != "injected lost policy-install acknowledgement":
                raise
        else:
            raise RuntimeError("policy-install acknowledgement fault was not exercised")

        release_ack_fault = False

        def release_then_drop_ack(job_id: str) -> None:
            nonlocal release_ack_fault
            slurm.release(job_id)
            if not release_ack_fault:
                release_ack_fault = True
                raise RuntimeError("injected lost release acknowledgement")

        try:
            run_handover(
                receipt_root,
                commit_artifacts=installer.commit,
                release=release_then_drop_ack,
                **common,
            )
        except RuntimeError as error:
            if str(error) != "injected lost release acknowledgement":
                raise
        else:
            raise RuntimeError("release acknowledgement fault was not exercised")

        result = run_handover(
            receipt_root,
            commit_artifacts=installer.commit,
            release=slurm.release,
            **common,
        )
        replacement = result["new_job_id"]
        replacement_row = slurm.observe(replacement)
        if "gres/gpu" in str(replacement_row.get("req_tres", "")):
            raise RuntimeError(f"replacement monitor unexpectedly requested a GPU: {replacement_row}")
        slurm.cancel(parent)
        terminal = wait(slurm, replacement, {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"})
        if terminal.get("state") != "COMPLETED" or not (root / "replacement_ran.txt").is_file():
            raise RuntimeError(f"Released shadow replacement did not execute successfully: {terminal}")
        final = {
            "schema": "look_v5_monitor_ibex_zero_gpu_shadow_v1",
            "state": "accepted",
            "parent_job": parent,
            "old_monitor_job": old,
            "replacement_monitor_job": replacement,
            "replacement_terminal": terminal,
            "requested_gpus": 0,
            "phase": result["phase"],
            "faults_exercised": {
                "policy_install_ack_lost": policy_ack_fault,
                "release_ack_lost": release_ack_fault,
            },
            "active_policy_sha256": sha256(active),
            "candidate_policy_sha256": sha256(candidate),
            "test_access": False,
        }
        write_exclusive(root / "shadow_receipt.json", final)
        print(json.dumps(final, sort_keys=True))
    except BaseException:
        for job in (old, parent):
            subprocess.run(["scancel", str(job)], check=False)
        raise


if __name__ == "__main__":
    main()
