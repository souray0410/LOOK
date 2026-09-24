#!/usr/bin/env python3
"""Run the v22 successor through the real Ibex adapter with zero GPU jobs."""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import time
from pathlib import Path

from look.runtime.v5_monitor_handover import run_handover, write_exclusive
from look.runtime.v5_monitor_ibex import (
    PRODUCTION_ROOT,
    SlurmAdapter,
    build_handover_spec,
    sha256,
)
from look.runtime.v5_monitor_successor import (
    PolicyPlanMonitorInstaller,
    build_failed_terminal_evidence,
    build_owner_contract,
    build_policy_plan,
    build_successor_binding,
    prepare_v22_package,
)
from look.runtime.v5_role_policy_transition import write_exclusive_bytes


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


def recover(root: Path) -> None:
    prepared = json.loads((root / "receipts/00-prepared.json").read_text())
    spec = prepared["spec"]
    last_phase = sorted((root / "receipts").glob("*.json"))[-1].stem.split("-", 1)[1]
    plan = json.loads((root / "policy-plan.json").read_text())
    slurm = SlurmAdapter(
        user=spec["expected_user"], account=spec["expected_account"], wait_seconds=30
    )
    installer = PolicyPlanMonitorInstaller(
        root / "binding.json",
        expected_sha256=spec["binding_sha256"],
        expected_active_policy=plan["active_policy"],
        expected_account_lock=spec["shared_lock"],
        expected_look_gpu_job_id=spec["gpu_job_id"],
        expected_user=spec["expected_user"],
        expected_account=spec["expected_account"],
        required_ul_gres_token=None,
    )
    result = run_handover(
        root / "receipts",
        lock_path=spec["shared_lock"],
        spec=spec,
        observe=slurm.observe,
        find_held=slurm.find_held,
        submit_held=lambda value: slurm.submit_held(value, installer.look_claim(), root),
        cancel=slurm.cancel,
        preflight_artifacts=installer.preflight,
        commit_artifacts=installer.commit,
        release=slurm.release,
        account_gpu_count=slurm.account_gpu_count,
    )
    parent = slurm.observe(str(spec["gpu_job_id"]))
    if parent.get("state") == "PENDING":
        slurm.cancel(str(spec["gpu_job_id"]))
    replacement = result["new_job_id"]
    done = wait(slurm, replacement, {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"})
    if done.get("state") != "COMPLETED" or not (root / "v22/replacement_ran.txt").is_file():
        raise RuntimeError(f"Recovered V22 replacement did not execute successfully: {done}")
    final = {
        "schema": "look_v5_monitor_v22_zero_gpu_shadow_v1",
        "state": "accepted",
        "recovered_from_phase": last_phase,
        "parent_job": str(spec["gpu_job_id"]),
        "old_monitor_job": str(spec["old_job_id"]),
        "replacement_monitor_job": replacement,
        "replacement_terminal": done,
        "requested_gpus": 0,
        "phase": result["phase"],
        "policy_plan_sha256": sha256(root / "policy-plan.json"),
        "test_access": False,
    }
    _write(root / "shadow_receipt.json", final)
    print(json.dumps(final, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--recover", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if root == PRODUCTION_ROOT or PRODUCTION_ROOT in root.parents:
        raise ValueError("Shadow root must be outside the production operation root")
    if args.recover:
        if not root.is_dir():
            raise ValueError("Recovery root does not exist")
        recover(root)
        return
    if root.exists():
        raise ValueError("Shadow root must be new and outside the production operation root")
    root.mkdir(parents=True)
    lock = root / "account.lock"; lock.touch(mode=0o600)
    parent_script = root / "parent.sh"
    old_script = root / "old-monitor.sh"
    fail_script = root / "failed-predecessor.sh"
    write_exclusive_bytes(parent_script, b"#!/bin/bash\nsleep 180\n", mode=0o555)
    write_exclusive_bytes(old_script, b"#!/bin/bash\nexit 0\n", mode=0o555)
    write_exclusive_bytes(fail_script, b"#!/bin/bash\nexit 1\n", mode=0o555)
    account, user = "pi-mengy", os.environ["USER"]
    slurm = SlurmAdapter(user=user, account=account, wait_seconds=30)
    common_submit = [
        f"--account={account}", "--partition=batch", "--time=00:05:00",
    ]
    failed = [
        submit(fail_script, *common_submit, "--cpus-per-task=1", "--mem=128M", f"--job-name=lookv22_shadow_failed_{lane}", f"--output={root}/failed-{lane}-%j.log")
        for lane in ("even", "odd")
    ]
    terminal = [wait(slurm, job, {"FAILED"}) for job in failed]
    parent = submit(
        parent_script, "--hold", *common_submit, "--cpus-per-task=1", "--mem=128M", "--job-name=lookv22_shadow_parent",
        f"--output={root}/parent-%j.log",
    )
    old = submit(
        old_script, f"--dependency=afterany:{parent}", *common_submit,
        "--cpus-per-task=2", "--mem=4G", "--job-name=lookv22_shadow_old",
        f"--output={root}/old-%j.log",
    )
    try:
        parent_row, old_row = slurm.observe(parent), slurm.observe(old)
        claim = root / "look-claim.json"
        claim_sha = _write(claim, {"job_id": parent, "state": "submitted"})
        look_journal = root / "look-journal.json"
        _write(look_journal, {"requests": [{"job_id": parent, "v5_claim_sha256": claim_sha}]})
        active = root / "role-policy.json"
        active_value = {"projects": {"LOOK": {"request_journals": [str(look_journal)]}, "Uncertainty_Lab": {"request_journals": []}}}
        _write(active, active_value)
        source = root / "source.tar"; write_exclusive_bytes(source, b"shadow-source")
        states, candidate_values = [], []
        previous = active_value
        for lane, failed_row in zip(("even", "odd"), terminal):
            lane_root = root / lane; lane_root.mkdir()
            packet = lane_root / "packet.json"
            _write(packet, {"lane_id": lane, "pins": {str(source): sha256(source)}, "test_access": False})
            batch = lane_root / "submit.sbatch"; write_exclusive_bytes(batch, b"#!/bin/bash\nexit 0\n", mode=0o555)
            journal = lane_root / "requests.json"
            _write(journal, {
                "schema": "ul_v5_production_request_v1", "project": "Uncertainty_Lab",
                "lane_id": lane, "packet": str(packet), "packet_sha256": sha256(packet),
                "state": "prepared", "requests": [], "requested_gpus": 1,
                "account_lock": str(lock), "policy": str(active),
                "batch_script": str(batch), "batch_script_sha256": sha256(batch),
            })
            evidence = lane_root / "terminal.json"; _write(evidence, build_failed_terminal_evidence(failed_row))
            contract = lane_root / "owner.json"
            _write(contract, build_owner_contract(
                lane=lane, journal=journal, packet=packet, source_commit="0" * 40,
                source_archive=source, predecessor_failed_job=failed_row["job_id"],
                predecessor_terminal_evidence=evidence,
            ))
            candidate_value = copy.deepcopy(previous)
            candidate_value["projects"]["Uncertainty_Lab"]["request_journals"].append(str(journal))
            candidate = root / f"policy-{lane}.json"; _write(candidate, candidate_value)
            states.append({"candidate_policy": str(candidate), "appended_journal": str(journal), "owner_contract": str(contract)})
            candidate_values.append(candidate)
            previous = candidate_value
        plan = root / "policy-plan.json"
        _write(plan, build_policy_plan(
            active_policy=active, look_gpu_job_id=parent, look_claim=claim,
            look_journal=look_journal, states=states,
        ))

        v21 = root / "v21"; v21.mkdir(); (v21 / "source").mkdir()
        write_exclusive_bytes(v21 / "source/helper.py", b"x=1\n")
        old_allowed = ["f" * 64, sha256(active)]
        old_allow_text = "{" + ",".join(repr(value) for value in old_allowed) + "}"
        write_exclusive_bytes(v21 / "chain_manager.py", f"PKG={str(v21)!r}\nALLOW={old_allow_text}\n".encode())
        write_exclusive_bytes(
            v21 / "monitor.sh",
            f"#!/bin/bash\nprintf '%s\\n' \"$SLURM_JOB_ID\" > {str(v21 / 'replacement_ran.txt')!r}\n".encode(),
            mode=0o555,
        )
        for name in ("stage_manager.py", "stage_monitor.sh", "run_gpu.sbatch", "run_pca.sbatch", "run_decision.sbatch"):
            write_exclusive_bytes(v21 / name, f"ROOT={str(v21)!r}\n".encode())
        v22 = root / "v22"
        package = prepare_v22_package(
            v21_package=v21, v22_package=v22,
            v21_chain_sha256=sha256(v21 / "chain_manager.py"),
            v21_monitor_sha256=sha256(v21 / "monitor.sh"),
            expected_v21_allowed_policy_sha256=old_allowed,
            plan_path=plan, plan_sha256=sha256(plan),
            expected_active_policy=active, expected_account_lock=lock,
            expected_look_gpu_job_id=parent, expected_user=user,
            expected_account=account, required_ul_gres_token=None,
        )
        binding = root / "binding.json"
        _write(binding, build_successor_binding(
            plan_path=plan, package_manifest=v22 / "package_manifest.json",
            replacement_command=v22 / "monitor.sh", gpu_job_id=parent,
        ))
        spec = build_handover_spec(
            old_job=old_row, gpu_job=parent_row, replacement_command=v22 / "monitor.sh",
            binding_path=binding, shared_lock=lock, mode="shadow",
        )
        installer = PolicyPlanMonitorInstaller(
            binding, expected_sha256=sha256(binding), expected_active_policy=active,
            expected_account_lock=lock, expected_look_gpu_job_id=parent,
            expected_user=user, expected_account=account, required_ul_gres_token=None,
        )
        receipt_root = root / "receipts"
        result = run_handover(
            receipt_root, lock_path=lock, spec=spec, observe=slurm.observe,
            find_held=slurm.find_held,
            submit_held=lambda value: slurm.submit_held(value, installer.look_claim(), root),
            cancel=slurm.cancel, preflight_artifacts=installer.preflight,
            commit_artifacts=installer.commit, release=slurm.release,
            account_gpu_count=slurm.account_gpu_count,
        )
        replacement = result["new_job_id"]
        replacement_row = slurm.observe(replacement)
        for label, row in {"parent": parent_row, "old": old_row, "replacement": replacement_row, "failed_even": terminal[0], "failed_odd": terminal[1]}.items():
            if "gres/gpu" in str(row.get("req_tres", "")):
                raise RuntimeError(f"{label} unexpectedly requested a GPU: {row}")
        slurm.cancel(parent)
        done = wait(slurm, replacement, {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"})
        if done.get("state") != "COMPLETED" or not (v22 / "replacement_ran.txt").is_file():
            raise RuntimeError(f"V22 replacement did not execute successfully: {done}")
        final = {
            "schema": "look_v5_monitor_v22_zero_gpu_shadow_v1", "state": "accepted",
            "parent_job": parent, "old_monitor_job": old, "replacement_monitor_job": replacement,
            "failed_predecessor_jobs": failed, "replacement_terminal": done,
            "requested_gpus": 0, "phase": result["phase"],
            "allowed_policy_sha256": package["allowed_policy_sha256"],
            "policy_plan_sha256": sha256(plan), "test_access": False,
        }
        _write(root / "shadow_receipt.json", final)
        print(json.dumps(final, sort_keys=True))
    except BaseException as error:
        failure = root / "shadow_failure.json"
        if not failure.exists():
            phases = [path.stem.split("-", 1)[1] for path in sorted((root / "receipts").glob("*.json"))]
            _write(failure, {
                "schema": "look_v5_monitor_v22_zero_gpu_shadow_failure_v1",
                "state": "failed_recoverable",
                "error_type": type(error).__name__,
                "error": str(error),
                "completed_phases": phases,
                "requested_gpus": 0,
                "test_access": False,
            })
        for job in (old, parent):
            subprocess.run(["scancel", str(job)], check=False)
        raise


def _write(path: Path, value: dict) -> str:
    write_exclusive(path, value)
    return sha256(path)


if __name__ == "__main__":
    main()
