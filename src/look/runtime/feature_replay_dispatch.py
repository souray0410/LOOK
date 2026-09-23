"""Strict finite dispatcher contract for a two-phase V4-to-V5 feature replay."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

from look.runtime.state import atomic_write_json, file_sha256, stable_hash


def read(path):
    return json.loads(Path(path).read_text())


def validate_spec(spec):
    if (spec.get("schema") != "look_feature_replay_dispatch_v1"
            or spec.get("test_access") is not False
            or spec.get("production_cutover") is not False
            or spec.get("phases") != ["reference", "check"]):
        raise ValueError("Invalid feature-replay dispatch specification")
    required = ("package_id", "run_phase_script", "run_phase_script_sha256",
                "package_manifest", "package_manifest_sha256", "weekly_delivery_policy",
                "weekly_delivery_policy_sha256", "expected_counts", "expected_sites")
    if any(key not in spec for key in required):
        raise ValueError("Incomplete feature-replay dispatch specification")
    if spec["expected_counts"] != {"train": 1264, "development": 296}:
        raise ValueError("Feature-replay cohort contract changed")
    if not isinstance(spec["expected_sites"], list) or len(spec["expected_sites"]) != 9:
        raise ValueError("Feature-replay site contract changed")
    for path_key, sha_key in (("run_phase_script", "run_phase_script_sha256"),
                              ("package_manifest", "package_manifest_sha256"),
                              ("weekly_delivery_policy", "weekly_delivery_policy_sha256")):
        if file_sha256(spec[path_key]) != spec[sha_key]:
            raise ValueError("Pinned feature-replay input changed: " + path_key)
    package = read(spec["package_manifest"])
    if (package.get("package_id") != spec["package_id"]
            or package.get("dispatch_authorized") is not False
            or package.get("preservation", {}).get("test_access") is not False):
        raise ValueError("Staged feature-replay package identity changed")
    return spec


def prerequisite_state(spec):
    validate_spec(spec)
    from look.runtime.weekly_delivery import release_state
    return release_state(spec["weekly_delivery_policy"])


def _active_slurm_identity(job, step):
    job_text = subprocess.check_output(["scontrol", "show", "job", job, "-o"], text=True, timeout=20)
    step_text = subprocess.check_output(["scontrol", "show", "step", f"{job}.{step}", "-o"], text=True, timeout=20)
    if f"JobId={job}" not in job_text or "JobState=RUNNING" not in job_text:
        raise ValueError("Feature-replay Slurm allocation is not active")
    if f"StepId={job}.{step}" not in step_text and f"StepId={step}" not in step_text:
        raise ValueError("Feature-replay Slurm step identity changed")


def validate_claim(spec_path, run, environ=None, active_slurm=_active_slurm_identity):
    environ = os.environ if environ is None else environ
    claim_path = environ.get("LOOK_ROTATION_CLAIM")
    job = environ.get("SLURM_JOB_ID"); step = environ.get("SLURM_STEP_ID")
    if not claim_path or not job or not step:
        raise ValueError("Feature replay requires claim, Slurm job and Slurm step identity")
    claim_file = Path(claim_path)
    if not claim_file.is_file() or claim_file.is_symlink():
        raise ValueError("Feature-replay claim path is not an immutable regular file")
    claim = read(claim_file); expected_spec = file_sha256(spec_path)
    if (claim.get("state") not in ("claimed", "running")
            or str(claim.get("run_dir", "")) != str(Path(run).resolve())
            or claim.get("spec_sha256") != expected_spec
            or claim.get("owner") != "look-workflow-" + str(job)
            or str(claim.get("job_id")) != str(job)
            or not isinstance(claim.get("generation"), int) or claim["generation"] < 1
            or claim.get("test_access", False) is not False):
        raise ValueError("Feature-replay claim identity changed")
    if claim.get("step") is not None and str(claim["step"]) != str(step):
        raise ValueError("Feature-replay claim step identity changed")
    active_slurm(str(job), str(step))
    return {"claim_path": str(claim_file.resolve()), "claim_sha256": file_sha256(claim_file),
            "owner": claim["owner"], "job_id": str(job), "step_id": str(step),
            "generation": claim["generation"], "dispatch_spec_sha256": expected_spec}


def verify_case(run, spec):
    validate_spec(spec); run = Path(run)
    reference = read(run / "reference_receipt.json")
    check = read(run / "check_receipt.json")
    if (reference.get("schema") != "look_node_feature_replay_v1"
            or reference.get("state") != "reference_exported"
            or reference.get("test_access") is not False
            or reference.get("counts") != spec["expected_counts"]
            or reference.get("sites") != spec["expected_sites"]):
        raise ValueError("Feature-replay reference receipt rejected")
    if (check.get("schema") != "look_node_feature_replay_v1"
            or check.get("state") != "accepted"
            or check.get("test_access") is not False
            or check.get("production_cutover") is not False
            or check.get("counts") != spec["expected_counts"]
            or check.get("sites") != spec["expected_sites"]
            or check.get("all_site_values_bitwise_equal") is not True):
        raise ValueError("Feature-replay V5 receipt rejected")
    if any(check.get(key) != reference.get(key) for key in
           ("source_checkpoint_sha256", "original_bank_sha256", "sites", "counts", "batches")):
        raise ValueError("Feature-replay phase identity changed")
    accepted = read(run / "accepted.json")
    if (accepted.get("schema") != "look_feature_replay_dispatch_acceptance_v1"
            or accepted.get("state") != "accepted"
            or accepted.get("identity") != stable_hash(spec)
            or accepted.get("reference_receipt_sha256") != file_sha256(run / "reference_receipt.json")
            or accepted.get("check_receipt_sha256") != file_sha256(run / "check_receipt.json")
            or accepted.get("run_dir") != str(run.resolve())
            or accepted.get("owner") != "look-workflow-" + str(accepted.get("job_id"))
            or not str(accepted.get("job_id", "")).isdigit()
            or not str(accepted.get("step_id", ""))
            or not isinstance(accepted.get("generation"), int) or accepted["generation"] < 1
            or len(str(accepted.get("claim_sha256", ""))) != 64
            or len(str(accepted.get("dispatch_spec_sha256", ""))) != 64
            or accepted.get("test_access") is not False):
        raise ValueError("Feature-replay dispatcher acceptance changed")
    return accepted


def execute(spec_path, run):
    spec = validate_spec(read(spec_path)); run = Path(run).resolve()
    if run != Path(read(spec["package_manifest"])["execution"]["output"]).resolve():
        raise ValueError("Feature-replay run directory differs from staged output")
    gate = prerequisite_state(spec)
    if gate.get("released") is not True:
        raise ValueError("Feature replay remains behind the whole-weekly prerequisite")
    claim = validate_claim(spec_path, run)
    for phase in spec["phases"]:
        subprocess.run([spec["run_phase_script"], phase], check=True)
    reference = run / "reference_receipt.json"; check = run / "check_receipt.json"
    provisional = {"schema": "look_feature_replay_dispatch_acceptance_v1", "state": "accepted",
        "identity": stable_hash(spec), "reference_receipt_sha256": file_sha256(reference),
        "check_receipt_sha256": file_sha256(check), "test_access": False,
        "run_dir": str(run), **claim,
        "production_cutover": False, "completed_at": time.time()}
    atomic_write_json(provisional, run / "accepted.json")
    verify_case(run, spec)
    atomic_write_json({"state": "completed", "identity": stable_hash(spec),
                       "test_access": False, "updated_at": time.time()}, run / "status.json")
    return provisional
