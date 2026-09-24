import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest

from look.runtime.v5_role_policy_transition import (
    build_pending_monitor_replacement,
    build_look_only_policy,
    prepare,
    verify_monitor_policy,
    verify_look_only_delta,
)


def write(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value, sort_keys=True))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def policy():
    return {
        "schema": "research_gpu_roles_v1",
        "account_ceiling": 24,
        "native_max_gpus": 0,
        "projects": {
            "LOOK": {"reserved_gpus": 7, "request_journals": ["/old.json"]},
            "Uncertainty_Lab": {"reserved_gpus": 2, "request_journals": ["/ul.json"]},
        },
    }


def test_policy_delta_only_appends_look_journal(tmp_path):
    current = policy()
    candidate = build_look_only_policy(current=current, dedicated_journal=tmp_path / "v5.json")
    verify_look_only_delta(current, candidate, tmp_path / "v5.json")
    assert candidate["projects"]["Uncertainty_Lab"] == current["projects"]["Uncertainty_Lab"]
    assert current["projects"]["LOOK"]["request_journals"] == ["/old.json"]


def test_delta_rejects_premature_models_journal(tmp_path):
    current = policy()
    candidate = build_look_only_policy(current=current, dedicated_journal=tmp_path / "v5.json")
    candidate["projects"]["Uncertainty_Lab"]["request_journals"].append("/models.json")
    with pytest.raises(ValueError, match="more than"):
        verify_look_only_delta(current, candidate, tmp_path / "v5.json")


def test_prepare_is_exclusive_and_binds_claim(tmp_path):
    current = tmp_path / "current.json"; current_sha = write(current, policy())
    claim = tmp_path / "claim.json"; claim_sha = write(claim, {"job_id": "52429877"})
    job = tmp_path / "job.json"
    write(job, {"job_id": "52429877", "name": "look_v5_mid_sharded",
                "command": "/immutable/run_gpu.sbatch", "submit_time": "2026-09-24T09:33:10",
                "state": "PENDING"})
    args = Namespace(current_policy=str(current), expected_current_sha256=current_sha,
                     job_identity=str(job), claim=str(claim), expected_claim_sha256=claim_sha,
                     output=str(tmp_path / "out"), finalizer_identity=None)
    receipt = prepare(args)
    assert receipt["state"] == "candidate_not_installed"
    assert receipt["models_next_previous_policy_sha256"] == receipt["look_policy_sha256"]
    with pytest.raises(FileExistsError):
        prepare(args)


def test_prepare_rejects_nonlive_or_changed_claim(tmp_path):
    current = tmp_path / "current.json"; current_sha = write(current, policy())
    claim = tmp_path / "claim.json"; write(claim, {"job_id": "52429877"})
    job = tmp_path / "job.json"
    write(job, {"job_id": "52429877", "name": "x", "command": "/x",
                "submit_time": "t", "state": "COMPLETED"})
    args = Namespace(current_policy=str(current), expected_current_sha256=current_sha,
                     job_identity=str(job), claim=str(claim), expected_claim_sha256="0" * 64,
                     output=str(tmp_path / "out"), finalizer_identity=None)
    with pytest.raises(ValueError):
        prepare(args)


def test_pending_monitor_accepts_intermediate_look_policy_only(tmp_path):
    old = tmp_path / "old.json"; old_sha = write(old, {"policy": "old"})
    look = tmp_path / "look.json"; look_sha = write(look, {"policy": "look"})
    models = tmp_path / "models.json"; write(models, {"policy": "models"})
    binding = build_pending_monitor_replacement(
        finalizer={"job_id": "52430491", "state": "PENDING",
                   "dependency": "afterany:52429877(unfulfilled)", "command": "/immutable/v20/monitor.sh"},
        current_policy_sha256=old_sha, look_policy_sha256=look_sha,
    )
    assert verify_monitor_policy(old, binding) == old_sha
    assert verify_monitor_policy(look, binding) == look_sha
    with pytest.raises(ValueError, match="not authorized"):
        verify_monitor_policy(models, binding)
    assert binding["hot_edit_allowed"] is False
