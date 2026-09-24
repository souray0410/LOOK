import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest

from look.runtime.v5_role_policy_transition import (
    build_pending_monitor_replacement,
    build_v21_chain_manager,
    build_look_only_policy,
    prepare_v21_package,
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
    replacement = tmp_path / "replacement.json"; replacement_sha = write(replacement, {"replacement": True})
    job = tmp_path / "job.json"
    write(job, {"job_id": "52429877", "name": "look_v5_mid_sharded",
                "command": "/immutable/run_gpu.sbatch", "submit_time": "2026-09-24T09:33:10",
                "state": "PENDING", "user": "mengh", "account": "pi-mengy", "req_tres": "gres/gpu:a100:1",
                "replaces_job_id": "52422962", "cancelled_retry_job_id": "52429606",
                "replacement_receipt": str(replacement), "replacement_receipt_sha256": replacement_sha})
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
    replacement = tmp_path / "replacement.json"; replacement_sha = write(replacement,{})
    write(job, {"job_id": "52429877", "name": "x", "command": "/x",
                "submit_time": "t", "state": "COMPLETED", "user":"mengh", "account":"pi-mengy",
                "req_tres":"gpu:1", "replaces_job_id":"52422962", "cancelled_retry_job_id":"52429606",
                "replacement_receipt":str(replacement), "replacement_receipt_sha256":replacement_sha})
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
                   "dependency": "afterany:52429877(unfulfilled)", "command": "/immutable/v20/monitor.sh",
                   "source_sha256":"e717d053"},
        current_policy_sha256=old_sha, look_policy_sha256=look_sha,
    )
    assert verify_monitor_policy(old, binding) == old_sha
    assert verify_monitor_policy(look, binding) == look_sha
    with pytest.raises(ValueError, match="not authorized"):
        verify_monitor_policy(models, binding)
    assert binding["hot_edit_allowed"] is False


def test_v21_redirects_successors_to_dedicated_journal(tmp_path):
    package = tmp_path/'v5_rollout_20260922'/'v20'
    package.mkdir(parents=True)
    old = package/'chain_manager.py'
    old.write_text("import pathlib\nPKG=ROOT/'v5_rollout_20260922/v20'\nJOURNAL=ROOT/'look_forward_20260916/lease_recovery_20260917/dispatcher/requests.json'\n"
                   "if policy_sha not in {'afde7998b16d39332818f2b1f6ad87401b7ff42f653666f85a922065cc1b3b72','efc451893fbeedf4fb2e22aa07a090910e3530e8e8595e33fb3aebc5e791ec02'}:raise ValueError\n")
    digest=hashlib.sha256(old.read_bytes()).hexdigest()
    new_package=tmp_path/'v5_rollout_20260922'/'v21'
    result=build_v21_chain_manager(v20_source=old,expected_sha256=digest,
                                   v20_package=package,v21_package=new_package,
                                   dedicated_journal=tmp_path/'dedicated.json',allowed_policy_sha256=['old','look'])
    assert str((tmp_path/'dedicated.json').resolve()) in result
    assert "lease_recovery_20260917" not in result
    assert "'old','look'" in result
    assert str(new_package.resolve()) in result
    assert str(package.resolve()) not in result


def test_v21_package_owns_all_future_stage_paths(tmp_path):
    old=tmp_path/'v5_rollout_20260922'/'v20'; old.mkdir(parents=True)
    new=tmp_path/'v5_rollout_20260922'/'v21'; journal=tmp_path/'look_v5_requests.json'
    (old/'source/pkg').mkdir(parents=True); (old/'source/pkg/module.py').write_text('VALUE=1\n')
    (old/'chain_manager.py').write_text(
        "import pathlib\nROOT=pathlib.Path('/root')\nPKG=ROOT/'v5_rollout_20260922/v20';JOURNAL=ROOT/'look_forward_20260916/lease_recovery_20260917/dispatcher/requests.json'\n"
        "if policy_sha not in {'afde7998b16d39332818f2b1f6ad87401b7ff42f653666f85a922065cc1b3b72','efc451893fbeedf4fb2e22aa07a090910e3530e8e8595e33fb3aebc5e791ec02'}:raise ValueError\n"
    )
    (old/'stage_manager.py').write_text(
        "import pathlib\nROOT=pathlib.Path('/root')\nPKG=ROOT/'v5_rollout_20260922/v20';J=ROOT/'look_forward_20260916/lease_recovery_20260917/dispatcher/requests.json'\n"
    )
    for name in ('monitor.sh','stage_monitor.sh','run_gpu.sbatch','run_pca.sbatch','run_decision.sbatch'):
        (old/name).write_text(f'#!/bin/bash\nB={old.resolve()}\n')
    result=prepare_v21_package(
        v20_package=old,v21_package=new,
        v20_chain_sha256=hashlib.sha256((old/'chain_manager.py').read_bytes()).hexdigest(),
        v20_monitor_sha256=hashlib.sha256((old/'monitor.sh').read_bytes()).hexdigest(),
        dedicated_journal=journal,allowed_policy_sha256=['base','look'],
    )
    assert result['v21_package']==str(new.resolve())
    for path in new.rglob('*'):
        if path.is_file() and path.name!='package_manifest.json':
            assert str(old.resolve()) not in path.read_text(errors='ignore')
    assert str(journal.resolve()) in (new/'chain_manager.py').read_text()
    assert str(journal.resolve()) in (new/'stage_manager.py').read_text()
