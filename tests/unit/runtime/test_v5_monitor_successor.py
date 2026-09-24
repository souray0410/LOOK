import copy
import hashlib
import json
from pathlib import Path

import pytest

from look.runtime.v5_monitor_successor import (
    build_failed_terminal_evidence,
    build_owner_contract,
    build_policy_plan,
    prepare_v22_package,
    verify_policy_plan,
)


def write(path: Path, value, *, mode=0o644) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n" if isinstance(value, dict) else value
    path.write_text(payload)
    path.chmod(mode)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_plan(tmp_path: Path):
    active = tmp_path / "active.json"
    lock = tmp_path / "account.lock"; write(lock, "")
    base = {
        "projects": {
            "LOOK": {"request_journals": ["/look.json"]},
            "Uncertainty_Lab": {"request_journals": ["/old-ul.json"]},
        }
    }
    write(active, base)
    claim = tmp_path / "look-claim.json"
    claim_sha = write(claim, {"job_id": "52429877", "state": "submitted"})
    look_journal = tmp_path / "look-journal.json"
    write(look_journal, {"requests": [{"job_id": "52429877", "v5_claim_sha256": claim_sha}]})
    source = tmp_path / "source.tar"; write(source, "source")
    states = []
    previous = base
    allowed = [hashlib.sha256(active.read_bytes()).hexdigest()]
    for number, (lane, failed) in enumerate((("even", "52447420"), ("odd", "52447453")), start=1):
        packet = tmp_path / lane / "packet.json"
        packet_sha = write(packet, {
            "lane_id": lane,
            "pins": {str(source.resolve()): hashlib.sha256(source.read_bytes()).hexdigest()},
            "test_access": False,
        })
        batch = tmp_path / lane / "submit.sbatch"; batch_sha = write(batch, "#!/bin/bash\n")
        journal = tmp_path / lane / "requests.json"
        write(journal, {
            "schema": "ul_v5_production_request_v1", "project": "Uncertainty_Lab",
            "lane_id": lane, "packet": str(packet.resolve()), "packet_sha256": packet_sha,
            "state": "prepared", "requests": [], "requested_gpus": 1,
            "account_lock": str(lock.resolve()), "policy": str(active.resolve()),
            "batch_script": str(batch.resolve()), "batch_script_sha256": batch_sha,
        })
        evidence = tmp_path / lane / "failed.json"
        write(evidence, build_failed_terminal_evidence({
            "job_id": failed, "name": f"old-{lane}", "state": "FAILED", "exit_code": "1:0",
            "user": "mengh", "account": "pi-mengy",
            "req_tres": "cpu=8,gres/gpu:a100=1,gres/gpu=1,mem=64G,node=1",
        }))
        contract = tmp_path / lane / "owner.json"
        write(contract, build_owner_contract(
            lane=lane, journal=journal, packet=packet,
            source_commit="0" * 40, source_archive=source,
            predecessor_failed_job=failed, predecessor_terminal_evidence=evidence,
        ))
        candidate_value = copy.deepcopy(previous)
        candidate_value["projects"]["Uncertainty_Lab"]["request_journals"].append(str(journal.resolve()))
        candidate = tmp_path / f"policy-{number}.json"; allowed.append(write(candidate, candidate_value))
        states.append({
            "candidate_policy": str(candidate), "appended_journal": str(journal),
            "owner_contract": str(contract),
        })
        previous = candidate_value
    plan = tmp_path / "plan.json"
    write(plan, build_policy_plan(
        active_policy=active, look_gpu_job_id="52429877", look_claim=claim,
        look_journal=look_journal, states=states,
    ))
    return active, lock, plan, allowed


def test_policy_plan_accepts_only_two_ordered_ul_appends(tmp_path):
    active, lock, plan, allowed = make_plan(tmp_path)
    verified = verify_policy_plan(
        plan, expected_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
        expected_active_policy=active,
        expected_account_lock=lock,
    )
    assert verified["allowed_policy_sha256"] == allowed


def test_policy_plan_rejects_unrelated_policy_change(tmp_path):
    active, lock, plan, _ = make_plan(tmp_path)
    value = json.loads(plan.read_text())
    candidate = Path(value["states"][0]["candidate_policy"])
    changed = json.loads(candidate.read_text()); changed["unexpected"] = True
    value["states"][0]["candidate_policy_sha256"] = write(candidate, changed)
    write(plan, value)
    with pytest.raises(ValueError, match="more than one ordered UL journal append"):
        verify_policy_plan(
            plan, expected_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
            expected_active_policy=active,
            expected_account_lock=lock,
        )


def test_v22_package_is_self_contained_and_has_exact_policy_allowlist(tmp_path):
    active, lock, plan, allowed = make_plan(tmp_path)
    old = tmp_path / "v21"; new = tmp_path / "v22"
    old.mkdir(); (old / "source").mkdir(); write(old / "source/helper.py", "x=1\n")
    old_allow = "{" + ",".join(repr(value) for value in allowed[:2]) + "}"
    chain_sha = write(old / "chain_manager.py", f"PKG={str(old)!r}\nALLOW={old_allow}\n")
    monitor_sha = write(old / "monitor.sh", f"#!/bin/bash\npython {old}/chain_manager.py\n", mode=0o555)
    for name in ("stage_manager.py", "stage_monitor.sh", "run_gpu.sbatch", "run_pca.sbatch", "run_decision.sbatch"):
        write(old / name, f"ROOT={str(old)!r}\n")
    result = prepare_v22_package(
        v21_package=old, v22_package=new, v21_chain_sha256=chain_sha,
        v21_monitor_sha256=monitor_sha, expected_v21_allowed_policy_sha256=allowed[:2],
        plan_path=plan, plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
        expected_active_policy=active,
        expected_account_lock=lock,
    )
    assert result["allowed_policy_sha256"] == allowed
    assert str(old) not in (new / "chain_manager.py").read_text()
    assert "ALLOW={" + ",".join(repr(value) for value in allowed) + "}" in (new / "chain_manager.py").read_text()
