import copy
import hashlib
import json
from pathlib import Path

import pytest

from look.runtime.v5_monitor_radon_successor import (
    RUNNER_COMMIT,
    build_radon_owner_contract,
    build_radon_policy_plan,
    prepare_v23_package,
    verify_radon_policy_plan,
)


def write(path: Path, value, *, mode=0o644) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n" if isinstance(value, dict) else value
    path.write_text(text); path.chmod(mode)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path: Path):
    active = tmp_path / "active.json"
    current = {
        "projects": {
            "LOOK": {"request_journals": ["/look-old.json"]},
            "Radon_Bridge": {"reserved_gpus": 9, "request_journals": ["/rb-old.json"]},
            "Uncertainty_Lab": {"reserved_gpus": 2, "request_journals": [f"/ul-{i}.json" for i in range(15)]},
        }
    }
    initial = write(active, current)
    claim = tmp_path / "look-claim.json"
    claim_sha = write(claim, {"job_id": "52429877", "state": "submitted"})
    look_journal = tmp_path / "look-journal.json"
    write(look_journal, {"requests": [{"job_id": "52429877", "v5_claim_sha256": claim_sha}]})
    journal = tmp_path / "radon-requests.json"
    journal_sha = write(journal, {"schema": "radon_paused_sixth_v5_requests_v2", "requests": []})
    candidate_value = copy.deepcopy(current)
    candidate_value["projects"]["Radon_Bridge"]["reserved_gpus"] = 10
    candidate_value["projects"]["Radon_Bridge"]["request_journals"].append(str(journal.resolve()))
    candidate = tmp_path / "candidate.json"; candidate_sha = write(candidate, candidate_value)
    rollback_value = copy.deepcopy(candidate_value)
    rollback_value["projects"]["Radon_Bridge"]["reserved_gpus"] = 9
    rollback = tmp_path / "rollback.json"; rollback_sha = write(rollback, rollback_value)
    snapshot = tmp_path / "capacity.json"
    snapshot_sha = write(snapshot, {
        "schema": "radon_v5_live_capacity_snapshot_v1",
        "policy_sha256": initial, "account_ceiling": 24,
        "account_running_pending": 22,
        "role_live_gpus": {
            "LOOK": 7, "Radon_Bridge": 9, "Uncertainty_Lab": 2,
            "legacy_native_unregistered": 4,
        },
        "unresolved_gpu_intents": {"LOOK": 0, "Radon_Bridge": 0, "Uncertainty_Lab": 0},
        "test_access": False,
    })
    slot = tmp_path / "slot.json"
    slot_sha = write(slot, {
        "schema": "radon_paused_sixth_v5_slot_transition_contract_v1",
        "state": "prepared_read_only_no_production_mutation",
        "transition": "temporary_radon_role_9_to_10_for_one_formal_v5_request",
        "production_mutated": False, "gpu_requested": False, "test_access": False,
        "initial_role_policy_sha256": initial,
        "proposed_role_policy_sha256": candidate_sha,
        "rollback_role_policy_sha256": rollback_sha,
        "live_capacity_snapshot": str(snapshot.resolve()),
        "live_capacity_snapshot_sha256": snapshot_sha,
        "v4_preemption": {"authorized_by_this_contract": False},
        "legacy_dispatcher": {"must_remain_unchanged_until_coupled_handover": True},
        "rollback_rules": [
            "install rollback only after the current LOOK successor accepts the rollback SHA or a new held-first LOOK monitor publishes matching stage2 acceptance"
        ],
        "capacity_at_snapshot": {
            "account_ceiling": 24, "account_running_pending": 22,
            "post_request_upper_bound": 23, "remaining_account_slots": 1,
            "radon_initial_cap": 9, "radon_proposed_cap": 10,
            "radon_running_pending": 9,
        },
    })
    archive = tmp_path / "runner.tar"; write(archive, "runner")
    acceptance = tmp_path / "runner-acceptance.json"
    write(acceptance, {
        "schema": "radon_paused_sixth_v5_control_candidate_v3",
        "candidate": {"implementation_commit": RUNNER_COMMIT},
        "ibex_validation": {"full_cpu": {"state": "COMPLETED", "exit_code": "0:0"}},
        "production": {"policy_mutated": False},
        "test_access": False,
    })
    review = tmp_path / "independent-review.json"
    write(review, {
        "schema": "radon_paused_sixth_v5_v3_independent_code_review_v1",
        "candidate": {"implementation_commit": RUNNER_COMMIT},
        "code_verdict": "GO",
        "production_verdict": "NO_GO", "test_access": False,
        "production_policy_mutated": False,
    })
    bundle_review = tmp_path / "bundle-review.json"
    proposal = tmp_path / "proposal.json"
    write(proposal, {
        "schema": "radon_paused_sixth_v5_policy_proposal_v1", "state": "accepted",
        "previous_role_policy_sha256": initial,
        "role_policy": str(candidate.resolve()), "role_policy_sha256": candidate_sha,
        "journal": str(journal.resolve()),
        "journal_initial_sha256": journal_sha, "runner_commit": RUNNER_COMMIT,
        "runner_archive": str(archive.resolve()),
        "runner_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "temporary_reserved_gpus": 10, "steady_state_reserved_gpus": 9,
        "production_dispatch_authorized": False, "production_policy_installed": False,
        "rollback_role_policy": str(rollback.resolve()),
        "rollback_role_policy_sha256": rollback_sha,
        "slot_transition_contract": str(slot.resolve()),
        "slot_transition_contract_sha256": slot_sha,
        "live_capacity_snapshot": str(snapshot.resolve()),
        "live_capacity_snapshot_sha256": snapshot_sha,
        "test_access": False,
    })
    write(bundle_review, {
        "schema": "radon_paused_sixth_v5_cap10_bundle_independent_review_v1",
        "conclusion": "GO_FOR_LOOK_V23_PREBINDING_ONLY_PRODUCTION_NO_GO",
        "reviewed_bundle": {
            "bundle_commit": "eb11632f24c5e58cb8ca08570b25bf6e255b2302",
            "runner_source_commit": RUNNER_COMMIT,
            "policy_proposal_sha256": hashlib.sha256(proposal.read_bytes()).hexdigest(),
            "slot_transition_contract_sha256": slot_sha,
            "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "journal_initial_sha256": journal_sha,
        },
        "verdict": {
            "look_v23_prebinding": "GO", "production_policy_install": "NO_GO",
            "radon_gpu_submission": "NO_GO", "rollback_policy_install": "NO_GO",
            "source_claim_reservation": "NO_GO",
        },
        "look_policy_compatibility": {
            "v23_prebinding_allowed_policy_sha256s": [initial, candidate_sha],
            "v23_accepts_rollback_policy": False,
            "rollback_requires_new_held_first_look_successor_stage2": True,
        },
    })
    contract = tmp_path / "owner.json"
    write(contract, build_radon_owner_contract(
        journal=journal, candidate_policy=candidate, runner_archive=archive,
        runner_acceptance=acceptance, runner_independent_review=review,
        bundle_independent_review=bundle_review,
        policy_proposal=proposal,
    ))
    plan = tmp_path / "plan.json"
    write(plan, build_radon_policy_plan(
        active_policy=active, look_gpu_job_id="52429877", look_claim=claim,
        look_journal=look_journal, candidate_policy=candidate,
        appended_journal=journal, owner_contract=contract,
    ))
    return active, initial, candidate, candidate_sha, rollback_sha, plan


def test_plan_accepts_exact_single_radon_append_and_preserves_ul(tmp_path):
    active, initial, _, candidate_sha, rollback_sha, plan = fixture(tmp_path)
    verified = verify_radon_policy_plan(
        plan, expected_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
        expected_active_policy=active, expected_initial_policy_sha256=initial,
    )
    assert verified["allowed_policy_sha256"] == [initial, candidate_sha]
    before = json.loads(active.read_text())
    after = json.loads(Path(verified["plan"]["candidate_policy"]).read_text())
    assert before["projects"]["Uncertainty_Lab"] == after["projects"]["Uncertainty_Lab"]
    assert before["projects"]["LOOK"] == after["projects"]["LOOK"]


def test_plan_rejects_any_nonjournal_policy_change(tmp_path):
    active, initial, candidate, _, _, plan = fixture(tmp_path)
    plan_value = json.loads(plan.read_text())
    candidate_value = json.loads(candidate.read_text()); candidate_value["projects"]["LOOK"]["reserved_gpus"] = 8
    plan_value["candidate_policy_sha256"] = write(candidate, candidate_value)
    write(plan, plan_value)
    with pytest.raises(ValueError, match="not the exact temporary R&B cap-10"):
        verify_radon_policy_plan(
            plan, expected_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
            expected_active_policy=active, expected_initial_policy_sha256=initial,
        )


def test_v23_package_drops_old_transition_states_and_keeps_only_exact_pair(tmp_path):
    active, initial, _, candidate_sha, rollback_sha, plan = fixture(tmp_path)
    old = tmp_path / "v22"; new = tmp_path / "v23"
    old.mkdir(); (old / "source").mkdir(); write(old / "source/helper.py", "x=1\n")
    old_allowed = ["a" * 64, "b" * 64, initial]
    old_allow = "{" + ",".join(repr(value) for value in old_allowed) + "}"
    chain_sha = write(old / "chain_manager.py", f"PKG={str(old)!r}\nALLOW={old_allow}\n")
    monitor_sha = write(old / "monitor.sh", f"#!/bin/bash\npython {old}/chain_manager.py\n", mode=0o555)
    for name in ("stage_manager.py", "stage_monitor.sh", "run_gpu.sbatch", "run_pca.sbatch", "run_decision.sbatch"):
        write(old / name, f"ROOT={str(old)!r}\n")
    package = prepare_v23_package(
        v22_package=old, v23_package=new, v22_chain_sha256=chain_sha,
        v22_monitor_sha256=monitor_sha, expected_v22_allowed_policy_sha256=old_allowed,
        plan_path=plan, plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
        expected_active_policy=active, expected_initial_policy_sha256=initial,
    )
    assert package["allowed_policy_sha256"] == [initial, candidate_sha]
    assert str(old) not in (new / "chain_manager.py").read_text()
