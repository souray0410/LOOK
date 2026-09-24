import json

import pytest

from look.runtime.v5_monitor_handover import PHASES, run_handover, stage2_action, validate_spec


def spec(tmp_path):
    return {
        "schema": "look_v5_monitor_handover_spec_v2",
        "mode": "production",
        "transaction_id": "look-v5-v21-stage1",
        "old_job_id": "52430491",
        "gpu_job_id": "52429877",
        "dependency": "afterany:52429877(unfulfilled)",
        "old_command": "/immutable/v20/monitor.sh",
        "gpu_command": "/immutable/v13/run_gpu.sbatch",
        "replacement_command": "/immutable/v21/monitor.sh",
        "replacement_source_sha256": "a" * 64,
        "binding_sha256": "b" * 64,
        "expected_user": "mengh",
        "expected_account": "pi-mengy",
        "expected_monitor_req_tres": "cpu=2,mem=4G,node=1,billing=2",
        "expected_gpu_req_tres": "cpu=16,mem=128G,node=1,billing=16,gres/gpu=1,gres/gpu:a100=1",
        "shared_lock": str((tmp_path / "account.lock").resolve()),
        "max_account_gpus": 24,
        "test_access": False,
    }


def harness(tmp_path):
    value = spec(tmp_path)
    (tmp_path / "account.lock").touch()
    old = {
        "job_id": "52430491",
        "state": "PENDING",
        "reason": "Dependency",
        "dependency": value["dependency"],
        "user": "mengh",
        "account": "pi-mengy",
        "req_tres": value["expected_monitor_req_tres"],
        "command": value["old_command"],
        "has_step": False,
    }
    gpu = {
        "job_id": "52429877",
        "state": "PENDING",
        "reason": "Priority",
        "user": "mengh",
        "account": "pi-mengy",
        "req_tres": value["expected_gpu_req_tres"],
        "command": value["gpu_command"],
        "has_step": False,
    }
    new = {
        "job_id": "600",
        "state": "PENDING",
        "reason": "JobHeldUser",
        "dependency": value["dependency"],
        "user": "mengh",
        "account": "pi-mengy",
        "req_tres": value["expected_monitor_req_tres"],
        "command": value["replacement_command"],
        "has_step": False,
    }
    states = {"52430491": old, "52429877": gpu, "600": new}
    events = []
    evidence = {
        "policy_sha256": "1" * 64,
        "journal_sha256": "2" * 64,
        "claim_sha256": "3" * 64,
        "provenance_sha256": "4" * 64,
    }

    def observe(job):
        return states[job]

    def submit(_):
        events.append("submit-held")
        return "600"

    def cancel(job):
        events.append("cancel-old")
        states[job] = {"job_id": job, "state": "CANCELLED", "has_step": False}

    def commit(_, job):
        events.append(f"commit:{job}")
        return evidence

    def release(job):
        events.append("release")
        states[job]["reason"] = "Dependency"

    kwargs = {
        "receipt_dir": tmp_path / "receipts",
        "lock_path": tmp_path / "account.lock",
        "spec": value,
        "observe": observe,
        "find_held": lambda _: [],
        "submit_held": submit,
        "cancel": cancel,
        "preflight_artifacts": lambda _: evidence,
        "commit_artifacts": commit,
        "release": release,
        "account_gpu_count": lambda: 23,
    }
    return kwargs, states, events


def test_single_lock_handover_is_durable_and_idempotent(tmp_path):
    kwargs, _, events = harness(tmp_path)
    result = run_handover(**kwargs)
    assert result["phase"] == "release_verified"
    assert events == ["submit-held", "cancel-old", "commit:600", "release"]
    assert [path.name for path in sorted((tmp_path / "receipts").iterdir())] == [
        f"{index:02d}-{phase}.json" for index, phase in enumerate(PHASES)
    ]
    assert run_handover(**kwargs)["phase"] == "release_verified"
    assert events == ["submit-held", "cancel-old", "commit:600", "release"]


def test_v23_production_identity_is_exact(tmp_path):
    value = spec(tmp_path)
    value.update(
        mode="production_v23",
        old_job_id="52462712",
        old_command="/immutable/v22/monitor.sh",
        replacement_command="/immutable/v23/monitor.sh",
    )
    validate_spec(value)
    value["old_job_id"] = "52455266"
    with pytest.raises(ValueError, match="V23 production chain identity"):
        validate_spec(value)


def test_recovery_discovers_one_held_job_without_duplicate_submit(tmp_path):
    kwargs, states, events = harness(tmp_path)
    kwargs["find_held"] = lambda _: ["600"]
    assert run_handover(**kwargs)["phase"] == "release_verified"
    assert "submit-held" not in events
    receipt = json.loads((tmp_path / "receipts/01-held_submitted.json").read_text())
    assert receipt["recovered_existing"] is True
    assert states["52430491"]["state"] == "CANCELLED"


def test_release_before_ack_is_reconciled_without_second_release(tmp_path):
    kwargs, states, events = harness(tmp_path)
    original_release = kwargs["release"]

    def crash_after_release(job):
        original_release(job)
        raise RuntimeError("lost release acknowledgement")

    kwargs["release"] = crash_after_release
    with pytest.raises(RuntimeError, match="lost release"):
        run_handover(**kwargs)
    assert (tmp_path / "receipts/06-release_requested.json").is_file()
    kwargs["release"] = lambda job: events.append("unexpected-second-release")
    assert run_handover(**kwargs)["phase"] == "release_verified"
    assert "unexpected-second-release" not in events
    assert states["600"]["reason"] == "Dependency"


def test_release_before_ack_reconciles_fast_successful_completion(tmp_path):
    kwargs, states, events = harness(tmp_path)

    def complete_then_drop_ack(job):
        states[job].update(
            state="COMPLETED",
            reason="None",
            dependency="(null)",
            exit_code="0:0",
            has_step=False,
        )
        raise RuntimeError("lost release acknowledgement")

    kwargs["release"] = complete_then_drop_ack
    with pytest.raises(RuntimeError, match="lost release"):
        run_handover(**kwargs)
    kwargs["release"] = lambda _: events.append("unexpected-second-release")
    assert run_handover(**kwargs)["phase"] == "release_verified"
    assert "unexpected-second-release" not in events


def test_fast_completion_must_match_success_and_original_held_identity(tmp_path):
    kwargs, states, _ = harness(tmp_path)

    def complete_then_drop_ack(job):
        states[job].update(
            state="COMPLETED",
            reason="None",
            dependency="(null)",
            exit_code="1:0",
            has_step=False,
        )
        raise RuntimeError("lost release acknowledgement")

    kwargs["release"] = complete_then_drop_ack
    with pytest.raises(RuntimeError, match="lost release"):
        run_handover(**kwargs)
    with pytest.raises(ValueError, match="successful Slurm evidence"):
        run_handover(**kwargs)


def test_policy_commit_before_ack_is_idempotently_recovered(tmp_path):
    kwargs, _, events = harness(tmp_path)
    committed = {"done": False}
    evidence = {
        "policy_sha256": "1" * 64,
        "journal_sha256": "2" * 64,
        "claim_sha256": "3" * 64,
        "provenance_sha256": "4" * 64,
    }

    def commit(_, __):
        if not committed["done"]:
            committed["done"] = True
            raise RuntimeError("lost policy-install acknowledgement")
        events.append("reconciled-policy")
        return evidence

    kwargs["commit_artifacts"] = commit
    with pytest.raises(RuntimeError, match="lost policy"):
        run_handover(**kwargs)
    assert run_handover(**kwargs)["phase"] == "release_verified"
    assert events.count("reconciled-policy") == 1


def test_ambiguous_held_jobs_fail_before_old_cancel(tmp_path):
    kwargs, states, events = harness(tmp_path)
    kwargs["find_held"] = lambda _: ["600", "601"]
    with pytest.raises(ValueError, match="Multiple held"):
        run_handover(**kwargs)
    assert states["52430491"]["state"] == "PENDING" and events == []


def test_changed_gpu_identity_fails_before_submit(tmp_path):
    kwargs, states, events = harness(tmp_path)
    states["52429877"]["command"] = "/wrong"
    with pytest.raises(ValueError, match="GPU predecessor"):
        run_handover(**kwargs)
    assert events == []


def test_account_ceiling_is_recalculated_before_submit(tmp_path):
    kwargs, states, events = harness(tmp_path)
    kwargs["account_gpu_count"] = lambda: 25
    with pytest.raises(ValueError, match="approved ceiling"):
        run_handover(**kwargs)
    assert states["52430491"]["state"] == "PENDING" and events == []


def test_artifact_drift_fails_under_lock_before_submit_or_cancel(tmp_path):
    kwargs, states, events = harness(tmp_path)

    def reject(_):
        raise ValueError("active policy drift")

    kwargs["preflight_artifacts"] = reject
    with pytest.raises(ValueError, match="policy drift"):
        run_handover(**kwargs)
    assert states["52430491"]["state"] == "PENDING"
    assert events == []


def test_completed_old_monitor_is_not_treated_as_safe_cancellation(tmp_path):
    kwargs, states, events = harness(tmp_path)
    states["52430491"]["state"] = "COMPLETED"
    with pytest.raises(ValueError, match="exact replaceable"):
        run_handover(**kwargs)
    assert events == []


def test_stage2_requires_fresh_audit():
    assert stage2_action("RUNNING") == "wait_for_stage1_monitor_terminal"
    assert stage2_action("PENDING") == "replace_pending_monitor_with_exact_next_policy_sha"
    assert stage2_action("CANCELLED") == "fresh_ownership_audit_before_next_policy_install"
