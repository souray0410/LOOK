import hashlib
import json
from pathlib import Path

import pytest

from look.runtime.v5_monitor_ibex import (
    ArtifactInstaller,
    SlurmAdapter,
    build_handover_spec,
    build_policy_binding,
)


def write(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, dict):
        path.write_text(json.dumps(value, sort_keys=True) + "\n")
    else:
        path.write_text(value)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def policy(journals=None):
    return {
        "schema": "research_gpu_roles_v1",
        "account_ceiling": 24,
        "native_max_gpus": 0,
        "projects": {
            "LOOK": {"reserved_gpus": 7, "request_journals": list(journals or ["/old.json"])},
            "Radon_Bridge": {"reserved_gpus": 9, "request_journals": ["/rb.json"]},
            "Uncertainty_Lab": {"reserved_gpus": 2, "request_journals": ["/liu-even.json", "/liu-odd.json"]},
        },
    }


def candidate(tmp_path):
    active = tmp_path / "active.json"
    base = tmp_path / "base.json"
    journal = tmp_path / "look_v5_requests.json"
    claim = tmp_path / "claim.json"
    provenance = tmp_path / "provenance.json"
    replacement = tmp_path / "v21/monitor.sh"
    write(active, policy()); write(base, policy())
    claim_sha = write(claim, {"job_id": "52429877", "test_access": False})
    write(journal, {"schema": "look_workflow_requests_v1", "requests": [
        {"job_id": "52429877", "v5_claim_sha256": claim_sha, "state": "submitted"}
    ]})
    write(provenance, {"schema": "v21", "test_access": False})
    write(replacement, "#!/bin/bash\nexit 0\n")
    proposed = tmp_path / "candidate.json"
    write(proposed, policy(["/old.json", str(journal.resolve())]))
    binding = build_policy_binding(
        active_policy=active, base_policy_snapshot=base, candidate_policy=proposed,
        dedicated_journal=journal, claim=claim, provenance=provenance,
        replacement_command=replacement, gpu_job_id="52429877",
    )
    binding_path = tmp_path / "binding.json"; binding_sha = write(binding_path, binding)
    spec = {
        "replacement_command": str(replacement.resolve()),
        "replacement_source_sha256": hashlib.sha256(replacement.read_bytes()).hexdigest(),
        "gpu_job_id": "52429877",
    }
    return active, binding_path, binding_sha, spec


def test_artifact_installer_changes_only_look_journal_and_recovers_idempotently(tmp_path):
    active, binding_path, binding_sha, spec = candidate(tmp_path)
    installer = ArtifactInstaller(binding_path, expected_sha256=binding_sha)
    evidence = installer.commit(spec, "600")
    installed = json.loads(active.read_text())
    assert installed["projects"]["Uncertainty_Lab"] == policy()["projects"]["Uncertainty_Lab"]
    assert len(installed["projects"]["LOOK"]["request_journals"]) == 2
    assert installer.commit(spec, "600") == evidence


def test_installer_rejects_policy_drift_before_write(tmp_path):
    active, binding_path, binding_sha, spec = candidate(tmp_path)
    write(active, {"unexpected": True})
    installer = ArtifactInstaller(binding_path, expected_sha256=binding_sha)
    with pytest.raises(ValueError, match="neither the bound base"):
        installer.commit(spec, "600")
    assert json.loads(active.read_text()) == {"unexpected": True}


def test_build_handover_spec_binds_exact_slurm_paths(tmp_path):
    _, binding_path, _, spec_stub = candidate(tmp_path)
    old = {
        "job_id": "52430491", "dependency": "afterany:52429877(unfulfilled)",
        "command": "/v20/monitor.sh", "user": "mengh", "account": "pi-mengy",
        "req_tres": "cpu=2,mem=4G,node=1,billing=2",
    }
    gpu = {
        "job_id": "52429877", "command": "/v13/run_gpu.sbatch", "user": "mengh",
        "account": "pi-mengy", "req_tres": "cpu=16,mem=128G,node=1,billing=16,gres/gpu=1,gres/gpu:a100=1",
    }
    result = build_handover_spec(
        old_job=old, gpu_job=gpu, replacement_command=spec_stub["replacement_command"],
        binding_path=binding_path, shared_lock=tmp_path / "lock",
    )
    assert result["old_job_id"] == "52430491" and result["gpu_job_id"] == "52429877"
    assert result["binding_sha256"] == hashlib.sha256(binding_path.read_bytes()).hexdigest()


class Result:
    def __init__(self, stdout="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, "", returncode


def test_slurm_adapter_parses_live_scontrol_and_counts_gpu_requests():
    timeouts = []

    def runner(args, **_):
        timeouts.append(_["timeout"])
        if args[:4] == ["scontrol", "show", "job", "-o"]:
            return Result(
                "JobId=52429877 JobName=look UserId=mengh(1) Account=pi-mengy "
                "JobState=PENDING Reason=Priority Dependency=(null) SubmitTime=t ExitCode=0:0 "
                "ReqTRES=cpu=16,mem=128G,node=1,billing=16,gres/gpu=1,gres/gpu:a100=1 "
                "Command=/v13/run_gpu.sbatch\n"
            )
        if args[0] == "sstat": return Result("", returncode=1)
        if args[0] == "squeue": return Result("1|gres/gpu:a100:1\n2|N/A\n")
        raise AssertionError(args)
    slurm = SlurmAdapter(user="mengh", account="pi-mengy", runner=runner)
    observed = slurm.observe("52429877")
    assert observed["state"] == "PENDING" and observed["has_step"] is False
    assert slurm.account_gpu_count() == 1
    assert timeouts and set(timeouts) == {20.0}
