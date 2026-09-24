"""Exact one-state LOOK v23 successor for the reviewed R&B V5 proposal."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from look.runtime.v5_monitor_handover import run_handover
from look.runtime.v5_monitor_ibex import (
    PRODUCTION_ACCOUNT,
    PRODUCTION_LOCK,
    PRODUCTION_POLICY,
    PRODUCTION_USER,
    SlurmAdapter,
    build_handover_spec,
    read_json,
    sha256,
)
from look.runtime.v5_role_policy_transition import (
    copy_immutable_tree,
    replace_exact_path,
    write_exclusive_bytes,
    write_exclusive_json,
)


INITIAL_POLICY_SHA256 = "0c90a1619bd165eb32fb753678b185793daf8ffc1c4da70632833cb7da4d1289"
RUNNER_COMMIT = "44ba496fd9d2aad065e141d8485649a0dba3eec0"
PLAN_FIELDS = {
    "schema", "active_policy", "initial_policy_sha256", "look_gpu_job_id",
    "look_claim", "look_claim_sha256", "look_journal", "look_journal_sha256",
    "candidate_policy", "candidate_policy_sha256", "appended_journal",
    "appended_journal_initial_sha256", "owner_contract", "owner_contract_sha256",
    "test_access",
}
OWNER_FIELDS = {
    "schema", "project", "journal", "journal_initial_sha256", "candidate_policy",
    "candidate_policy_sha256", "runner_commit", "runner_archive",
    "runner_archive_sha256", "runner_acceptance", "runner_acceptance_sha256",
    "runner_independent_review", "runner_independent_review_sha256",
    "policy_proposal", "policy_proposal_sha256", "test_access",
}


def _require_sha(path: str | Path, expected: str, label: str) -> Path:
    value = Path(path).resolve()
    if not value.is_file() or sha256(value) != expected:
        raise ValueError(f"{label} identity changed")
    return value


def build_radon_owner_contract(
    *,
    journal: str | Path,
    candidate_policy: str | Path,
    runner_archive: str | Path,
    runner_acceptance: str | Path,
    runner_independent_review: str | Path,
    policy_proposal: str | Path,
) -> dict[str, Any]:
    values = {
        "journal": Path(journal).resolve(),
        "candidate_policy": Path(candidate_policy).resolve(),
        "runner_archive": Path(runner_archive).resolve(),
        "runner_acceptance": Path(runner_acceptance).resolve(),
        "runner_independent_review": Path(runner_independent_review).resolve(),
        "policy_proposal": Path(policy_proposal).resolve(),
    }
    for path in values.values():
        if not path.is_file():
            raise ValueError(f"R&B prebinding artifact is missing: {path}")
    return {
        "schema": "radon_bridge_v5_monitor_prebinding_v1",
        "project": "Radon_Bridge",
        "journal": str(values["journal"]),
        "journal_initial_sha256": sha256(values["journal"]),
        "candidate_policy": str(values["candidate_policy"]),
        "candidate_policy_sha256": sha256(values["candidate_policy"]),
        "runner_commit": RUNNER_COMMIT,
        "runner_archive": str(values["runner_archive"]),
        "runner_archive_sha256": sha256(values["runner_archive"]),
        "runner_acceptance": str(values["runner_acceptance"]),
        "runner_acceptance_sha256": sha256(values["runner_acceptance"]),
        "runner_independent_review": str(values["runner_independent_review"]),
        "runner_independent_review_sha256": sha256(values["runner_independent_review"]),
        "policy_proposal": str(values["policy_proposal"]),
        "policy_proposal_sha256": sha256(values["policy_proposal"]),
        "test_access": False,
    }


def build_radon_policy_plan(
    *,
    active_policy: str | Path,
    look_gpu_job_id: str,
    look_claim: str | Path,
    look_journal: str | Path,
    candidate_policy: str | Path,
    appended_journal: str | Path,
    owner_contract: str | Path,
) -> dict[str, Any]:
    values = {
        "active_policy": Path(active_policy).resolve(),
        "look_claim": Path(look_claim).resolve(),
        "look_journal": Path(look_journal).resolve(),
        "candidate_policy": Path(candidate_policy).resolve(),
        "appended_journal": Path(appended_journal).resolve(),
        "owner_contract": Path(owner_contract).resolve(),
    }
    for path in values.values():
        if not path.is_file():
            raise ValueError(f"R&B policy-plan artifact is missing: {path}")
    return {
        "schema": "look_v5_monitor_radon_policy_plan_v1",
        "active_policy": str(values["active_policy"]),
        "initial_policy_sha256": sha256(values["active_policy"]),
        "look_gpu_job_id": str(look_gpu_job_id),
        "look_claim": str(values["look_claim"]),
        "look_claim_sha256": sha256(values["look_claim"]),
        "look_journal": str(values["look_journal"]),
        "look_journal_sha256": sha256(values["look_journal"]),
        "candidate_policy": str(values["candidate_policy"]),
        "candidate_policy_sha256": sha256(values["candidate_policy"]),
        "appended_journal": str(values["appended_journal"]),
        "appended_journal_initial_sha256": sha256(values["appended_journal"]),
        "owner_contract": str(values["owner_contract"]),
        "owner_contract_sha256": sha256(values["owner_contract"]),
        "test_access": False,
    }


def verify_radon_policy_plan(
    plan_path: str | Path,
    *,
    expected_sha256: str,
    expected_active_policy: str | Path = PRODUCTION_POLICY,
    expected_look_gpu_job_id: str = "52429877",
    expected_initial_policy_sha256: str = INITIAL_POLICY_SHA256,
) -> dict[str, Any]:
    path = _require_sha(plan_path, expected_sha256, "R&B policy plan")
    plan = read_json(path)
    if set(plan) != PLAN_FIELDS or plan.get("schema") != "look_v5_monitor_radon_policy_plan_v1":
        raise ValueError("Unknown R&B monitor policy plan")
    if plan.get("test_access") is not False or str(plan.get("look_gpu_job_id")) != str(expected_look_gpu_job_id):
        raise ValueError("R&B plan changed the LOOK execution boundary")
    active = Path(plan["active_policy"]).resolve()
    if (
        active != Path(expected_active_policy).resolve()
        or plan["initial_policy_sha256"] != expected_initial_policy_sha256
        or sha256(active) != expected_initial_policy_sha256
    ):
        raise ValueError("R&B plan does not start at the exact accepted UL policy")
    claim = _require_sha(plan["look_claim"], plan["look_claim_sha256"], "LOOK claim")
    look_journal = _require_sha(plan["look_journal"], plan["look_journal_sha256"], "LOOK journal")
    claim_value, look_value = read_json(claim), read_json(look_journal)
    if str(claim_value.get("job_id")) != str(expected_look_gpu_job_id) or claim_value.get("state") != "submitted":
        raise ValueError("LOOK claim no longer binds the preserved GPU job")
    rows = [row for row in look_value.get("requests", []) if str(row.get("job_id")) == str(expected_look_gpu_job_id)]
    if len(rows) != 1 or rows[0].get("v5_claim_sha256") != plan["look_claim_sha256"]:
        raise ValueError("LOOK journal no longer uniquely binds the preserved job")

    candidate_path = _require_sha(
        plan["candidate_policy"], plan["candidate_policy_sha256"], "R&B candidate policy"
    )
    journal = _require_sha(
        plan["appended_journal"],
        plan["appended_journal_initial_sha256"],
        "R&B initial journal",
    )
    before, candidate = read_json(active), read_json(candidate_path)
    expected = copy.deepcopy(before)
    journals = expected["projects"]["Radon_Bridge"]["request_journals"]
    if str(journal) in journals:
        raise ValueError("R&B journal is already registered")
    journals.append(str(journal))
    if candidate != expected:
        raise ValueError("Candidate policy changes more than one R&B journal append")
    journal_value = read_json(journal)
    if (
        journal_value.get("schema") != "radon_paused_sixth_v5_requests_v2"
        or journal_value.get("requests") != []
    ):
        raise ValueError("R&B journal is not the exact empty v3 request journal")

    contract_path = _require_sha(
        plan["owner_contract"], plan["owner_contract_sha256"], "R&B prebinding contract"
    )
    contract = read_json(contract_path)
    if (
        set(contract) != OWNER_FIELDS
        or contract.get("schema") != "radon_bridge_v5_monitor_prebinding_v1"
        or contract.get("project") != "Radon_Bridge"
        or contract.get("runner_commit") != RUNNER_COMMIT
        or contract.get("test_access") is not False
        or Path(contract.get("journal", "")).resolve() != journal
        or contract.get("journal_initial_sha256") != plan["appended_journal_initial_sha256"]
        or Path(contract.get("candidate_policy", "")).resolve() != candidate_path
        or contract.get("candidate_policy_sha256") != plan["candidate_policy_sha256"]
    ):
        raise ValueError("R&B prebinding contract changed the reviewed proposal")
    _require_sha(contract["runner_archive"], contract["runner_archive_sha256"], "R&B runner archive")
    acceptance = read_json(_require_sha(
        contract["runner_acceptance"], contract["runner_acceptance_sha256"], "R&B runner acceptance"
    ))
    if (
        acceptance.get("schema") != "radon_paused_sixth_v5_control_candidate_v3"
        or acceptance.get("candidate", {}).get("implementation_commit") != RUNNER_COMMIT
        or acceptance.get("ibex_validation", {}).get("full_cpu", {}).get("state") != "COMPLETED"
        or acceptance.get("ibex_validation", {}).get("full_cpu", {}).get("exit_code") != "0:0"
        or acceptance.get("production", {}).get("policy_mutated") is not False
        or acceptance.get("test_access") is not False
    ):
        raise ValueError("R&B v3 runner acceptance is not exact")
    review = read_json(_require_sha(
        contract["runner_independent_review"],
        contract["runner_independent_review_sha256"],
        "R&B independent code review",
    ))
    if (
        review.get("schema") != "radon_paused_sixth_v5_v3_independent_code_review_v1"
        or review.get("candidate", {}).get("implementation_commit") != RUNNER_COMMIT
        or review.get("code_verdict") != "GO"
        or review.get("production_verdict") != "NO_GO"
        or review.get("test_access") is not False
        or review.get("production_policy_mutated") is not False
    ):
        raise ValueError("R&B independent review does not preserve the production gate")
    proposal = read_json(_require_sha(
        contract["policy_proposal"], contract["policy_proposal_sha256"], "R&B policy proposal"
    ))
    if (
        proposal.get("schema") != "radon_paused_sixth_v5_policy_proposal_v1"
        or proposal.get("state") != "accepted"
        or proposal.get("role_policy_sha256") != plan["candidate_policy_sha256"]
        or proposal.get("journal") != str(journal)
        or proposal.get("journal_initial_sha256") != plan["appended_journal_initial_sha256"]
        or proposal.get("runner_commit") != RUNNER_COMMIT
        or proposal.get("test_access") is not False
    ):
        raise ValueError("R&B policy proposal is not independently accepted")
    return {
        "plan": plan,
        "plan_path": path,
        "allowed_policy_sha256": [expected_initial_policy_sha256, plan["candidate_policy_sha256"]],
    }


def prepare_v23_package(
    *,
    v22_package: str | Path,
    v23_package: str | Path,
    v22_chain_sha256: str,
    v22_monitor_sha256: str,
    expected_v22_allowed_policy_sha256: list[str],
    plan_path: str | Path,
    plan_sha256: str,
    expected_active_policy: str | Path = PRODUCTION_POLICY,
    expected_look_gpu_job_id: str = "52429877",
    expected_initial_policy_sha256: str = INITIAL_POLICY_SHA256,
) -> dict[str, Any]:
    verified = verify_radon_policy_plan(
        plan_path,
        expected_sha256=plan_sha256,
        expected_active_policy=expected_active_policy,
        expected_look_gpu_job_id=expected_look_gpu_job_id,
        expected_initial_policy_sha256=expected_initial_policy_sha256,
    )
    old, new = Path(v22_package).resolve(), Path(v23_package).resolve()
    if new.exists():
        raise FileExistsError(new)
    _require_sha(old / "chain_manager.py", v22_chain_sha256, "v22 chain manager")
    _require_sha(old / "monitor.sh", v22_monitor_sha256, "v22 monitor")
    new.mkdir(parents=True)
    source_hashes = copy_immutable_tree(old / "source", new / "source")
    old_allow = "{" + ",".join(repr(value) for value in expected_v22_allowed_policy_sha256) + "}"
    new_allow = "{" + ",".join(repr(value) for value in verified["allowed_policy_sha256"]) + "}"
    transformed: dict[str, str] = {}
    chain = replace_exact_path((old / "chain_manager.py").read_text(), old, new, label="chain manager")
    if chain.count(old_allow) != 1:
        raise ValueError("v22 policy allowlist anchor changed")
    write_exclusive_bytes(new / "chain_manager.py", chain.replace(old_allow, new_allow).encode(), mode=0o444)
    transformed["chain_manager.py"] = sha256(new / "chain_manager.py")
    for name in (
        "stage_manager.py", "monitor.sh", "stage_monitor.sh", "run_gpu.sbatch",
        "run_pca.sbatch", "run_decision.sbatch",
    ):
        source = old / name
        text = replace_exact_path(source.read_text(), old, new, label=name)
        write_exclusive_bytes(new / name, text.encode(), mode=source.stat().st_mode & 0o777)
        transformed[name] = sha256(new / name)
    for target in new.rglob("*"):
        if target.is_file() and str(old) in target.read_text(errors="ignore"):
            raise ValueError(f"v23 package retains a v22 path: {target.relative_to(new)}")
    manifest = {
        "schema": "look_v5_v23_monitor_package_manifest_v1",
        "v22_package": str(old),
        "v22_chain_sha256": v22_chain_sha256,
        "v22_monitor_sha256": v22_monitor_sha256,
        "v23_package": str(new),
        "policy_plan": str(Path(plan_path).resolve()),
        "policy_plan_sha256": plan_sha256,
        "allowed_policy_sha256": verified["allowed_policy_sha256"],
        "transformed_files": transformed,
        "source_files": source_hashes,
        "test_access": False,
    }
    write_exclusive_json(new / "package_manifest.json", manifest)
    return {**manifest, "package_manifest_sha256": sha256(new / "package_manifest.json")}


class RadonPlanMonitorInstaller:
    REQUIRED = {
        "schema", "policy_plan", "policy_plan_sha256", "package_manifest",
        "package_manifest_sha256", "replacement_command", "replacement_source_sha256",
        "gpu_job_id", "test_access",
    }

    def __init__(
        self,
        binding_path: str | Path,
        *,
        expected_sha256: str,
        expected_active_policy: str | Path = PRODUCTION_POLICY,
        expected_look_gpu_job_id: str = "52429877",
        expected_initial_policy_sha256: str = INITIAL_POLICY_SHA256,
    ) -> None:
        self.path = _require_sha(binding_path, expected_sha256, "v23 binding")
        self.binding = read_json(self.path)
        if (
            set(self.binding) != self.REQUIRED
            or self.binding.get("schema") != "look_v5_monitor_radon_successor_binding_v1"
            or self.binding.get("test_access") is not False
            or self.binding.get("gpu_job_id") != str(expected_look_gpu_job_id)
        ):
            raise ValueError("Unknown v23 monitor binding")
        self.context = {
            "expected_active_policy": expected_active_policy,
            "expected_look_gpu_job_id": str(expected_look_gpu_job_id),
            "expected_initial_policy_sha256": expected_initial_policy_sha256,
        }

    def _plan(self) -> dict[str, Any]:
        return verify_radon_policy_plan(
            self.binding["policy_plan"],
            expected_sha256=self.binding["policy_plan_sha256"],
            **self.context,
        )

    def verify(self, spec: dict[str, Any]) -> dict[str, str]:
        verified = self._plan()
        manifest = read_json(_require_sha(
            self.binding["package_manifest"],
            self.binding["package_manifest_sha256"],
            "v23 package manifest",
        ))
        replacement = _require_sha(
            self.binding["replacement_command"],
            self.binding["replacement_source_sha256"],
            "v23 replacement monitor",
        )
        if str(replacement) != str(Path(spec["replacement_command"]).resolve()):
            raise ValueError("V23 replacement command changed")
        if manifest.get("allowed_policy_sha256") != verified["allowed_policy_sha256"]:
            raise ValueError("V23 package allowlist differs from the R&B plan")
        return {
            "policy_sha256": verified["allowed_policy_sha256"][0],
            "journal_sha256": verified["plan"]["appended_journal_initial_sha256"],
            "claim_sha256": verified["plan"]["look_claim_sha256"],
            "provenance_sha256": self.binding["package_manifest_sha256"],
        }

    preflight = verify

    def commit(self, spec: dict[str, Any], _: str) -> dict[str, str]:
        return self.verify(spec)

    def look_claim(self) -> Path:
        return Path(self._plan()["plan"]["look_claim"]).resolve()


def build_radon_successor_binding(
    *,
    plan_path: str | Path,
    package_manifest: str | Path,
    replacement_command: str | Path,
    gpu_job_id: str,
) -> dict[str, Any]:
    plan = Path(plan_path).resolve()
    manifest = Path(package_manifest).resolve()
    replacement = Path(replacement_command).resolve()
    for path in (plan, manifest, replacement):
        if not path.is_file():
            raise ValueError(f"Missing v23 binding artifact: {path}")
    return {
        "schema": "look_v5_monitor_radon_successor_binding_v1",
        "policy_plan": str(plan),
        "policy_plan_sha256": sha256(plan),
        "package_manifest": str(manifest),
        "package_manifest_sha256": sha256(manifest),
        "replacement_command": str(replacement),
        "replacement_source_sha256": sha256(replacement),
        "gpu_job_id": str(gpu_job_id),
        "test_access": False,
    }


def build_v23_spec(
    *,
    old_job: dict[str, Any],
    gpu_job: dict[str, Any],
    replacement_command: str | Path,
    binding_path: str | Path,
) -> dict[str, Any]:
    return build_handover_spec(
        old_job=old_job,
        gpu_job=gpu_job,
        replacement_command=replacement_command,
        binding_path=binding_path,
        shared_lock=PRODUCTION_LOCK,
        mode="production_v23",
    )


def apply_v23_successor(
    *,
    spec_path: str | Path,
    binding_path: str | Path,
    receipt_dir: str | Path,
    allow_production: bool,
) -> dict[str, Any]:
    if not allow_production:
        raise PermissionError("Explicit production apply flag is required")
    spec = read_json(spec_path)
    if spec.get("mode") != "production_v23" or Path(spec["shared_lock"]).resolve() != PRODUCTION_LOCK:
        raise ValueError("V23 production spec changed mode or lock")
    if spec.get("expected_user") != PRODUCTION_USER or spec.get("expected_account") != PRODUCTION_ACCOUNT:
        raise ValueError("V23 production Slurm identity changed")
    installer = RadonPlanMonitorInstaller(binding_path, expected_sha256=spec["binding_sha256"])
    slurm = SlurmAdapter(user=PRODUCTION_USER, account=PRODUCTION_ACCOUNT)
    return run_handover(
        receipt_dir,
        lock_path=PRODUCTION_LOCK,
        spec=spec,
        observe=slurm.observe,
        find_held=slurm.find_held,
        submit_held=lambda value: slurm.submit_held(value, installer.look_claim(), receipt_dir),
        cancel=slurm.cancel,
        preflight_artifacts=installer.preflight,
        commit_artifacts=installer.commit,
        release=slurm.release,
        account_gpu_count=slurm.account_gpu_count,
    )
