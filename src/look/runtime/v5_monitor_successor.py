"""Exact-policy v22 successor for the pending LOOK V5 monitor.

This module does not install role-policy proposals.  It verifies an ordered
append-only policy plan, builds an immutable monitor package that accepts only
those exact policy digests, and replaces the still-pending v21 monitor under
the existing shared account lock.
"""
from __future__ import annotations

import copy
import json
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


PLAN_FIELDS = {
    "schema",
    "active_policy",
    "initial_policy_sha256",
    "look_gpu_job_id",
    "look_claim",
    "look_claim_sha256",
    "look_journal",
    "look_journal_sha256",
    "states",
    "test_access",
}
STATE_FIELDS = {
    "index",
    "project",
    "previous_policy_sha256",
    "candidate_policy",
    "candidate_policy_sha256",
    "appended_journal",
    "appended_journal_initial_sha256",
    "owner_contract",
    "owner_contract_sha256",
}
OWNER_FIELDS = {
    "schema",
    "project",
    "lane",
    "journal",
    "journal_initial_sha256",
    "packet",
    "packet_sha256",
    "source_commit",
    "source_archive",
    "source_archive_sha256",
    "predecessor_failed_job",
    "predecessor_terminal_evidence",
    "predecessor_terminal_evidence_sha256",
    "test_access",
}

TERMINAL_FIELDS = {
    "schema",
    "job_id",
    "state",
    "exit_code",
    "user",
    "account",
    "req_tres",
    "job_name",
    "test_access",
}


def _require_sha(path: str | Path, expected: str, label: str) -> Path:
    value = Path(path).resolve()
    if not value.is_file() or sha256(value) != expected:
        raise ValueError(f"{label} identity changed")
    return value


def build_failed_terminal_evidence(job: dict[str, Any]) -> dict[str, Any]:
    """Freeze the exact scheduler identity of one failed A100 allocation."""
    required = {"job_id", "state", "exit_code", "user", "account", "req_tres", "name"}
    if not required.issubset(job):
        raise ValueError("Terminal Slurm observation is incomplete")
    return {
        "schema": "ul_v5_failed_gpu_attempt_v1",
        "job_id": str(job["job_id"]),
        "state": job["state"],
        "exit_code": job["exit_code"],
        "user": job["user"],
        "account": job["account"],
        "req_tres": job["req_tres"],
        "job_name": job["name"],
        "test_access": False,
    }


def build_owner_contract(
    *,
    lane: str,
    journal: str | Path,
    packet: str | Path,
    source_commit: str,
    source_archive: str | Path,
    predecessor_failed_job: str,
    predecessor_terminal_evidence: str | Path,
) -> dict[str, Any]:
    journal_path, packet_path = Path(journal).resolve(), Path(packet).resolve()
    source_path = Path(source_archive).resolve()
    evidence_path = Path(predecessor_terminal_evidence).resolve()
    for path in (journal_path, packet_path, source_path, evidence_path):
        if not path.is_file():
            raise ValueError(f"Owner-contract artifact is missing: {path}")
    return {
        "schema": "ul_v5_retry_owner_contract_v1",
        "project": "Uncertainty_Lab",
        "lane": lane,
        "journal": str(journal_path),
        "journal_initial_sha256": sha256(journal_path),
        "packet": str(packet_path),
        "packet_sha256": sha256(packet_path),
        "source_commit": source_commit,
        "source_archive": str(source_path),
        "source_archive_sha256": sha256(source_path),
        "predecessor_failed_job": str(predecessor_failed_job),
        "predecessor_terminal_evidence": str(evidence_path),
        "predecessor_terminal_evidence_sha256": sha256(evidence_path),
        "test_access": False,
    }


def build_policy_plan(
    *,
    active_policy: str | Path,
    look_gpu_job_id: str,
    look_claim: str | Path,
    look_journal: str | Path,
    states: list[dict[str, str]],
) -> dict[str, Any]:
    active = Path(active_policy).resolve()
    claim, journal = Path(look_claim).resolve(), Path(look_journal).resolve()
    for path in (active, claim, journal):
        if not path.is_file():
            raise ValueError(f"Policy-plan artifact is missing: {path}")
    previous = sha256(active)
    result_states: list[dict[str, Any]] = []
    for index, item in enumerate(states, start=1):
        candidate = Path(item["candidate_policy"]).resolve()
        appended = Path(item["appended_journal"]).resolve()
        contract = Path(item["owner_contract"]).resolve()
        for path in (candidate, appended, contract):
            if not path.is_file():
                raise ValueError(f"Policy-plan state artifact is missing: {path}")
        result_states.append(
            {
                "index": index,
                "project": "Uncertainty_Lab",
                "previous_policy_sha256": previous,
                "candidate_policy": str(candidate),
                "candidate_policy_sha256": sha256(candidate),
                "appended_journal": str(appended),
                "appended_journal_initial_sha256": sha256(appended),
                "owner_contract": str(contract),
                "owner_contract_sha256": sha256(contract),
            }
        )
        previous = sha256(candidate)
    return {
        "schema": "look_v5_monitor_policy_plan_v1",
        "active_policy": str(active),
        "initial_policy_sha256": sha256(active),
        "look_gpu_job_id": str(look_gpu_job_id),
        "look_claim": str(claim),
        "look_claim_sha256": sha256(claim),
        "look_journal": str(journal),
        "look_journal_sha256": sha256(journal),
        "states": result_states,
        "test_access": False,
    }


def _verify_owner_contract(
    state: dict[str, Any],
    journal: Path,
    *,
    active_policy: Path,
    account_lock: Path,
    expected_user: str,
    expected_account: str,
    required_gres_token: str | None,
) -> None:
    contract_path = _require_sha(
        state["owner_contract"], state["owner_contract_sha256"], "UL owner contract"
    )
    contract = read_json(contract_path)
    if set(contract) != OWNER_FIELDS or contract.get("schema") != "ul_v5_retry_owner_contract_v1":
        raise ValueError("Unknown UL owner contract")
    if contract.get("project") != "Uncertainty_Lab" or contract.get("test_access") is not False:
        raise ValueError("UL owner contract changed project or test boundary")
    if str(Path(contract["journal"]).resolve()) != str(journal):
        raise ValueError("Owner contract journal differs from the policy plan")
    if contract["journal_initial_sha256"] != state["appended_journal_initial_sha256"]:
        raise ValueError("Owner contract journal digest differs from the policy plan")
    packet_path = _require_sha(
        contract["packet"], contract["packet_sha256"], "UL immutable packet"
    )
    source_archive = _require_sha(
        contract["source_archive"],
        contract["source_archive_sha256"],
        "UL source archive",
    )
    packet = read_json(packet_path)
    if packet.get("test_access") is not False or packet.get("lane_id") != contract["lane"]:
        raise ValueError("UL packet changed lane or test boundary")
    if packet.get("pins", {}).get(str(source_archive)) != contract["source_archive_sha256"]:
        raise ValueError("UL packet does not pin the owner-contract source archive")
    journal_value = read_json(journal)
    if (
        journal_value.get("schema") != "ul_v5_production_request_v1"
        or journal_value.get("project") != "Uncertainty_Lab"
        or journal_value.get("lane_id") != contract["lane"]
        or journal_value.get("packet") != str(packet_path)
        or journal_value.get("packet_sha256") != contract["packet_sha256"]
        or journal_value.get("state") != "prepared"
        or journal_value.get("requests") != []
        or journal_value.get("requested_gpus") != 1
        or Path(journal_value.get("account_lock", "")).resolve() != account_lock
        or Path(journal_value.get("policy", "")).resolve() != active_policy
    ):
        raise ValueError("UL journal is not the exact unsubmitted initial request")
    _require_sha(
        journal_value.get("batch_script", ""),
        journal_value.get("batch_script_sha256", ""),
        "UL batch script",
    )
    evidence = _require_sha(
        contract["predecessor_terminal_evidence"],
        contract["predecessor_terminal_evidence_sha256"],
        "UL predecessor terminal evidence",
    )
    rows = read_json(evidence)
    if set(rows) != TERMINAL_FIELDS or rows.get("schema") != "ul_v5_failed_gpu_attempt_v1":
        raise ValueError("Unknown UL predecessor evidence")
    if str(rows.get("job_id")) != str(contract["predecessor_failed_job"]):
        raise ValueError("UL predecessor evidence binds a different job")
    if (
        rows.get("state") != "FAILED"
        or rows.get("exit_code") != "1:0"
        or rows.get("user") != expected_user
        or rows.get("account") != expected_account
        or (
            required_gres_token is not None
            and required_gres_token not in str(rows.get("req_tres"))
        )
        or rows.get("test_access") is not False
    ):
        raise ValueError("UL predecessor is not the exact failed allocation")
    source = contract.get("source_commit")
    if (
        not isinstance(source, str)
        or len(source) != 40
        or any(character not in "0123456789abcdef" for character in source)
    ):
        raise ValueError("UL owner contract lacks an exact source commit")


def verify_policy_plan(
    plan_path: str | Path,
    *,
    expected_sha256: str,
    expected_active_policy: str | Path = PRODUCTION_POLICY,
    expected_account_lock: str | Path = PRODUCTION_LOCK,
    expected_look_gpu_job_id: str = "52429877",
    expected_user: str = PRODUCTION_USER,
    expected_account: str = PRODUCTION_ACCOUNT,
    required_ul_gres_token: str | None = "gres/gpu:a100=1",
) -> dict[str, Any]:
    path = _require_sha(plan_path, expected_sha256, "policy plan")
    plan = read_json(path)
    if set(plan) != PLAN_FIELDS or plan.get("schema") != "look_v5_monitor_policy_plan_v1":
        raise ValueError("Unknown monitor policy plan")
    if plan.get("test_access") is not False or str(plan.get("look_gpu_job_id")) != str(expected_look_gpu_job_id):
        raise ValueError("Monitor policy plan changed the LOOK execution boundary")
    active = Path(plan["active_policy"]).resolve()
    if active != Path(expected_active_policy).resolve() or sha256(active) != plan["initial_policy_sha256"]:
        raise ValueError("Active policy no longer matches the plan's initial state")
    claim = _require_sha(plan["look_claim"], plan["look_claim_sha256"], "LOOK claim")
    journal = _require_sha(plan["look_journal"], plan["look_journal_sha256"], "LOOK journal")
    claim_value, journal_value = read_json(claim), read_json(journal)
    if str(claim_value.get("job_id")) != str(expected_look_gpu_job_id) or claim_value.get("state") != "submitted":
        raise ValueError("LOOK claim no longer binds the preserved GPU job")
    matching = [
        row
        for row in journal_value.get("requests", [])
        if str(row.get("job_id")) == str(expected_look_gpu_job_id)
    ]
    if len(matching) != 1 or matching[0].get("v5_claim_sha256") != plan["look_claim_sha256"]:
        raise ValueError("LOOK dedicated journal does not uniquely bind the preserved job")

    previous_path, previous_sha = active, plan["initial_policy_sha256"]
    allowed = [previous_sha]
    states = plan.get("states")
    if not isinstance(states, list) or not states:
        raise ValueError("At least one explicit future policy state is required")
    for index, state in enumerate(states, start=1):
        if not isinstance(state, dict) or set(state) != STATE_FIELDS:
            raise ValueError("Malformed policy-plan state")
        if state["index"] != index or state["project"] != "Uncertainty_Lab":
            raise ValueError("Policy-plan states must be ordered UL appends")
        if state["previous_policy_sha256"] != previous_sha:
            raise ValueError("Policy-plan predecessor SHA is discontinuous")
        candidate_path = _require_sha(
            state["candidate_policy"], state["candidate_policy_sha256"], "candidate policy"
        )
        journal_path = _require_sha(
            state["appended_journal"],
            state["appended_journal_initial_sha256"],
            "UL journal initial state",
        )
        before, candidate = read_json(previous_path), read_json(candidate_path)
        expected = copy.deepcopy(before)
        journals = expected["projects"]["Uncertainty_Lab"]["request_journals"]
        journal_string = str(journal_path)
        if journal_string in journals:
            raise ValueError("UL journal is already registered")
        journals.append(journal_string)
        if candidate != expected:
            raise ValueError("Candidate policy changes more than one ordered UL journal append")
        _verify_owner_contract(
            state,
            journal_path,
            active_policy=active,
            account_lock=Path(expected_account_lock).resolve(),
            expected_user=expected_user,
            expected_account=expected_account,
            required_gres_token=required_ul_gres_token,
        )
        previous_path, previous_sha = candidate_path, state["candidate_policy_sha256"]
        if previous_sha in allowed:
            raise ValueError("Policy-plan state does not change the policy identity")
        allowed.append(previous_sha)
    return {"plan": plan, "plan_path": path, "allowed_policy_sha256": allowed}


def prepare_v22_package(
    *,
    v21_package: str | Path,
    v22_package: str | Path,
    v21_chain_sha256: str,
    v21_monitor_sha256: str,
    expected_v21_allowed_policy_sha256: list[str],
    plan_path: str | Path,
    plan_sha256: str,
    expected_active_policy: str | Path = PRODUCTION_POLICY,
    expected_account_lock: str | Path = PRODUCTION_LOCK,
    expected_look_gpu_job_id: str = "52429877",
    expected_user: str = PRODUCTION_USER,
    expected_account: str = PRODUCTION_ACCOUNT,
    required_ul_gres_token: str | None = "gres/gpu:a100=1",
) -> dict[str, Any]:
    verified = verify_policy_plan(
        plan_path,
        expected_sha256=plan_sha256,
        expected_active_policy=expected_active_policy,
        expected_account_lock=expected_account_lock,
        expected_look_gpu_job_id=expected_look_gpu_job_id,
        expected_user=expected_user,
        expected_account=expected_account,
        required_ul_gres_token=required_ul_gres_token,
    )
    old, new = Path(v21_package).resolve(), Path(v22_package).resolve()
    if new.exists():
        raise FileExistsError(new)
    _require_sha(old / "chain_manager.py", v21_chain_sha256, "v21 chain manager")
    _require_sha(old / "monitor.sh", v21_monitor_sha256, "v21 monitor")
    new.mkdir(parents=True)
    source_hashes = copy_immutable_tree(old / "source", new / "source")
    transformed: dict[str, str] = {}
    old_allow = "{" + ",".join(repr(value) for value in expected_v21_allowed_policy_sha256) + "}"
    new_allow = "{" + ",".join(repr(value) for value in verified["allowed_policy_sha256"]) + "}"
    chain = replace_exact_path((old / "chain_manager.py").read_text(), old, new, label="chain manager")
    if chain.count(old_allow) != 1:
        raise ValueError("v21 policy allowlist anchor changed")
    chain = chain.replace(old_allow, new_allow)
    write_exclusive_bytes(new / "chain_manager.py", chain.encode(), mode=0o444)
    transformed["chain_manager.py"] = sha256(new / "chain_manager.py")
    for name in (
        "stage_manager.py",
        "monitor.sh",
        "stage_monitor.sh",
        "run_gpu.sbatch",
        "run_pca.sbatch",
        "run_decision.sbatch",
    ):
        source = old / name
        text = replace_exact_path(source.read_text(), old, new, label=name)
        write_exclusive_bytes(new / name, text.encode(), mode=source.stat().st_mode & 0o777)
        transformed[name] = sha256(new / name)
    for target in new.rglob("*"):
        if target.is_file() and str(old) in target.read_text(errors="ignore"):
            raise ValueError(f"v22 package retains a v21 path: {target.relative_to(new)}")
    manifest = {
        "schema": "look_v5_v22_monitor_package_manifest_v1",
        "v21_package": str(old),
        "v21_chain_sha256": v21_chain_sha256,
        "v21_monitor_sha256": v21_monitor_sha256,
        "v22_package": str(new),
        "policy_plan": str(Path(plan_path).resolve()),
        "policy_plan_sha256": plan_sha256,
        "allowed_policy_sha256": verified["allowed_policy_sha256"],
        "transformed_files": transformed,
        "source_files": source_hashes,
        "test_access": False,
    }
    write_exclusive_json(new / "package_manifest.json", manifest)
    return {**manifest, "package_manifest_sha256": sha256(new / "package_manifest.json")}


class PolicyPlanMonitorInstaller:
    """No-op installer that revalidates the complete plan under the account lock."""

    REQUIRED = {
        "schema",
        "policy_plan",
        "policy_plan_sha256",
        "package_manifest",
        "package_manifest_sha256",
        "replacement_command",
        "replacement_source_sha256",
        "gpu_job_id",
        "test_access",
    }

    def __init__(
        self,
        binding_path: str | Path,
        *,
        expected_sha256: str,
        expected_active_policy: str | Path = PRODUCTION_POLICY,
        expected_account_lock: str | Path = PRODUCTION_LOCK,
        expected_look_gpu_job_id: str = "52429877",
        expected_user: str = PRODUCTION_USER,
        expected_account: str = PRODUCTION_ACCOUNT,
        required_ul_gres_token: str | None = "gres/gpu:a100=1",
    ) -> None:
        self.path = _require_sha(binding_path, expected_sha256, "v22 binding")
        self.binding = read_json(self.path)
        if set(self.binding) != self.REQUIRED or self.binding.get("schema") != "look_v5_monitor_successor_binding_v1":
            raise ValueError("Unknown v22 monitor binding")
        if self.binding.get("test_access") is not False or self.binding.get("gpu_job_id") != str(expected_look_gpu_job_id):
            raise ValueError("V22 binding changed the execution boundary")
        self.plan_context = {
            "expected_active_policy": expected_active_policy,
            "expected_account_lock": expected_account_lock,
            "expected_look_gpu_job_id": str(expected_look_gpu_job_id),
            "expected_user": expected_user,
            "expected_account": expected_account,
            "required_ul_gres_token": required_ul_gres_token,
        }

    def verify(self, spec: dict[str, Any]) -> dict[str, str]:
        verified = verify_policy_plan(
            self.binding["policy_plan"],
            expected_sha256=self.binding["policy_plan_sha256"],
            **self.plan_context,
        )
        manifest_path = _require_sha(
            self.binding["package_manifest"],
            self.binding["package_manifest_sha256"],
            "v22 package manifest",
        )
        manifest = read_json(manifest_path)
        replacement = _require_sha(
            self.binding["replacement_command"],
            self.binding["replacement_source_sha256"],
            "v22 replacement monitor",
        )
        if str(replacement) != str(Path(spec["replacement_command"]).resolve()):
            raise ValueError("V22 replacement command changed")
        if manifest.get("allowed_policy_sha256") != verified["allowed_policy_sha256"]:
            raise ValueError("V22 package allowlist differs from the policy plan")
        return {
            "policy_sha256": verified["allowed_policy_sha256"][0],
            "journal_sha256": self.binding["policy_plan_sha256"],
            "claim_sha256": verified["plan"]["look_claim_sha256"],
            "provenance_sha256": self.binding["package_manifest_sha256"],
        }

    def look_claim(self) -> Path:
        verified = verify_policy_plan(
            self.binding["policy_plan"],
            expected_sha256=self.binding["policy_plan_sha256"],
            **self.plan_context,
        )
        return Path(verified["plan"]["look_claim"]).resolve()

    preflight = verify

    def commit(self, spec: dict[str, Any], _: str) -> dict[str, str]:
        return self.verify(spec)


def build_successor_binding(
    *, plan_path: str | Path, package_manifest: str | Path,
    replacement_command: str | Path, gpu_job_id: str,
) -> dict[str, Any]:
    plan, manifest, replacement = (
        Path(plan_path).resolve(),
        Path(package_manifest).resolve(),
        Path(replacement_command).resolve(),
    )
    for path in (plan, manifest, replacement):
        if not path.is_file():
            raise ValueError(f"Missing v22 binding artifact: {path}")
    return {
        "schema": "look_v5_monitor_successor_binding_v1",
        "policy_plan": str(plan),
        "policy_plan_sha256": sha256(plan),
        "package_manifest": str(manifest),
        "package_manifest_sha256": sha256(manifest),
        "replacement_command": str(replacement),
        "replacement_source_sha256": sha256(replacement),
        "gpu_job_id": str(gpu_job_id),
        "test_access": False,
    }


def apply_v22_successor(
    *, spec_path: str | Path, binding_path: str | Path,
    receipt_dir: str | Path, allow_production: bool,
) -> dict[str, Any]:
    if not allow_production:
        raise PermissionError("Explicit production apply flag is required")
    spec = read_json(spec_path)
    if spec.get("mode") != "production_v22" or Path(spec["shared_lock"]).resolve() != PRODUCTION_LOCK:
        raise ValueError("V22 production spec changed mode or lock")
    if spec.get("expected_user") != PRODUCTION_USER or spec.get("expected_account") != PRODUCTION_ACCOUNT:
        raise ValueError("V22 production Slurm identity changed")
    installer = PolicyPlanMonitorInstaller(binding_path, expected_sha256=spec["binding_sha256"])
    slurm = SlurmAdapter(user=PRODUCTION_USER, account=PRODUCTION_ACCOUNT)
    return run_handover(
        receipt_dir,
        lock_path=PRODUCTION_LOCK,
        spec=spec,
        observe=slurm.observe,
        find_held=slurm.find_held,
        submit_held=lambda value: slurm.submit_held(
            value, installer.look_claim(), receipt_dir
        ),
        cancel=slurm.cancel,
        preflight_artifacts=installer.preflight,
        commit_artifacts=installer.commit,
        release=slurm.release,
        account_gpu_count=slurm.account_gpu_count,
    )


def build_v22_spec(
    *, old_job: dict[str, Any], gpu_job: dict[str, Any], replacement_command: str | Path,
    binding_path: str | Path,
) -> dict[str, Any]:
    return build_handover_spec(
        old_job=old_job,
        gpu_job=gpu_job,
        replacement_command=replacement_command,
        binding_path=binding_path,
        shared_lock=PRODUCTION_LOCK,
        mode="production_v22",
    )
