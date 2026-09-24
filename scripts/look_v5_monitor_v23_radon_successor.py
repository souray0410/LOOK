#!/usr/bin/env python3
"""Prepare or explicitly apply the exact R&B-aware LOOK v23 successor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from look.runtime.v5_monitor_handover import write_exclusive
from look.runtime.v5_monitor_ibex import (
    PRODUCTION_ACCOUNT,
    PRODUCTION_POLICY,
    PRODUCTION_USER,
    SlurmAdapter,
    sha256,
)
from look.runtime.v5_monitor_radon_successor import (
    INITIAL_POLICY_SHA256,
    apply_v23_successor,
    build_radon_owner_contract,
    build_radon_policy_plan,
    build_radon_successor_binding,
    build_v23_spec,
    prepare_v23_package,
    verify_radon_policy_plan,
)


def _exact(path: str, expected: str, label: str) -> Path:
    value = Path(path).resolve()
    if not value.is_file() or sha256(value) != expected:
        raise ValueError(f"{label} identity changed")
    return value


def prepare(args: argparse.Namespace) -> dict:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if sha256(PRODUCTION_POLICY) != INITIAL_POLICY_SHA256:
        raise ValueError("Live role policy is not the exact accepted UL state")
    claim = _exact(args.look_claim, args.look_claim_sha256, "LOOK claim")
    look_journal = _exact(args.look_journal, args.look_journal_sha256, "LOOK journal")
    journal = _exact(args.radon_journal, args.radon_journal_sha256, "R&B initial journal")
    candidate = _exact(args.candidate_policy, args.candidate_policy_sha256, "R&B candidate policy")
    archive = _exact(args.runner_archive, args.runner_archive_sha256, "R&B runner archive")
    acceptance = _exact(args.runner_acceptance, args.runner_acceptance_sha256, "R&B runner acceptance")
    independent_review = _exact(
        args.runner_independent_review,
        args.runner_independent_review_sha256,
        "R&B independent code review",
    )
    bundle_review = _exact(
        args.bundle_independent_review,
        args.bundle_independent_review_sha256,
        "R&B cap-10 bundle independent review",
    )
    transaction_review = _exact(
        args.transaction_independent_review,
        args.transaction_independent_review_sha256,
        "R&B production-transaction independent review",
    )
    proposal = _exact(args.policy_proposal, args.policy_proposal_sha256, "R&B policy proposal")

    slurm = SlurmAdapter(user=PRODUCTION_USER, account=PRODUCTION_ACCOUNT)
    gpu = slurm.observe("52429877")
    old = slurm.observe("52462712")
    if gpu.get("state") not in {"PENDING", "RUNNING"}:
        raise ValueError("LOOK GPU predecessor is no longer live")
    if (
        old.get("state") != "PENDING"
        or old.get("dependency") != "afterany:52429877(unfulfilled)"
        or old.get("has_step") is not False
    ):
        raise ValueError("LOOK v22 monitor is no longer safely replaceable")

    output.mkdir(parents=True)
    contract = output / "radon_prebinding.json"
    write_exclusive(
        contract,
        build_radon_owner_contract(
            journal=journal,
            candidate_policy=candidate,
            runner_archive=archive,
            runner_acceptance=acceptance,
            runner_independent_review=independent_review,
            bundle_independent_review=bundle_review,
            transaction_independent_review=transaction_review,
            policy_proposal=proposal,
        ),
    )
    plan = output / "policy_plan.json"
    write_exclusive(
        plan,
        build_radon_policy_plan(
            active_policy=PRODUCTION_POLICY,
            look_gpu_job_id="52429877",
            look_claim=claim,
            look_journal=look_journal,
            candidate_policy=candidate,
            appended_journal=journal,
            owner_contract=contract,
        ),
    )
    verified = verify_radon_policy_plan(plan, expected_sha256=sha256(plan))
    expected_allowed = [INITIAL_POLICY_SHA256, args.candidate_policy_sha256]
    if verified["allowed_policy_sha256"] != expected_allowed:
        raise ValueError("V23 plan differs from the exact current-to-R&B policy pair")

    v23 = output / "look_cataract_middle_v5_feature_chain_sharded_v23"
    package = prepare_v23_package(
        v22_package=args.v22_package,
        v23_package=v23,
        v22_chain_sha256=args.v22_chain_sha256,
        v22_monitor_sha256=args.v22_monitor_sha256,
        expected_v22_allowed_policy_sha256=args.v22_allowed_policy_sha256,
        plan_path=plan,
        plan_sha256=sha256(plan),
    )
    binding = output / "binding.json"
    write_exclusive(
        binding,
        build_radon_successor_binding(
            plan_path=plan,
            package_manifest=v23 / "package_manifest.json",
            replacement_command=v23 / "monitor.sh",
            gpu_job_id="52429877",
        ),
    )
    spec = output / "handover_spec.json"
    write_exclusive(
        spec,
        build_v23_spec(
            old_job=old,
            gpu_job=gpu,
            replacement_command=v23 / "monitor.sh",
            binding_path=binding,
        ),
    )
    receipt = {
        "schema": "look_v5_monitor_v23_radon_candidate_v1",
        "state": "prepared_not_applied",
        "old_monitor_to_replace": "52462712",
        "gpu_job_preserved": "52429877",
        "initial_policy_sha256": INITIAL_POLICY_SHA256,
        "candidate_policy_sha256": args.candidate_policy_sha256,
        "allowed_policy_sha256": expected_allowed,
        "radon_journal_initial_sha256": args.radon_journal_sha256,
        "radon_prebinding_sha256": sha256(contract),
        "policy_plan_sha256": sha256(plan),
        "package_manifest_sha256": package["package_manifest_sha256"],
        "binding_sha256": sha256(binding),
        "handover_spec_sha256": sha256(spec),
        "requested_gpus": 0,
        "test_access": False,
    }
    write_exclusive(output / "candidate_receipt.json", receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for field in (
        "output", "v22-package", "v22-chain-sha256", "v22-monitor-sha256",
        "look-claim", "look-claim-sha256", "look-journal", "look-journal-sha256",
        "radon-journal", "radon-journal-sha256", "candidate-policy",
        "candidate-policy-sha256", "runner-archive", "runner-archive-sha256",
        "runner-acceptance", "runner-acceptance-sha256", "policy-proposal",
        "policy-proposal-sha256", "runner-independent-review",
        "runner-independent-review-sha256", "bundle-independent-review",
        "bundle-independent-review-sha256",
        "transaction-independent-review", "transaction-independent-review-sha256",
    ):
        prep.add_argument(f"--{field}", required=True)
    prep.add_argument("--v22-allowed-policy-sha256", action="append", required=True)
    apply = commands.add_parser("apply")
    apply.add_argument("--spec", required=True)
    apply.add_argument("--binding", required=True)
    apply.add_argument("--receipt-dir", required=True)
    apply.add_argument("--apply-production", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare":
        value = prepare(args)
    else:
        value = apply_v23_successor(
            spec_path=args.spec,
            binding_path=args.binding,
            receipt_dir=args.receipt_dir,
            allow_production=args.apply_production,
        )
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
