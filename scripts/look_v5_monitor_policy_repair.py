#!/usr/bin/env python3
"""Prepare or explicitly apply the LOOK V5 monitor-only policy repair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from look.runtime.v5_monitor_ibex import (
    PRODUCTION_ACCOUNT,
    PRODUCTION_LOCK,
    PRODUCTION_POLICY,
    PRODUCTION_USER,
    SlurmAdapter,
    apply_production,
    build_handover_spec,
    build_policy_binding,
    sha256,
)
from look.runtime.v5_monitor_handover import write_exclusive
from look.runtime.v5_role_policy_transition import (
    build_dedicated_journal,
    build_look_only_policy,
    prepare_v21_package,
    verify_look_only_delta,
    write_exclusive_bytes,
    write_exclusive_json,
)


def prepare(args: argparse.Namespace) -> dict:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if sha256(PRODUCTION_POLICY) != args.expected_current_policy_sha256:
        raise ValueError("Live role policy changed before candidate preparation")
    if sha256(args.claim) != args.expected_claim_sha256:
        raise ValueError("LOOK V5 claim changed")
    if sha256(args.replacement_receipt) != args.expected_replacement_receipt_sha256:
        raise ValueError("LOOK replacement receipt changed")
    slurm = SlurmAdapter(user=PRODUCTION_USER, account=PRODUCTION_ACCOUNT)
    gpu = slurm.observe("52429877")
    old = slurm.observe("52430491")
    if gpu.get("state") not in {"PENDING", "RUNNING"}:
        raise ValueError("GPU predecessor is no longer live")
    if old.get("state") != "PENDING" or old.get("has_step") is not False:
        raise ValueError("Old monitor is no longer safely replaceable")
    output.mkdir(parents=True)
    base = output / "role_policy.base.json"
    write_exclusive_bytes(base, PRODUCTION_POLICY.read_bytes())
    journal_path = output / "look_v5_requests.json"
    job = {
        "job_id": gpu["job_id"],
        "name": gpu["name"],
        "command": gpu["command"],
        "submit_time": gpu["submit_time"],
        "state": gpu["state"],
        "user": gpu["user"],
        "account": gpu["account"],
        "req_tres": gpu["req_tres"],
        "replaces_job_id": "52422962",
        "cancelled_retry_job_id": "52429606",
        "replacement_receipt": str(Path(args.replacement_receipt).resolve()),
        "replacement_receipt_sha256": args.expected_replacement_receipt_sha256,
    }
    journal = build_dedicated_journal(
        job=job, claim_path=args.claim, expected_claim_sha256=args.expected_claim_sha256
    )
    write_exclusive_json(journal_path, journal)
    base_value = json.loads(base.read_text())
    candidate_value = build_look_only_policy(current=base_value, dedicated_journal=journal_path)
    candidate = output / "role_policy.look_only.json"
    write_exclusive_json(candidate, candidate_value)
    verify_look_only_delta(base_value, candidate_value, journal_path)
    v21 = output / "look_cataract_middle_v5_feature_chain_sharded_v21"
    package = prepare_v21_package(
        v20_package=args.v20_package,
        v21_package=v21,
        v20_chain_sha256=args.v20_chain_sha256,
        v20_monitor_sha256=args.v20_monitor_sha256,
        dedicated_journal=journal_path,
        allowed_policy_sha256=[sha256(base), sha256(candidate)],
    )
    provenance = output / "provenance.json"
    write_exclusive_json(
        provenance,
        {
            "schema": "look_v5_monitor_policy_repair_provenance_v1",
            "base_policy_sha256": sha256(base),
            "candidate_policy_sha256": sha256(candidate),
            "dedicated_journal_sha256": sha256(journal_path),
            "claim": str(Path(args.claim).resolve()),
            "claim_sha256": args.expected_claim_sha256,
            "replacement_receipt": str(Path(args.replacement_receipt).resolve()),
            "replacement_receipt_sha256": args.expected_replacement_receipt_sha256,
            "gpu_job": gpu,
            "old_monitor": old,
            "v21_package_manifest_sha256": package["package_manifest_sha256"],
            "test_access": False,
        },
    )
    binding_value = build_policy_binding(
        active_policy=PRODUCTION_POLICY,
        base_policy_snapshot=base,
        candidate_policy=candidate,
        dedicated_journal=journal_path,
        claim=args.claim,
        provenance=provenance,
        replacement_command=v21 / "monitor.sh",
        gpu_job_id="52429877",
    )
    binding = output / "binding.json"
    write_exclusive(binding, binding_value)
    spec_value = build_handover_spec(
        old_job=old,
        gpu_job=gpu,
        replacement_command=v21 / "monitor.sh",
        binding_path=binding,
        shared_lock=PRODUCTION_LOCK,
    )
    spec = output / "handover_spec.json"
    write_exclusive(spec, spec_value)
    receipt = {
        "schema": "look_v5_monitor_policy_repair_candidate_v1",
        "state": "prepared_not_applied",
        "base_policy_sha256": sha256(base),
        "candidate_policy_sha256": sha256(candidate),
        "dedicated_journal_sha256": sha256(journal_path),
        "v21_package_manifest_sha256": package["package_manifest_sha256"],
        "binding_sha256": sha256(binding),
        "handover_spec_sha256": sha256(spec),
        "gpu_job_preserved": "52429877",
        "old_monitor_to_replace": "52430491",
        "test_access": False,
    }
    write_exclusive(output / "candidate_receipt.json", receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--output", required=True)
    prep.add_argument("--v20-package", required=True)
    prep.add_argument("--v20-chain-sha256", required=True)
    prep.add_argument("--v20-monitor-sha256", required=True)
    prep.add_argument("--expected-current-policy-sha256", required=True)
    prep.add_argument("--claim", required=True)
    prep.add_argument("--expected-claim-sha256", required=True)
    prep.add_argument("--replacement-receipt", required=True)
    prep.add_argument("--expected-replacement-receipt-sha256", required=True)
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
        value = apply_production(
            spec_path=args.spec,
            binding_path=args.binding,
            receipt_dir=args.receipt_dir,
            allow_production=args.apply_production,
        )
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
