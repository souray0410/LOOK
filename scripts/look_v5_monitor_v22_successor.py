#!/usr/bin/env python3
"""Prepare or explicitly apply the exact-state LOOK v22 monitor successor."""
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
    read_json,
    sha256,
)
from look.runtime.v5_monitor_successor import (
    apply_v22_successor,
    build_failed_terminal_evidence,
    build_owner_contract,
    build_policy_plan,
    build_successor_binding,
    build_v22_spec,
    prepare_v22_package,
    verify_policy_plan,
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
    if sha256(PRODUCTION_POLICY) != args.expected_current_policy_sha256:
        raise ValueError("Live role policy changed before v22 preparation")
    claim = _exact(args.look_claim, args.look_claim_sha256, "LOOK claim")
    look_journal = _exact(args.look_journal, args.look_journal_sha256, "LOOK journal")
    source = _exact(args.source_archive, args.source_archive_sha256, "UL source archive")
    if len(args.source_commit) != 40:
        raise ValueError("UL source commit must be full length")

    lanes = []
    for prefix in ("even", "odd"):
        lane = {
            "name": getattr(args, f"{prefix}_lane"),
            "journal": _exact(
                getattr(args, f"{prefix}_journal"),
                getattr(args, f"{prefix}_journal_sha256"),
                f"{prefix} initial journal",
            ),
            "packet": _exact(
                getattr(args, f"{prefix}_packet"),
                getattr(args, f"{prefix}_packet_sha256"),
                f"{prefix} packet",
            ),
            "candidate": _exact(
                getattr(args, f"{prefix}_candidate_policy"),
                getattr(args, f"{prefix}_candidate_policy_sha256"),
                f"{prefix} candidate policy",
            ),
            "predecessor": str(getattr(args, f"{prefix}_predecessor_job")),
        }
        lanes.append(lane)

    slurm = SlurmAdapter(user=PRODUCTION_USER, account=PRODUCTION_ACCOUNT)
    gpu = slurm.observe("52429877")
    old = slurm.observe("52455266")
    if gpu.get("state") not in {"PENDING", "RUNNING"}:
        raise ValueError("LOOK GPU predecessor is no longer live")
    if (
        old.get("state") != "PENDING"
        or old.get("dependency") != "afterany:52429877(unfulfilled)"
        or old.get("has_step") is not False
    ):
        raise ValueError("LOOK v21 monitor is no longer safely replaceable")

    terminal_rows = []
    for lane in lanes:
        row = slurm.observe(lane["predecessor"])
        if row.get("state") != "FAILED" or row.get("exit_code") != "1:0":
            raise ValueError(f"UL predecessor {lane['predecessor']} is not exact FAILED 1:0")
        terminal_rows.append(row)

    output.mkdir(parents=True)
    states = []
    for lane, terminal in zip(lanes, terminal_rows, strict=True):
        lane_root = output / lane["name"]
        evidence = lane_root / "predecessor_terminal.json"
        write_exclusive(evidence, build_failed_terminal_evidence(terminal))
        contract = lane_root / "owner_contract.json"
        write_exclusive(
            contract,
            build_owner_contract(
                lane=lane["name"],
                journal=lane["journal"],
                packet=lane["packet"],
                source_commit=args.source_commit,
                source_archive=source,
                predecessor_failed_job=lane["predecessor"],
                predecessor_terminal_evidence=evidence,
            ),
        )
        states.append(
            {
                "candidate_policy": str(lane["candidate"]),
                "appended_journal": str(lane["journal"]),
                "owner_contract": str(contract),
            }
        )

    plan_path = output / "policy_plan.json"
    write_exclusive(
        plan_path,
        build_policy_plan(
            active_policy=PRODUCTION_POLICY,
            look_gpu_job_id="52429877",
            look_claim=claim,
            look_journal=look_journal,
            states=states,
        ),
    )
    verified = verify_policy_plan(plan_path, expected_sha256=sha256(plan_path))
    expected_allowed = [
        args.expected_current_policy_sha256,
        args.even_candidate_policy_sha256,
        args.odd_candidate_policy_sha256,
    ]
    if verified["allowed_policy_sha256"] != expected_allowed:
        raise ValueError("V22 policy plan does not match the explicit three-state sequence")

    v22 = output / "look_cataract_middle_v5_feature_chain_sharded_v22"
    package = prepare_v22_package(
        v21_package=args.v21_package,
        v22_package=v22,
        v21_chain_sha256=args.v21_chain_sha256,
        v21_monitor_sha256=args.v21_monitor_sha256,
        expected_v21_allowed_policy_sha256=args.v21_allowed_policy_sha256,
        plan_path=plan_path,
        plan_sha256=sha256(plan_path),
    )
    binding_path = output / "binding.json"
    write_exclusive(
        binding_path,
        build_successor_binding(
            plan_path=plan_path,
            package_manifest=v22 / "package_manifest.json",
            replacement_command=v22 / "monitor.sh",
            gpu_job_id="52429877",
        ),
    )
    spec_path = output / "handover_spec.json"
    write_exclusive(
        spec_path,
        build_v22_spec(
            old_job=old,
            gpu_job=gpu,
            replacement_command=v22 / "monitor.sh",
            binding_path=binding_path,
        ),
    )
    receipt = {
        "schema": "look_v5_monitor_v22_candidate_v1",
        "state": "prepared_not_applied",
        "old_monitor_to_replace": "52455266",
        "gpu_job_preserved": "52429877",
        "initial_policy_sha256": args.expected_current_policy_sha256,
        "allowed_policy_sha256": expected_allowed,
        "policy_plan_sha256": sha256(plan_path),
        "package_manifest_sha256": package["package_manifest_sha256"],
        "binding_sha256": sha256(binding_path),
        "handover_spec_sha256": sha256(spec_path),
        "terminal_predecessors": [row["job_id"] for row in terminal_rows],
        "requested_gpus": 0,
        "test_access": False,
    }
    write_exclusive(output / "candidate_receipt.json", receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--output", required=True)
    prep.add_argument("--expected-current-policy-sha256", required=True)
    prep.add_argument("--v21-package", required=True)
    prep.add_argument("--v21-chain-sha256", required=True)
    prep.add_argument("--v21-monitor-sha256", required=True)
    prep.add_argument("--v21-allowed-policy-sha256", action="append", required=True)
    prep.add_argument("--look-claim", required=True)
    prep.add_argument("--look-claim-sha256", required=True)
    prep.add_argument("--look-journal", required=True)
    prep.add_argument("--look-journal-sha256", required=True)
    prep.add_argument("--source-commit", required=True)
    prep.add_argument("--source-archive", required=True)
    prep.add_argument("--source-archive-sha256", required=True)
    for prefix in ("even", "odd"):
        prep.add_argument(f"--{prefix}-lane", required=True)
        prep.add_argument(f"--{prefix}-journal", required=True)
        prep.add_argument(f"--{prefix}-journal-sha256", required=True)
        prep.add_argument(f"--{prefix}-packet", required=True)
        prep.add_argument(f"--{prefix}-packet-sha256", required=True)
        prep.add_argument(f"--{prefix}-candidate-policy", required=True)
        prep.add_argument(f"--{prefix}-candidate-policy-sha256", required=True)
        prep.add_argument(f"--{prefix}-predecessor-job", required=True)
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
        value = apply_v22_successor(
            spec_path=args.spec,
            binding_path=args.binding,
            receipt_dir=args.receipt_dir,
            allow_production=args.apply_production,
        )
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
