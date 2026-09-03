#!/usr/bin/env python3
"""Bounded validation-only continuation of baseline qualification."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path


def regularization_profiles(base):
    profiles = []
    for decay in (1e-3, 1e-2):
        for smoothing in (0.0, 0.05, 0.1):
            profile = copy.deepcopy(base)
            profile.update(name=f"{base['name']}__wd-{decay:.0e}__ls-{smoothing:.2f}",
                           weight_decay=decay, label_smoothing=smoothing)
            profiles.append(profile)
    return profiles


def gate_checks(winner, rows, references):
    from look_core.study_grid import (
        BASELINE_AUROC_GATE, BASELINE_MACRO_F1_GATE,
        MIN_SENSITIVITY_GATE, MIN_SPECIFICITY_GATE, CONFIRMATION_SEEDS,
    )
    values = [r[k] for r in rows + references for k in ("macro_auroc_ovr", "macro_f1", "ece_15")]
    return {
        "finite_metrics": all(math.isfinite(v) for v in values),
        "three_seeds": sorted(r["seed"] for r in rows) == CONFIRMATION_SEEDS,
        "both_unimodal_references": {r["fusion_position"] for r in references} == {"oct_only", "cfp_only"},
        "mean_auroc": winner["mean_macro_auroc_ovr"] >= BASELINE_AUROC_GATE,
        "mean_macro_f1": winner["mean_macro_f1"] >= BASELINE_MACRO_F1_GATE,
        "sensitivity": statistics.fmean(r["sensitivity_per_class"][1] for r in rows) >= MIN_SENSITIVITY_GATE,
        "specificity": statistics.fmean(r["specificity_per_class"][1] for r in rows) >= MIN_SPECIFICITY_GATE,
        "multimodal_not_worse": winner["mean_macro_auroc_ovr"] >= max(r["macro_auroc_ovr"] for r in references),
    }


def find_candidate(root, confirmation_plan):
    matches = []
    for path in sorted(root.glob("baseline_candidate__*.json")):
        value = json.loads(path.read_text())
        if value.get("stage_b_plan_id") == confirmation_plan:
            matches.append((path, value))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one candidate for {confirmation_plan}; got {len(matches)}")
    return matches[0]


def main():
    from look_core.cli import add_runtime_arguments, resolve_runtime_arguments
    from look_core.state import PipelineState, atomic_write_json, atomic_write_text, file_sha256, stable_hash, utc_now

    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument("--confirmation-plan", required=True)
    parser.add_argument("--wait-session", default="look-glaucoma-baseline")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--auto-look-validation", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.max_hours) or args.max_hours <= 0:
        parser.error("--max-hours must be finite and positive")
    paths = resolve_runtime_arguments(args)
    identity = dict(confirmation_plan=args.confirmation_plan,
                    labels_sha256=file_sha256(paths.labels_csv),
                    script_sha256=file_sha256(Path(__file__)), gpus=args.gpus,
                    max_hours=args.max_hours, auto_look_validation=args.auto_look_validation)
    output = paths.runs_root / "overnight" / stable_hash(identity)[:12]
    summary = dict(identity, status="planned", stage="waiting", output=str(output), test_access=False)
    if not args.execute:
        print(json.dumps(summary, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    with PipelineState(paths.cache_root / "pipeline_state", "overnight_validation", identity, [paths.labels_csv]) as state:
        start_path = output / "started.json"
        if not start_path.exists():
            atomic_write_json({"unix": time.time(), "utc": utc_now()}, start_path)
        deadline = json.loads(start_path.read_text())["unix"] + args.max_hours * 3600

        def report(stage, **values):
            summary.update(stage=stage, updated_at_utc=utc_now(), **values)
            atomic_write_json(summary, output / "summary.json")
            atomic_write_text("# Overnight Validation\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n", output / "SUMMARY.md")
            state.checkpoint(stage=stage, summary=str(output / "summary.json"))
            print(f"{utc_now()} stage={stage} status={summary['status']}", flush=True)

        def budget():
            if time.time() >= deadline:
                raise TimeoutError("Soft time budget reached; results retained; no new stage started")

        try:
            report("waiting_for_baseline", status="waiting")
            while subprocess.run(["tmux", "has-session", "-t", args.wait_session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                budget()
                time.sleep(30)
            env = paths.runs_root / "logs/detached_validation_status.env"
            predecessor = dict(line.split("=", 1) for line in env.read_text().splitlines() if "=" in line)
            if predecessor.get("session") != args.wait_session or predecessor.get("state") != "complete":
                raise RuntimeError("Predecessor failed/interrupted; refusing to consume stale candidates")
            from look_core.distributed import parse_gpu_devices
            devices = parse_gpu_devices(args.gpus)
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
            import torch
            from look_core.artifacts import build_file_manifest, validate_file_manifest
            from look_core.task_selection import validate_selected_task
            from look_core.study_grid import (
                StudyGrid, _rows, _ranking_key, _aggregate_confirmation,
                _disabled_look_profile, run_study_grid, baseline_search_grid,
                baseline_confirmation_grid, unimodal_reference_grid,
            )
            validate_selected_task(paths)
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA required")
            device = torch.device("cuda:0")
            candidate_path, candidate = find_candidate(paths.runs_root / "baseline_selection/candidates", args.confirmation_plan)
            if not candidate.get("artifacts"):
                raise RuntimeError("Candidate has no artifact evidence")
            errors = validate_file_manifest(candidate["artifacts"])
            if errors:
                raise RuntimeError(f"Candidate artifact verification failed: {errors[:5]}")
            report("baseline_complete", status="running", original_candidate=str(candidate_path), original_ranking=candidate["ranking"])

            def run(grid, name):
                budget()
                if name == "look_validation_pilot":
                    summary["look_started"] = True
                report(name, status="running", active_grid=grid.__dict__)
                result = run_study_grid(grid, paths, device, execute=True, phase="validation", gpu_devices=devices)
                rows = _rows(paths, result)
                summary.setdefault("completed_stages", {})[name] = dict(plan_id=result["plan_id"], rows=rows)
                report(name + "_complete")
                return result, rows

            rows = _rows(paths, {"plan_id": candidate["stage_b_plan_id"]})
            winner_rows = [r for r in rows if r["fusion_position"] == candidate["winner"]["fusion_position"]]
            checks = gate_checks(candidate["winner"], winner_rows, candidate["unimodal_references"])
            if not all(checks.values()):
                profiles = regularization_profiles(candidate["classifier_profile"])
                grid = StudyGrid(fusion_positions=["feature"], seeds=[3407],
                    filling_strategies=["normalized_mean"], classifier_profiles=profiles,
                    gan_profiles=[{"name": "not_applicable"}], look_profiles=[_disabled_look_profile()])
                calibration, rows = run(grid, "six_regularization_profiles")
                original_rows = _rows(paths, {"plan_id": candidate["calibration_plan_id"]})
                if _ranking_key(rows[0]) >= _ranking_key(original_rows[0]):
                    report("complete", status="quality_gate_failed", reason="No new profile improved the original reference ranking", original_gate_checks=checks, look_started=False)
                    state.complete([output / "summary.json", output / "SUMMARY.md"])
                    return
                profile = next(p for p in profiles if p["name"] == rows[0]["classifier_profile"])
                refs_plan, refs = run(unimodal_reference_grid(profile), "regularized_unimodal_references")
                stage_a, rows = run(baseline_search_grid(profile), "regularized_seven_fusions")
                positions = [r["fusion_position"] for r in rows[:3]]
                stage_b, rows = run(baseline_confirmation_grid(positions, profile), "regularized_top3_confirmation")
                ranking = _aggregate_confirmation(rows)
                winner = ranking[0]
                winner_rows = [r for r in rows if r["fusion_position"] == winner["fusion_position"]]
                checks = gate_checks(winner, winner_rows, refs)
                artifacts = []
                for row in [*winner_rows, *refs]:
                    b = paths.runs_root / "backbones" / row["backbone_id"]
                    artifacts.extend([b / "best.pt", b / "training_complete.json", paths.runs_root / "experiments" / row["experiment_id"] / "validation_result.json"])
                candidate = dict(protocol="ukb_glaucoma_binary_baseline_selection",
                    classifier_profile=profile, winner=winner, ranking=ranking,
                    calibration_plan_id=calibration["plan_id"], stage_a_plan_id=stage_a["plan_id"],
                    stage_b_plan_id=stage_b["plan_id"], unimodal_reference_plan_id=refs_plan["plan_id"],
                    unimodal_references=refs, artifacts=build_file_manifest(artifacts),
                    quality_gate_checks=checks, quality_gate_passed=all(checks.values()),
                    provenance="bounded exploratory follow-up on the same validation split; not independent confirmation")
                candidate["selection_id"] = stable_hash(candidate)[:12]
                candidate_path = output / "regularized_candidate.json"
                atomic_write_json(candidate, candidate_path)
            report("candidate_checked", candidate=str(candidate_path), ranking=candidate["ranking"], quality_gate_checks=checks)
            if not all(checks.values()):
                report("complete", status="quality_gate_failed", reason="No additional search; unchanged quality gates failed", look_started=False)
            elif args.auto_look_validation:
                approved = dict(candidate, status="frozen", approval_mode="automated_predeclared_gate",
                    scope="validation_pilot_only", approved_at_utc=utc_now())
                atomic_write_json(approved, output / "validation_only_baseline.json")
                pilot = StudyGrid(fusion_positions=[candidate["winner"]["fusion_position"]], seeds=[3407],
                    filling_strategies=["raw_zero", "normalized_mean"], classifier_profiles=[candidate["classifier_profile"]])
                report("look_validation", look_requested=True, approval_scope="single-seed zero/mean LOOK validation; no GAN or sealed test")
                run(pilot, "look_validation_pilot")
                report("complete", status="complete", reason="Baseline and validation-only LOOK pilot complete")
            else:
                report("complete", status="candidate_ready", look_started=False)
            state.complete([output / "summary.json", output / "SUMMARY.md"])
        except TimeoutError as error:
            report("budget_reached", status="budget_reached", reason=str(error))
            state.complete([output / "summary.json", output / "SUMMARY.md"])
        except BaseException as error:
            report("failed", status="failed", error=repr(error))
            raise


if __name__ == "__main__":
    main()
