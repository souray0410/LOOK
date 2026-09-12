"""Freeze a finite native screen and audit eligibility without launching LOOK.

Queue status is not acceptance. No candidate is selected while another candidate
in the same screen is unresolved. This module neither reads test nor submits GPUs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from look.runtime.state import atomic_write_json, file_sha256, stable_hash, utc_now

MODELS = ("resnet50", "densenet121", "swin_b")
TRACKS = ("cfp_2d", "oct_bscan_2d")
DISEASES = ("cataract", "glaucoma", "macular_degeneration")


def collect(queues):
    """Read exact candidate specs, excluding other diseases and volume inputs."""
    candidates = {}
    data_identities = {}
    for queue in queues:
        q = json.loads(Path(queue).read_text())
        for task in q.get("tasks", []):
            if not all(k in task for k in ("spec", "spec_sha256", "run_dir")):
                continue
            spec_path = Path(task["spec"])
            if file_sha256(spec_path) != task["spec_sha256"]:
                raise ValueError("Candidate spec changed")
            spec = json.loads(spec_path.read_text())
            if (spec.get("model", {}).get("name") not in MODELS or
                    spec.get("track") not in TRACKS or
                    spec.get("disease") not in DISEASES or
                    spec.get("training", {}).get("seed") != 3416):
                continue
            if spec.get("test_used") is not False or spec["model"]["spatial_dims"] != 2:
                raise ValueError("Only test-sealed 2D native candidates are allowed")
            identity = {k: spec[k] for k in ("train_manifest_sha256", "development_manifest_sha256",
                        "cache_receipt_sha256", "aggregation")}
            key = spec["disease"] + "/" + spec["track"]
            if key in data_identities and data_identities[key] != identity:
                raise ValueError("Incompatible cohort/aggregation in one native screen")
            data_identities[key] = identity
            run = str(Path(task["run_dir"]).resolve())
            row = dict(run_dir=run, spec=str(spec_path.resolve()),
                       spec_sha256=task["spec_sha256"], model=spec["model"]["name"],
                       track=spec["track"], disease=spec["disease"])
            if run in candidates and candidates[run] != row:
                raise ValueError("Conflicting aliases for one native execution")
            candidates[run] = row
    if not candidates:
        raise ValueError("No applicable native candidates")
    return dict(schema="look_native_screen_v1", created_at=utc_now(),
                screening_seed=3416, replication_seeds=[3416, 3417, 3418],
                selection="development_macro_f1_then_auroc_then_spec_sha256",
                test_access=False, data_identities=data_identities,
                source_queues=[dict(path=str(Path(q).resolve()),
                    sha256=file_sha256(Path(q))) for q in queues],
                candidates=sorted(candidates.values(), key=lambda r: r["run_dir"]))


def audit(catalog, verify_completion):
    """Verifier is the explicitly pinned independent trainer's full acceptor.

    Returns nominations, never project-dispatch permission. The project still
    requires full replay, participant/eye matching, three seeds and host acceptance.
    """
    if catalog.get("schema") != "look_native_screen_v1" or catalog.get("test_access") is not False:
        raise ValueError("Unknown or unsealed screen")
    groups = {f"{d}/{m}/{t}": [] for d in DISEASES for m in MODELS for t in TRACKS}
    for row in catalog["candidates"]:
        group = groups[f"{row['disease']}/{row['model']}/{row['track']}"]
        entry = dict(row)
        try:
            if file_sha256(Path(row["spec"])) != row["spec_sha256"]:
                raise ValueError("Frozen candidate changed")
            spec = json.loads(Path(row["spec"]).read_text())
            if (spec.get("test_used") is not False or spec["model"]["name"] != row["model"] or
                    spec["track"] != row["track"] or spec["disease"] != row["disease"] or
                    spec["training"]["seed"] != 3416):
                raise ValueError("Candidate identity mismatch")
            receipt_path = Path(row["run_dir"]) / "accepted.json"
            if not receipt_path.exists():
                entry["state"] = "awaiting_native_acceptance"
            else:
                verify_completion(row["run_dir"], spec)
                receipt = json.loads(receipt_path.read_text())
                metrics = receipt["metrics"]
                if not all(math.isfinite(metrics[k]) and 0 <= metrics[k] <= 1
                           for k in ("macro_f1", "auroc")):
                    raise ValueError("Undefined selection metric")
                entry.update(state="accepted", metrics=metrics,
                             accepted_sha256=file_sha256(receipt_path),
                             best_sha256=receipt["files"]["best.pt"])
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            entry.update(state="needs_review", reason=str(exc))
        group.append(entry)
    results = {}
    for key, rows in groups.items():
        closed = bool(rows) and all(r["state"] == "accepted" for r in rows)
        selected = (sorted(rows, key=lambda r: (-r["metrics"]["macro_f1"],
                    -r["metrics"]["auroc"], r["spec_sha256"], r["run_dir"]))[0] if closed else None)
        results[key] = dict(state="screen_closed" if closed else "waiting",
                            candidate_count=len(rows),
                            accepted_count=sum(r["state"] == "accepted" for r in rows),
                            selected=selected, candidates=rows)
    return dict(schema="look_native_readiness_v1", updated_at=utc_now(),
                catalog_sha256=stable_hash(catalog), groups=results,
                project_dispatch_ready=False, test_access=False,
                remaining_gates=["three_seed_parent_receipts", "project_owned_strict_replay",
                    "paired_participant_and_eye_acceptance", "fusion_host_training_acceptance",
                    "full_LOOK_correction_and_resource_acceptance"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(collect(args.queue), output / "catalog.json")
