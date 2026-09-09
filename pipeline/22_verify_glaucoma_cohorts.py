#!/usr/bin/env python3
"""Step 22: verify glaucoma and task-scout cohorts, then publish the data manifest."""

from __future__ import annotations

import argparse
import json

from look_core.cli import add_runtime_arguments, resolve_runtime_arguments
from look_core.cohort import verify_glaucoma_cohorts
from look_core.reproducibility import sha256, write_json_atomic
from look_core.task_scout import verify_task_cohort_bank


def main() -> None:
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    parser.add_argument("--skip-image-check", action="store_true")
    args = parser.parse_args()
    paths = resolve_runtime_arguments(args)
    report = verify_glaucoma_cohorts(
        paths.cohort_root,
        paths.image_root,
        check_images=not args.skip_image_check,
    )
    task_bank = verify_task_cohort_bank(
        paths.dataset_root / "cohorts" / "task_scout",
        paths.image_root,
        check_images=not args.skip_image_check,
    )
    project = json.loads((paths.project_root / "project.json").read_text(encoding="utf-8"))
    candidate = paths.dataset_root / "phenotypes" / "record_phenotype_candidates.csv"
    paired = paths.dataset_root / "paired_eye_manifest.csv"
    data_manifest = {
        "schema_version": 4,
        "release_id": project["release_id"],
        "status": report["status"],
        "reference_standard": report["reference_standard"],
        "unit_of_analysis": report["unit_of_analysis"],
        "data_root": str(paths.data_root),
        "dataset_root": str(paths.dataset_root),
        "image_root": str(paths.image_root),
        "cohort_root": str(paths.cohort_root),
        "preprocess_cache_root": str(paths.preprocess_cache_root),
        "paired_eye_manifest_sha256": sha256(paired),
        "phenotype_candidates_sha256": sha256(candidate),
        "cohorts": report["cohorts"],
        "matching_balance": report["matching_balance"],
        "analysis_readiness": report["analysis_readiness"],
        "task_scout": task_bank,
    }
    write_json_atomic(data_manifest, paths.data_root / "data_manifest.json")
    print(json.dumps({"glaucoma": report, "task_scout": task_bank}, indent=2))


if __name__ == "__main__":
    main()
