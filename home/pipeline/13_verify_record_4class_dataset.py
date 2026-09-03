#!/usr/bin/env python3
"""Step 13: verify participant-level cohorts and publish the data manifest."""

from __future__ import annotations

import argparse
import json

from look_core.cli import add_runtime_arguments, resolve_runtime_arguments
from look_core.cohort import verify_four_class_cohorts
from look_core.reproducibility import sha256, write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    parser.add_argument("--skip-image-check", action="store_true")
    args = parser.parse_args()
    paths = resolve_runtime_arguments(args)
    report = verify_four_class_cohorts(
        paths.cohort_root,
        paths.image_root,
        check_images=not args.skip_image_check,
    )
    project = json.loads((paths.project_root / "project.json").read_text(encoding="utf-8"))
    candidate = paths.dataset_root / "phenotypes/record_phenotype_candidates.csv"
    paired = paths.dataset_root / "paired_eye_manifest.csv"
    data_manifest = {
        "schema_version": 3,
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
        "analysis_readiness": report["analysis_readiness"],
    }
    write_json_atomic(data_manifest, paths.data_root / "data_manifest.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
