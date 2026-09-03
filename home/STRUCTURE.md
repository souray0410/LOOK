# Project Structure

```text
/home/<user>/LOOK/<timestamp>/
|-- project.json
|-- README.md
|-- STRUCTURE.md
|-- PIPELINE_LEDGER.md
|-- pyproject.toml
|-- pipeline/
|   |-- 1_initialize_layout.py
|   |-- 2_sync_project_to_remote.sh
|   |-- 3_create_environment.sh
|   |-- 4_verify_source_mounts.sh
|   |-- 5_export_ukb_ophthalmology.py
|   |-- 6_verify_ophthalmology_export.py
|   |-- 7_audit_ophthalmology_pairs.py
|   |-- 8_remove_unpaired_ophthalmology.py
|   |-- 9_verify_paired_ophthalmology.py
|   |-- 10_extract_matched_phenotypes.py
|   |-- 11_audit_eye_labels.py
|   |-- 12_build_balanced_4class_dataset.py
|   |-- 13_verify_balanced_4class_dataset.py
|   |-- 14_clean_intermediates.py
|   |-- 15_run_tests.sh
|   |-- 16_run_graph_smoke.py
|   |-- 17_run_pipeline_smoke.py
|   |-- 18_UKB_LOOK_ResNet50_MHD.ipynb
|   |-- 19_run_study_sweep.py
|   `-- 20_aggregate_matrix_analysis.py
`-- tool/
    |-- look_core/          reusable task and pipeline package
    |-- MHD_Project/       MHD Framework/Utils V4
    |-- tests/
    |-- environment/       requirements and environment lock, not the shared venv
    |-- operations/        detached launch/status and approval helpers
    `-- research/          protocol, slides, papers and compact result summaries
```

```text
/data/<user>/LOOK/<timestamp>/
|-- dataset/
|   |-- 21015/             left/right CFP exports
|   |-- 21016/
|   |-- 21017/             left/right central OCT slices
|   |-- 21018/
|   |-- reference_labels.csv       source five-class weak-reference table
|   `-- cohorts/ukb_retinal_4class_weak/
|       |-- cohort_manifest.json
|       |-- cohort_flow.json
|       |-- verification.json
|       |-- balanced/
|       |   |-- reference_labels.csv
|       |   `-- class_mapping.json
|       `-- natural/
|           |-- reference_labels.csv
|           `-- class_mapping.json
|-- cache/
|   |-- preprocessed_pairs/
|   |-- pipeline_state/
|   |-- partial/
|   `-- quarantine/
`-- runs/
    |-- backbones/
    |-- baseline_selection/
    |-- experiments/
    |-- generators/
    |-- sweeps/
    |-- freezes/
    |-- logs/
    `-- legacy_pilot/      compact provenance only
```

Scientific code refers to images by paths relative to `image_root`. `image_root`, cohort
CSV paths, cache and runs may live anywhere and are resolved through explicit arguments,
environment variables or `project.json`.

## Structure Transitions

```text
authorized read-only media
  -> Steps 5-9: paired CFP and central OCT export
  -> Steps 10-11: matched weak-reference eye labels
  -> Steps 12-13: balanced primary + natural secondary four-class manifests
  -> Steps 15-17: code, graph/DDP and tiny end-to-end verification
  -> Step 19 baseline selection: calibration -> fusion -> multi-seed -> review
  -> Step 19 formal study: filling -> LOOK -> freeze -> sealed tests
  -> Step 20: cross-experiment correction-matrix analysis
```

Data are stored physically once. A timestamp update changes source and run identity, not
the number of copies of the processed image collection.
