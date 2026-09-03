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
|   |-- 11_build_record_phenotypes.py
|   |-- 12_build_record_4class_dataset.py
|   |-- 13_verify_record_4class_dataset.py
|   |-- 14_clean_intermediates.py
|   |-- 15_run_tests.sh
|   |-- 16_run_graph_smoke.py
|   |-- 17_run_pipeline_smoke.py
|   |-- 18_UKB_LOOK_ResNet50_MHD.ipynb
|   |-- 19_run_study_sweep.py
|   `-- 20_aggregate_matrix_analysis.py
`-- tool/
    |-- look_core/          reusable phenotype, data, graph and experiment package
    |-- MHD_Project/       MHD Framework and Utils V4
    |-- tests/             unit, topology, DDP and recovery tests
    |-- environment/       requirements and lock; shared venv is external
    |-- operations/        launch, status, cache and approval helpers
    `-- research/          preregistered protocols and compact study records
```

```text
/data/<user>/LOOK/<timestamp>/
|-- dataset/
|   |-- 21015/  21016/                 left/right CFP
|   |-- 21017/  21018/                 left/right central OCT slices
|   |-- paired_eye_manifest.csv         earliest complete bilateral visit
|   |-- phenotypes/
|   |   |-- extraction_manifest.json
|   |   |-- matched_fields/             selected source fields only
|   |   |-- record_phenotype_candidates.csv
|   |   `-- record_phenotype_candidates.audit.json
|   `-- cohorts/ukb_record_prevalent_4class_bilateral/
|       |-- cohort_manifest.json
|       |-- cohort_flow.json
|       |-- cohort_characteristics.csv
|       |-- evidence_source_counts.json
|       |-- analysis_readiness.json
|       |-- verification.json
|       |-- balanced/{reference_labels.csv,class_mapping.json}
|       |-- natural/{reference_labels.csv,class_mapping.json}
|       `-- incident/{reference_labels.csv,class_mapping.json}
|-- cache/
|   |-- preprocessed_pairs/
|   |-- pipeline_state/
|   |-- partial/
|   `-- quarantine/
|-- runs/
|   |-- backbones/
|   |-- baseline_selection/
|   |-- experiments/
|   |-- generators/
|   |-- sweeps/
|   |-- freezes/
|   `-- logs/
`-- data_manifest.json
```

## Data Flow

```text
authorized read-only image media
  -> Steps 5-9: CFP export + central OCT extraction + strict eye pairing
  -> Step 10: earliest complete bilateral visit + selected phenotype fields
  -> Step 11: prevalent/incident/uncertain evidence classification
  -> Steps 12-13: balanced development + natural secondary + incident manifests
  -> Steps 15-17: unit, MHD graph/DDP and tiny end-to-end verification
  -> Step 19 baseline: calibration -> unimodal controls -> seven fusions -> three seeds
  -> Step 19 LOOK: filling -> correction -> freeze -> sealed tests
  -> Step 20: correction-matrix and cross-experiment analysis
```

Images exist once. Cohorts are CSV references. Internal imports are relative to the
installed source tree; deployment paths are supplied through CLI, environment variables,
or `project.json`.
