# Project Structure

```text
2026_09_03_19_35_04/
|-- README.md
|-- GENERAL_PROJECT_STANDARD.md
|-- release_manifest.json
|-- data/
|   |-- dataset/
|   |-- cache/
|   |-- runs/
|   `-- data_manifest.example.json
`-- home/
    |-- README.md
    |-- CHANGELOG.md
    |-- PIPELINE_LEDGER.md
    |-- STRUCTURE.md
    |-- project.json
    |-- pyproject.toml
    |-- tool/
    |   |-- MHD_Project/
    |   |-- look_core/
    |   |-- operations/
    |   |-- research/
    |   `-- tests/
    `-- pipeline/
        |-- 1_initialize_layout.py
        |-- 2_sync_project_to_remote.sh
        |-- 3_create_environment.sh
        |-- 4_verify_source_mounts.sh
        |-- 5_export_ukb_ophthalmology.py
        |-- 6_verify_ophthalmology_export.py
        |-- 7_audit_ophthalmology_pairs.py
        |-- 8_remove_unpaired_ophthalmology.py
        |-- 9_verify_paired_ophthalmology.py
        |-- 10_extract_matched_phenotypes.py
        |-- 11_build_record_phenotypes.py
        |-- history/2026_09_03_08_30_00/  (superseded Steps 12-20)
        |-- 21_build_glaucoma_cohorts.py
        |-- 22_verify_glaucoma_cohorts.py
        |-- 23_clean_glaucoma_intermediates.py
        |-- 24_run_tests.sh
        |-- 25_run_graph_smoke.py
        |-- 26_run_pipeline_smoke.py
        |-- 27_UKB_LOOK_Glaucoma_ResNet50_MHD.ipynb
        |-- 28_run_study_sweep.py
        |-- 29_aggregate_matrix_analysis.py
        `-- 30_run_task_scout.py
```

Author runtime:

```text
/data/mengh/LOOK/2026_09_03_19_35_04/
|-- dataset/
|   |-- paired_eye_manifest.csv
|   |-- phenotypes/
|   `-- cohorts/
|       |-- ukb_record_glaucoma_binary_bilateral/
|       `-- task_scout/
|           |-- participant_split_manifest.csv
|           |-- <task-profile>/{primary,natural,incident}/
|           `-- task_bank_verification.json
|-- cache/
|   |-- pipeline_state/
|   |-- partial/
|   `-- quarantine/
`-- runs/
    |-- backbones/
    |-- baseline_selection/
    |-- experiments/
    |-- generators/
    |-- freezes/
    |-- smoke/
    |-- task_scout/
    `-- sweeps/
```

Images remain only under the explicit immutable `image_root`; preprocessing arrays
remain only under `preprocess_cache_root`.
