# LOOK Structure

## Release Bundle

```text
2026_09_01_16_06_11/
|-- README.md
|-- release_manifest.json
|-- home/                         deploy contents as the project root
|   |-- project.json
|   |-- pyproject.toml
|   |-- README.md
|   |-- STRUCTURE.md
|   |-- PIPELINE_LEDGER.md
|   |-- file_manifest.json
|   |-- .vscode/settings.json      project-local interpreter defaults
|   |-- tool/
|   |   |-- look_core/            reusable experiment and recovery package
|   |   |-- MHD_Project/          MHD Framework V3 package
|   |   |-- tests/
|   |   |-- environment/
|   |   `-- research/
|   `-- pipeline/                 ordered executable Steps 1-20
`-- data/                          empty, redistributable runtime skeleton
    |-- data_manifest.example.json
    |-- dataset/
    |-- cache/{pipeline_state,partial,quarantine}/
    `-- runs/{backbones,generators,experiments,sweeps,freezes,smoke}/
```

The source package contains no UK Biobank records, images, model checkpoints, cache,
or virtual environment.

## Deployed Trees

```text
/home/<user>/LOOK/<release>/       source; installable and synchronized
|-- .vscode/settings.json          resolves the deployment-local interpreter
|-- project.json                   may reference a reusable environment
|-- tool/{look_core,MHD_Project}/
`-- pipeline/1_...20_...

/data/<user>/LOOK/<release>/       cache and runs; never synchronized as source
|-- cache/
|   |-- pipeline_state/            lock, cursor, fingerprints, output manifests
|   |-- partial/                   rebuildable incomplete products
|   `-- quarantine/                invalid or explicitly filtered derived products
`-- runs/
    |-- backbones/                 complete-modality checkpoints and monitors
    |-- generators/                independent cGAN checkpoints and monitors
    |-- experiments/               deterministic formal results and LOOK artifacts
    |-- sweeps/                    plans, per-case progress and summaries
    |-- freezes/                   hashed validation selections for sealed test
    `-- smoke/                     disposable integration products

/data/<user>/LOOK/<dataset-release>/dataset/
`-- validated images, labels, split and integrity manifests
```

Step 3 registers the same portable kernel name, `look`, on every host while its
machine-local kernelspec records the configured environment's absolute Python path.
Notebook source therefore remains portable without relying on a personal path.

## Data Evolution

Before Step 5, `dataset/` is empty. Steps 5-6 create and verify:

```text
dataset/
|-- 21015/                         left-eye CFP PNGs
|-- 21016/                         right-eye CFP PNGs
|-- 21017/                         left-eye upper-middle OCT B-scan PNGs
|-- 21018/                         right-eye upper-middle OCT B-scan PNGs
|-- ophthalmology_export_*.csv
`-- ophthalmology_verification.txt
```

Steps 7-9 add pairing reports and remove only manifest-listed unpaired derived images:

```text
dataset/pairing_reports/
|-- strict_paired_fundus_oct.csv
|-- strict_unpaired_fundus_oct.csv
|-- pairing_summary.json
|-- removed_unpaired_*.csv
`-- final_paired_verification.txt
```

Steps 10-11 retain all matched phenotype columns and derive auditable eye-level label
candidates without modifying source CSVs:

```text
dataset/labels/
|-- paired_image_eids.csv
|-- phenotype_match_manifest.json
|-- matched_raw/{ukb670300_matched.csv,ukb679947_matched.csv}
|-- metadata/
`-- curation/
    |-- ukb_eye_label_candidates.csv
    `-- ukb_eye_label_candidates.audit.json
```

Steps 12-14 build, verify, and clean the final training dataset:

```text
dataset/
|-- 21015/  21016/  21017/  21018/
|-- reference_labels.csv            one eye-level class and two image paths per row
|-- class_mapping.json              IDs, split method, reference-standard limitation
|-- cohort_flow.json                inclusion/exclusion and class/split counts
|-- cohort_exclusions.csv.gz        retained audit trail
`-- verification.json               final integrity gate
```

After cleanup, intermediate pairing and matched phenotype tables are reproducible from
the read-only sources and are absent from the training-facing dataset root.

## Experiment Evolution

```text
runs/backbones/<backbone_id>/
|-- {last,best}.pt, history.json
`-- monitor_history.jsonl, training_curves.{csv,png,pdf}

runs/generators/<generator_id>/<direction>/
|-- paired-cGAN checkpoints and history
`-- monitor_history.jsonl, training_curves.{csv,png,pdf}

runs/experiments/<experiment_id>/
|-- validation/test manifests, predictions and metrics
`-- look/<missing_pattern>/
    |-- correction artifacts and search history
    |-- matrix_analysis.{json,csv}
    |-- matrix_spectra/
    `-- matrix_summary.{png,pdf}

runs/sweeps/<phase>__<plan_id>/
|-- study_plan.json
`-- progress.json

runs/freezes/freeze__<freeze_id>/
`-- frozen_configuration_manifest.json
```

Fusion positions are `input`, `stem`, `layer1`, `layer2`, `layer3`, `layer4`, and
`feature`. MHD Framework file and class names remain `MHD_Framework_V3.py` and the
original scientific API; only the import package root is `MHD_Project`.
