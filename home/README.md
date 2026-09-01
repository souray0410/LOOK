# LOOK 1.0.0

LOOK studies linear latent correction for incomplete paired fundus photography (CFP)
and OCT classification. A frozen ImageNet-pretrained ResNet50 is first adapted using
complete paired data. Missing inputs are then handled without classifier fine-tuning:
normalized-mean filling or an independently trained paired cGAN is followed by LOOK.
MHD Framework V3 exposes the selected fusion-stage nodes for correction injection.

This is a strict single-GPU, sequential Linux workflow. UK Biobank data are not
included. The reference labels are derived from doctor-informed participant self-report
with eye laterality; they are **not** masked expert image-grading ground truth.

## Roots And Precedence

Author defaults are centralized in `project.json`:

```text
project root  /home/mengh/LOOK/2026_08_30_11_20_47
data root     /data/mengh/LOOK/2026_08_30_11_20_47
SSH alias     ws
image source  /mnt/ukbiobank/UKB
label source  /mnt/liverpool_ukbiobank/ukbb/UKB利物浦文件夹/csv
```

Resolution is `CLI > LOOK_* environment variable > project.json`. Python entries
support `--project-root`, `--data-root`, `--dataset-root`, `--cache-root`, and
`--runs-root`; source-data entries also accept source/label roots, mounts, and UUIDs.
Run any entry with `--help` for its exact interface.

For a custom Linux deployment:

```bash
export LOOK_PROJECT_ROOT=/srv/alice/LOOK/2026_08_30_11_20_47
export LOOK_DATA_ROOT=/scratch/alice/LOOK/2026_08_30_11_20_47
export LOOK_DATASET_ROOT=$LOOK_DATA_ROOT/dataset
export LOOK_CACHE_ROOT=$LOOK_DATA_ROOT/cache
export LOOK_RUNS_ROOT=$LOOK_DATA_ROOT/runs
```

Project modules and resources use relative paths. No compatibility package, symbolic
link, notebook path injection, or old project path is required.

## Author Deployment

From the release bundle on the Mac:

```bash
cd /Users/mengh/Downloads/LOOK/2026_08_30_11_20_47/home
python3 pipeline/1_initialize_layout.py --source-only
bash pipeline/2_sync_project_to_remote.sh
ssh ws
cd /home/mengh/LOOK/2026_08_30_11_20_47
bash pipeline/3_create_environment.sh
```

For another host, pass `--local-root`, `--host`, and `--remote-root` to Step 2. For an
offline deployment, copy the contents of the release `home/` and `data/` trees to the
chosen roots, then pass those roots explicitly.

## Fresh Data Rebuild

The source volumes must be mounted read-only. Commands below use the author defaults;
custom deployments add the path options described above.

```bash
python3 pipeline/1_initialize_layout.py
bash pipeline/4_verify_source_mounts.sh
tool/environment/.venv/bin/python pipeline/5_export_ukb_ophthalmology.py
tool/environment/.venv/bin/python pipeline/6_verify_ophthalmology_export.py
tool/environment/.venv/bin/python pipeline/7_audit_ophthalmology_pairs.py
tool/environment/.venv/bin/python pipeline/8_remove_unpaired_ophthalmology.py
tool/environment/.venv/bin/python pipeline/8_remove_unpaired_ophthalmology.py --execute
tool/environment/.venv/bin/python pipeline/9_verify_paired_ophthalmology.py
tool/environment/.venv/bin/python pipeline/10_extract_matched_phenotypes.py
tool/environment/.venv/bin/python pipeline/11_audit_eye_labels.py
tool/environment/.venv/bin/python pipeline/12_build_5class_dataset.py
tool/environment/.venv/bin/python pipeline/12_build_5class_dataset.py --execute
tool/environment/.venv/bin/python pipeline/13_verify_5class_dataset.py
tool/environment/.venv/bin/python pipeline/14_clean_intermediates.py
tool/environment/.venv/bin/python pipeline/14_clean_intermediates.py --execute
tool/environment/.venv/bin/python pipeline/13_verify_5class_dataset.py
```

Steps 8, 12, and 14 are intentionally two-stage: inspect the dry run first, then add
`--execute`. Original source files are never modified. OCT archives are decoded only
to export the upper-middle B-scan; the original 3D archives remain unchanged.

## Verification And Experiments

```bash
bash pipeline/15_run_tests.sh
tool/environment/.venv/bin/python pipeline/16_run_graph_smoke.py --gpu 0
tool/environment/.venv/bin/python pipeline/17_run_pipeline_smoke.py --gpu 0
tool/environment/.venv/bin/python pipeline/19_run_study_sweep.py --dry-run --gpu 0
tool/environment/.venv/bin/python pipeline/20_aggregate_matrix_analysis.py
```

Step 18 is the interactive scientific entry. Select
`tool/environment/.venv/bin/python` manually in VS Code/Jupyter, edit only its single
configuration cell, then run all cells. Scalar choices are represented as lists:
a singleton runs one choice; multiple values expand a deterministic Cartesian grid.

Independent axes are backbone, fusion position, seed, filling strategy, and named
classifier/GAN/LOOK profiles. LOOK profile fields include missing patterns/ratios,
correction nodes, downsample candidates, PCA latent dimensions, ridge alphas, maximum
rank, and primary validation metric. The test phase refuses to run unless the complete
configuration is frozen and `CONFIGURATION_FROZEN` is supplied.

## Recovery Contract

- Valid completed artifacts are verified by size and SHA-256 before reuse.
- Partial work uses `.partial`, progress state, epoch checkpoints, and process locks.
- A changed scientific configuration produces a new deterministic run directory.
- Damaged or fingerprint-mismatched runtime products are moved to `cache/quarantine/`.
- `--restart` is scoped to the selected experiment stage; resume is the default.
- Sweeps persist a plan and progress after every configuration.
- Source-data deletions are restricted to derived-data allowlists and explicit execution.

The reusable `look_core.batch_runner` replaces scripts that modify source text between
runs. `look_core.file_selection` replaces direct keyword deletion with a dry-run,
content-addressed allowlist and quarantine move.

## Expected Validated Cohort

```text
pairs                         162,202
images                        324,404
participants                   80,976
participant split leakage           0
missing or extraneous images        0
reference_labels.csv SHA-256  8b55005521a8cb426ad40fcb6dd417cfc50f71699554ac33be1880d207842911
```

Class counts are recorded in `cohort_flow.json` and rechecked by Step 13. The current
distribution is highly imbalanced; publication experiments must report per-class
metrics, macro metrics, uncertainty, and limitations in addition to overall accuracy.

## Documents

- `STRUCTURE.md`: release, deployed source, runtime, and stage-by-stage trees.
- `PIPELINE_LEDGER.md`: each entry's inputs, outputs, validation, and recovery unit.
- `tool/research/PUBLICATION_PROTOCOL.md`: leakage and reporting controls.
- `../README.md`: reusable project-management standard for this release bundle.

## Acceptance Gates

Release acceptance requires continuous Steps 1-20, clean canonical imports, shell and
Python syntax checks, all tests, graph and dual-filling smoke runs, deterministic sweep
planning, notebook validation, dataset verification, source-manifest equality, strict
single-GPU visibility, and confirmation that both source disks remain read-only.
