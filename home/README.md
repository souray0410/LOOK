# LOOK UK Biobank Task Scout And Glaucoma Benchmark

## Scientific Contract

The task is participant-level binary classification of `Normal` and `Glaucoma`.
Each sample contains left/right CFP and one central 2D OCT B-scan from the earliest
complete bilateral visit. The primary cohort uses high-confidence record-derived
phenotypes; it is not an expert image-grading gold standard.

The primary cohort contains 716 prevalent glaucoma cases and 716 deterministic 1:1
matched strict controls. Cases require pre-imaging HES ICD evidence, glaucoma-specific
treatment/procedure evidence, or concordant 6148 and 20002 self-report. A frozen-model
natural-prevalence test provides sensitivity analysis for the broader UKB definition.

Before committing the full project to that task, this release also builds a prespecified
validation-only task bank. It tests evidence quality against available sample size without
opening any test split. The scout is for task selection, not a paper result.

## Default Paths

```text
project root      /home/mengh/LOOK/2026_09_03_19_35_04
data root         /data/mengh/LOOK/2026_09_03_19_35_04
image root        /data/mengh/LOOK/2026_09_03_08_30_00/dataset
cohort root       /data/mengh/LOOK/2026_09_03_19_35_04/dataset/cohorts/ukb_record_glaucoma_binary_bilateral
preprocess cache  /data/mengh/LOOK/2026_09_03_08_30_00/cache/preprocessed_pairs
environment       /home/mengh/LOOK/2026_08_30_11_20_47/tool/environment/.venv
```

Every path can be replaced through the matching CLI option or `LOOK_*` environment
variable. Source media remain read-only. The new timestamp does not copy images.

## Build The Cohorts

Activate the existing environment:

```bash
cd /home/mengh/LOOK/2026_09_03_19_35_04
source /home/mengh/LOOK/2026_08_30_11_20_47/tool/environment/.venv/bin/activate
```

Steps 1-11 remain the complete reconstruction path from authorized raw media. When
reusing the already verified image export, Steps 10-11 write only small metadata tables:

```bash
python pipeline/10_extract_matched_phenotypes.py \
  --paired-csv /data/mengh/LOOK/2026_09_03_08_30_00/dataset/paired_eye_manifest.csv
python pipeline/11_build_record_phenotypes.py
python pipeline/21_build_glaucoma_cohorts.py
python pipeline/22_verify_glaucoma_cohorts.py

python pipeline/23_clean_glaucoma_intermediates.py
python pipeline/23_clean_glaucoma_intermediates.py --execute
```

Step 21 builds both the primary glaucoma cohort and seven binary task candidates under one
global participant split. Step 22 checks evidence, 1:1 matching, SMD, image references,
and split leakage before writing `data_manifest.json`.

## Scout Candidate Tasks

The first four profiles span the main quality-volume trade-off and run first:

1. `glaucoma_high_confidence`: 716 cases; objective evidence or concordant self-report.
2. `glaucoma_all_evidence`: 925 cases; more data with greater record noise.
3. `any_target_eye_disease_high_confidence`: 1,052 pooled DR/glaucoma/AMD cases under
   the high-confidence rule.
4. `any_target_eye_disease`: 1,945 cases from all eligible prevalent target-disease
   evidence, including target comorbidity; this has the most data and heterogeneity.

Three diagnostic profiles follow: `glaucoma_objective_only` (236 cases),
`diabetic_eye_disease_all_evidence` (449 cases), and
`macular_degeneration_all_evidence` (436 cases). They expose whether cleaner evidence or
disease-specific appearance helps, but remain exploratory-only because they miss the
prespecified sample-size gate.

Run the short feature-fusion scout in detached mode:

```bash
bash tool/operations/start_detached_validation.sh \
  --session look-task-scout --gpus 0,1 --mode task-scout --skip-cache-warm
```

Monitor it with:

```bash
bash tool/operations/check_detached_validation.sh --session look-task-scout
```

This runs 12 epochs with patience 4, one fixed ImageNet V2 feature-fusion baseline and
seed 3407 per task. It ranks only from validation AUROC and Macro-F1. It does not run
test, LOOK, GAN, fusion-stage search, or three-seed confirmation.

## Verify Code And MHD

```bash
bash pipeline/24_run_tests.sh --gpus 0,1
python pipeline/25_run_graph_smoke.py --gpus 0,1
python pipeline/26_run_pipeline_smoke.py --gpus 0,1
```

These verify MHD V4 forward/backward behavior, ImageNet V2 mapping, binary metrics,
linear bilateral mean, DDP aggregation, raw-zero/normalized-mean/cGAN filling, LOOK
recovery, and sealed-test protection.

## Select The Formal Task And Baseline

Review `runs/task_scout/*/leaderboard.csv`, label provenance, case counts, SMD and
unimodal plausibility together. Do not automatically choose the numerically highest task.
The desired benchmark has a credible complete-modality baseline, useful signal in both
modalities, a reproducible missing-modality deficit, and enough headroom to measure
recovery. It is never weakened deliberately to make LOOK appear stronger.
After selecting an eligible profile, pass its two reference tables explicitly to every
formal Step 28 call, for example:

```bash
PROFILE=glaucoma_high_confidence
PRIMARY=/data/mengh/LOOK/2026_09_03_19_35_04/dataset/cohorts/task_scout/$PROFILE/primary/reference_labels.csv
NATURAL=/data/mengh/LOOK/2026_09_03_19_35_04/dataset/cohorts/task_scout/$PROFILE/natural/reference_labels.csv
python pipeline/28_run_study_sweep.py --mode baseline-selection --phase validation \
  --gpus 0,1 --labels-csv "$PRIMARY" --natural-labels-csv "$NATURAL"
```

The original high-confidence glaucoma default can still use the detached helper:

```bash
bash tool/operations/start_detached_validation.sh \
  --session look-glaucoma-baseline --gpus 0,1 --mode baseline-selection
```

The search runs six conventional LR/dropout profiles at feature fusion, OCT-only and
CFP-only references, seven fusion positions at seed 3407, and Top-3 confirmation at
seeds 3407, 3408, and 3409. Checkpoints use complete-validation AUROC.

Monitor after disconnecting or closing the MacBook:

```bash
ssh ws
cd /home/mengh/LOOK/2026_09_03_19_35_04
bash tool/operations/check_detached_validation.sh --session look-glaucoma-baseline
```

Use `--follow` for logs or `--attach` for tmux. Detach with `Ctrl-b`, then `d`.

A candidate requires mean AUROC at least 0.80, mean Macro-F1 at least 0.70, positive-class
sensitivity and specificity at least 0.65, and no AUROC deficit versus the best
unimodal model. Passing remains subject to explicit review:

```bash
python tool/operations/approve_baseline_candidate.py \
  --candidate /data/mengh/LOOK/2026_09_03_19_35_04/runs/baseline_selection/candidates/<candidate>.json \
  --reviewer-note "Reviewed phenotype provenance, errors, stability and modality references." \
  --execute
```

## Run LOOK

```bash
python pipeline/28_run_study_sweep.py --mode full-study --phase validation --gpus 0,1 \
  --baseline-selection-manifest <approved-baseline.json>

python pipeline/28_run_study_sweep.py --mode full-study --phase freeze --gpus 0,1 \
  --baseline-selection-manifest <approved-baseline.json>

python pipeline/28_run_study_sweep.py --mode full-study --phase test --gpus 0,1 \
  --baseline-selection-manifest <approved-baseline.json> \
  --frozen-manifest <frozen-study.json>

python pipeline/29_aggregate_matrix_analysis.py
```

Step 27 is the interactive equivalent. Its default mode is `task_scout`; its only
configuration cell lists every scout and study axis. After formal task selection, switch
the mode and pass the selected label paths through the documented path controls. A
singleton list runs one configuration; longer lists expand deterministically.
Completed artifacts are validated and reused, incomplete stages resume from `last/`,
and changed scientific parameters produce a new run ID.

## Publication Boundary

Validation selects all settings. Primary and natural tests remain sealed until the
baseline and LOOK configuration are frozen. Report AUROC, AUPRC, Macro-F1, balanced
accuracy, sensitivity, specificity, ECE, confusion matrices, participant bootstrap 95%
intervals, and paired LOOK-versus-fill comparisons. Claims are restricted to controlled
internal method validation because expert image grades and an external clinical test are
not available.
