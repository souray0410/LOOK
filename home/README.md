# LOOK UK Biobank All-Evidence Glaucoma Benchmark

## Scientific Contract

The selected task is participant-level binary classification of `Normal` and `Glaucoma`.
Each sample contains left/right CFP and one central 2D OCT B-scan from the earliest
complete bilateral visit. The primary cohort uses all eligible prevalent record-derived
glaucoma evidence; it is not an expert image-grading gold standard.

The primary cohort contains 925 prevalent glaucoma cases and 925 deterministic 1:1
matched strict controls. Evidence may be pre-imaging HES ICD, glaucoma-specific
treatment/procedure, concordant self-report, or a single eligible self-report source.
Evidence provenance is retained for subgroup sensitivity analysis. A frozen-model
natural-prevalence test evaluates the selected definition without matched controls.

A prespecified validation-only task scout compared seven candidate definitions without
opening any test split. `glaucoma_all_evidence` ranked first among eligible profiles
(AUROC 0.6948, Macro-F1 0.6402; one seed, 12-epoch budget) and is now selected for a
full conventional baseline search. These scout values are not paper-level baseline or
test results.

## Current Reviewed LOOK Study (Step 34)

On 2026-09-04 the researcher approved proceeding with LOOK rather than continuing
backbone tuning. Select the original `layer3` configuration by the original primary
criterion, three-seed mean validation AUROC: 0.7835 (Macro-F1 0.6938). The regularized
feature candidate had AUROC 0.7760 (F1 0.7047), so it is not substituted based on F1.
The original automatic thresholds were not met. This is an explicit reviewed development
decision, not a passed gate, a clinical qualification, or an approval to access test.
No labels, splits, backbone parameters or existing checkpoints are changed.

Step 34 runs one fusion configuration, not three winning fusions. It reuses its three
existing seed checkpoints, verified by exact training IDs and hashes before execution:

1. Seed 3407: raw-zero and normalized-mean filling, each before/after LOOK.
2. Seeds 3408/3409: the same two filling strategies and LOOK search protocol.
3. All three seeds: independently trained paired cGAN filling, before/after LOOK.

There are nine outer cases. Each tests missing CFP and missing OCT, with missing ratios
20/40/60/80/100 percent. Inside each case, fit three complete artifact banks with shared
spatial factors `[4,8,16]`. At each node, select among latent dimensions
`[8,16,32,64,128,256]` on validation, using the existing sequential node-wise search.
This is not exhaustive search over every combination of node dimensions. The final
global factor is selected by full-bank validation AUROC, with larger factor breaking
exact ties. Vector nodes use identity compression. Dimension and factor lists can be
overridden explicitly; changing them starts new identified experiments, not overwrites.

All fitting data come from train. PCA now retains every training sample, including
the short final batch. Ridge GCV uses the exact residual SSE and an unpenalized
intercept in its effective degrees of freedom. No residual-strength alpha is used:
`z_corrected = z_missing + (z_missing W + b)`. Backbone training code and MHD V4 are
unchanged. GAN training is independent and uses an internal training-set holdout;
there is no missing-input backbone fine-tuning. Frozen backbone/LOOK inference is on
one GPU; GAN training uses the requested DDP devices. Low GPU utilization during CPU
PCA/GCV is expected, not evidence of a stalled DDP worker.

Validation is reused for development, including dimension/factor selection per seed;
seed repeats quantify initialization sensitivity, not independent test replication.
Report all cases, including negative LOOK effects. Test remains sealed until a separate
final configuration review. There is no metric-based stop between these nine cases;
technical failures stop visibly and the same command resumes validated artifacts.

```bash
bash tool/operations/start_reviewed_look.sh \
  --candidate /data/mengh/LOOK/2026_09_03_19_35_04/runs/baseline_selection/candidates/baseline_candidate__025a0ceb64d2.json \
  --reviewer-note 'Souray approved 2026-09-04: proceed with original AUROC-ranked layer3, all three seeds; retain failed automatic gates; validation only, no test.' \
  --gpus 0,1 --execute
python3 tool/operations/check_reviewed_look.py
```

Add `--follow` to follow the detached log. Summary and reviewed baseline evidence are in
`runs/reviewed_look/<fingerprint>/`. Per-case predictions, paired bootstrap, metrics and
matrix analyses are in `runs/experiments/<id>/`. Partial banks record candidate validation
scores and completed nodes; the monitor displays these before a full case finishes.
Same-command reruns validate/reuse checkpoints and completed results, then resume
unfinished banks. Original source media are never modified.

## Historical Bounded Follow-Up (Step 33)

Step 33 waits for the specified baseline session to finish successfully, then reads the
candidate from the exact confirmation plan. It never stops a running predecessor.
If the unchanged baseline gates fail, it evaluates six bounded regularization profiles:
weight decay `[1e-3, 1e-2]` crossed with label smoothing `[0, 0.05, 0.1]`. LR, dropout,
labels, architecture, split and seed remain fixed during this feature-fusion calibration.
If none improves the original calibration ranking, it reports failure and stops.
Otherwise the selected full profile (including weight decay) is used uniformly for
unimodal references, all seven fusion positions and Top-3 three-seed confirmation.

This is an exploratory follow-up on the same validation split, not independent evidence
or retrospective preregistration. All old results are retained. There is no automatic
threshold relaxation, label editing or indefinite tuning.

With `--auto-look-validation`, passing candidates receive explicitly machine-gated,
validation-pilot-only approval. The pilot uses seed 3407, raw-zero and normalized-mean
filling, then LOOK with global factors `[4,8,16]`. No GAN, formal test freeze, or sealed
test is scheduled by Step 33. A manual scientific review is still needed before final
publication claims and test access. There is an eight-hour soft budget: an active stage
finishes and saves checkpoints, but no subsequent stage starts after the deadline.

```bash
bash tool/operations/start_overnight_validation.sh \
  --confirmation-plan 9761ee03da42 --wait-session look-glaucoma-baseline \
  --gpus 0,1 --max-hours 8 --auto-look-validation --execute
python3 tool/operations/check_overnight_validation.py
```

Add `--follow` to monitor the log. The durable morning summary is
`runs/overnight/<fingerprint>/summary.json` (also `SUMMARY.md`); per-case training and
evaluation still use the standard runs directories. Rerun the same command to reuse
validated checkpoints/results. Extending `--max-hours` starts a new orchestration budget
while still reusing identical scientific configurations. Failure or interruption of the
predecessor is reported, never mistaken for completion. Runtime root overrides are
forwarded to Step 33; use `--runs-root` with the monitor for a custom runtime.

## Canonical Repository

The canonical source is the private GitHub repository
`https://github.com/souray0410/LOOK`. This directory is a compute deployment of a
reviewed commit. Update it from GitHub and verify `file_manifest.json` before starting
formal computation; do not treat uncommitted server edits as authoritative source.

## Default Paths

```text
project root      /home/mengh/LOOK/2026_09_03_19_35_04
data root         /data/mengh/LOOK/2026_09_03_19_35_04
image root        /data/mengh/LOOK/2026_09_03_08_30_00/dataset
cohort root       /data/mengh/LOOK/2026_09_03_19_35_04/dataset/cohorts/ukb_record_glaucoma_binary_bilateral
primary labels    /data/mengh/LOOK/2026_09_03_19_35_04/dataset/cohorts/task_scout/glaucoma_all_evidence/primary/reference_labels.csv
natural labels    /data/mengh/LOOK/2026_09_03_19_35_04/dataset/cohorts/task_scout/glaucoma_all_evidence/natural/reference_labels.csv
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

After all candidates finish, Step 31 writes `task_usability_summary.json` and
`task_usability_summary.md`. These combine sample size, phenotype definition, validation
signal and limitations; they deliberately do not select or freeze a task automatically.

The completed scout selected the all-evidence glaucoma profile. Record that reviewed
decision once with Step 32:

```bash
python pipeline/32_select_formal_task.py \
  --profile glaucoma_all_evidence --scout-id 8cc4833841be \
  --reviewer-note "Selected after reviewing sample size, phenotype provenance and validation-only task-scout performance." \
  --execute
```

This creates a small, hash-checked selection manifest. It neither trains a model nor
accesses test data.

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

The selected profile is now the default in `project.json`; Step 28 refuses a formal run
unless the Step 32 manifest points to the same label files and SHA-256 hashes. Start the
formal baseline search with no label-path overrides:

```bash
python pipeline/28_run_study_sweep.py --mode baseline-selection --phase validation \
  --gpus 0,1
```

For lid-independent execution, use the detached helper:

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

Step 27 is the interactive equivalent. Its default mode is `baseline_selection`; its only
configuration cell lists every scout and study axis. The selected label paths resolve
from `project.json`. A singleton list runs one configuration; longer lists expand
deterministically.
Completed artifacts are validated and reused, incomplete stages resume from `last/`,
and changed scientific parameters produce a new run ID.

## Publication Boundary

Validation selects all settings. Primary and natural tests remain sealed until the
baseline and LOOK configuration are frozen. Report AUROC, AUPRC, Macro-F1, balanced
accuracy, sensitivity, specificity, ECE, confusion matrices, participant bootstrap 95%
intervals, and paired LOOK-versus-fill comparisons. Claims are restricted to controlled
internal method validation because expert image grades and an external clinical test are
not available.
