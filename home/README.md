# LOOK Bilateral UK Biobank Study

## Scientific Contract

The primary task is participant-level classification of `Normal`, diabetes-related eye
disease (`DR`), `Glaucoma`, and age-related macular degeneration (`AMD`). Each sample is
the earliest visit with complete left/right colour fundus photography (CFP) and central
OCT slices. Cataract, severe ocular trauma, retinal detachment/occlusion, other serious
eye disease, target-disease comorbidity, and temporally uncertain target records are
excluded from the primary cohort.

The labels are **record-derived clinical phenotypes**, not expert retinal-image grades.
They combine available UK Biobank self-report, hospital ICD records, diagnosis timing,
and procedure evidence. Evidence after imaging is stored in an incident cohort and never
used to define prevalent disease. Strict controls require an explicit no-eye-disease
response and no target diagnosis during available follow-up.

Data readiness is checked independently from model performance. Project-level minimums
are 2,000 balanced participants, 400 total and 250 training participants per disease,
and 60 participants per class in both validation and internal test. These are transparent
method-study safeguards, not TMI/MIA acceptance rules. The audit also reports per-class
95% interval precision. Without expert image grading and an independent external test,
the dataset cannot support clinical-deployment claims.

The primary development cohort retains every eligible disease participant and caps
Normal within each participant-level split to the largest disease class. Normal selection
matches the combined disease distribution by five-year age bin and sex. The natural-
prevalence cohort is a secondary evaluation set. Validation selects all settings; balanced
and natural test splits remain sealed until the study is frozen.

## Baseline Contract

- ImageNet V2 pretrained ResNet50 branches with shared weights across left/right eyes.
- Seven CFP/OCT fusion positions using linear projection plus normalization only.
- A parameter-free MHD bilateral-mean Edge creates one participant feature before the head.
- Unweighted cross-entropy, natural shuffle without replacement, AdamW and cosine decay.
- Five warm-up epochs, at most 100 epochs, patience 15, single GPU or DDP from one GPU list.
- OCT-only and CFP-only references use the same training profile.

The review gate requires three-seed balanced-validation mean Macro-F1 `>= 0.65`, every
mean class F1 `>= 0.45`, no evident training collapse, and multimodal performance no worse
than the best unimodal reference. `0.70` remains an aspirational target, not a label-
selection target. Failure generates an audit and blocks LOOK until scientific review.

## Default Deployment

```text
project root  /home/mengh/LOOK/2026_09_03_08_30_00
data root     /data/mengh/LOOK/2026_09_03_08_30_00
image root    /data/mengh/LOOK/2026_09_03_08_30_00/dataset
cohort root   /data/mengh/LOOK/2026_09_03_08_30_00/dataset/cohorts/ukb_record_prevalent_4class_bilateral
cache         /data/mengh/LOOK/2026_09_03_08_30_00/cache/preprocessed_pairs
environment   /home/mengh/LOOK/2026_08_30_11_20_47/tool/environment/.venv
```

Paths resolve in the order CLI, environment variable, then `project.json`. Supported
runtime overrides are `--project-root`, `--data-root`, `--dataset-root`, `--image-root`,
`--cohort-root`, `--labels-csv`, `--natural-labels-csv`, `--preprocess-cache-root`,
`--cache-root`, and `--runs-root`. Equivalent variables use the `LOOK_` prefix.

## Rebuild Data From Authorized Sources

Steps 1-9 export CFP and central OCT slices and enforce pairing. Existing verified images
do not need to be copied again. Starting from an existing paired image export, run:

```bash
cd /home/mengh/LOOK/2026_09_03_08_30_00
source /home/mengh/LOOK/2026_08_30_11_20_47/tool/environment/.venv/bin/activate

python pipeline/10_extract_matched_phenotypes.py
python pipeline/11_build_record_phenotypes.py
python pipeline/12_build_record_4class_dataset.py
python pipeline/13_verify_record_4class_dataset.py

# Destructive only for derived intermediates; inspect first, then execute.
python pipeline/14_clean_intermediates.py
python pipeline/14_clean_intermediates.py --execute
```

For the current one-time transition, Step 10 may explicitly read the existing image map:

```bash
python pipeline/10_extract_matched_phenotypes.py \
  --paired-csv /data/mengh/LOOK/2026_09_03_08_30_00/dataset/reference_labels.csv
```

Original mounted UKB media must remain read-only. The default phenotype source is
`ukb670300.csv`, which contains the complete selected field families needed at the first
two imaging instances; `ukb679947.csv` adds only later 5326/5327 instances for this field
set. Additional exports may be audited with repeated `--source`. Extraction stores only
selected columns and an evidence table; it does not copy the raw 35/51 GB CSV files.

## Verify Code And Graph

```bash
bash pipeline/15_run_tests.sh --gpus 0,1
python pipeline/16_run_graph_smoke.py --gpus 0,1
python pipeline/17_run_pipeline_smoke.py --gpus 0,1
```

The graph smoke checks all seven fusion positions plus both unimodal references. The
pipeline smoke checks DDP, checkpoint recovery, normalized-mean filling, paired cGAN,
LOOK correction, and sealed-test protection.

## Select The Baseline

```bash
bash tool/operations/start_detached_validation.sh \
  --session look-bilateral-baseline --gpus 0,1 --mode baseline-selection
```

Monitor after disconnecting or closing the MacBook:

```bash
ssh ws
cd /home/mengh/LOOK/2026_09_03_08_30_00
bash tool/operations/check_detached_validation.sh --session look-bilateral-baseline
```

Use `--follow` for live logs and `--attach` for tmux. Detach with `Ctrl-b`, then `d`.
Baseline selection performs six LR/dropout calibration runs, seven fusion runs at seed
3407, top-three confirmation at seeds 3407/3408/3409, and two unimodal references.

A passing candidate still requires explicit approval:

```bash
python tool/operations/approve_baseline_candidate.py \
  --candidate /data/mengh/LOOK/2026_09_03_08_30_00/runs/baseline_selection/candidates/<candidate>.json \
  --reviewer-note "Reviewed phenotype provenance, class errors, stability and modality references." \
  --execute
```

## LOOK Validation And Sealed Test

```bash
python pipeline/19_run_study_sweep.py --mode full-study --phase validation --gpus 0,1 \
  --baseline-selection-manifest <approved-baseline.json>

python pipeline/19_run_study_sweep.py --mode full-study --phase freeze --gpus 0,1 \
  --baseline-selection-manifest <approved-baseline.json>

python pipeline/19_run_study_sweep.py --mode full-study --phase test --gpus 0,1 \
  --baseline-selection-manifest <approved-baseline.json> \
  --frozen-manifest <frozen-study.json>

python pipeline/20_aggregate_matrix_analysis.py
```

LOOK validation compares normalized-mean and independently trained paired-cGAN filling,
missing CFP/OCT settings, random missing ratios, correction nodes, and global spatial
compression factors `[4, 8, 16]`. Sealed tests report Macro-F1, balanced accuracy,
Macro/per-class AUROC and AUPRC, sensitivity, specificity, ECE, confusion matrices,
participant bootstrap 95% confidence intervals, and paired corrected-vs-fill tests.

## Recovery

Completed artifacts are reused only after fingerprint and manifest validation. Training
resumes from `last/`, preserves `best/`, and writes each configuration before advancing.
Changing labels, fusion, LR, dropout, seed, filling, or LOOK parameters creates a new
deterministic run ID. The author server launcher defaults to `NCCL_P2P_DISABLE=1` for its
known transport constraint; this changes transport only, not the scientific computation.

## Data Restriction

UK Biobank data are controlled access. A public release contains code, schemas and
protocols only. Reproducers must obtain authorization and provide equivalent source fields.
