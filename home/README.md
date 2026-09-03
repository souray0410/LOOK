# LOOK Four-Class CFP-OCT Study

## Scientific Contract

Primary classes are Normal, diabetes-related eye disease, Glaucoma, and macular
degeneration. Cataract is excluded. Available laterality-aware doctor-informed self-
report is a weak reference for method development, not expert retinal-image grading.

The primary cohort is balanced within the existing participant-level train/validation/
test split by retaining all disease rows and deterministically capping Normal to the
Glaucoma count. The secondary cohort preserves natural prevalence after excluding
Cataract. Validation selects every setting. Both tests remain sealed until freeze.

The baseline is conventional and deliberately transparent:

- two ImageNet V2 pretrained ResNet50 modality branches;
- linear concatenation projection plus normalization at one of seven fusion positions;
- unweighted cross-entropy, natural shuffle without replacement;
- one-stage end-to-end AdamW fine-tuning, cosine decay, five-epoch warm-up;
- at most 100 epochs with patience 15;
- one selected GPU or one DDP process per selected GPU.

`Macro-F1 >= 0.70` is a validation review gate. It is not a promised result. Below the
gate, the pipeline writes a label/image/model audit and refuses to freeze or start LOOK.

## Default Deployment

```text
project root  /home/mengh/LOOK/2026_09_03_08_30_00
data root     /data/mengh/LOOK/2026_09_03_08_30_00
image root    /data/mengh/LOOK/2026_09_03_08_30_00/dataset
cohort root   /data/mengh/LOOK/2026_09_03_08_30_00/dataset/cohorts/ukb_retinal_4class_weak
cache         /data/mengh/LOOK/2026_09_03_08_30_00/cache/preprocessed_pairs
environment   /home/mengh/LOOK/2026_08_30_11_20_47/tool/environment/.venv
```

There is one physical processed dataset and one preprocessing cache. Source code never
needs modification for another deployment. Every relevant entry accepts:

```text
--project-root --data-root --dataset-root --image-root --cohort-root
--labels-csv --natural-labels-csv --preprocess-cache-root --cache-root --runs-root
```

Equivalent environment variables are `LOOK_PROJECT_ROOT`, `LOOK_DATA_ROOT`,
`LOOK_DATASET_ROOT`, `LOOK_IMAGE_ROOT`, `LOOK_COHORT_ROOT`, `LOOK_LABELS_CSV`,
`LOOK_NATURAL_LABELS_CSV`, `LOOK_PREPROCESS_CACHE_ROOT`, `LOOK_CACHE_ROOT`, and
`LOOK_RUNS_ROOT`. Resolution is CLI, environment, then `project.json`.

## Current Author Run

On `ws`, the existing validated image collection and preprocessing cache have already
been adopted by the current data tree without copying. The following two adoption
commands document that completed one-time transition; they are not part of routine
training:

```bash
cd /home/mengh/LOOK/2026_09_03_08_30_00
source /home/mengh/LOOK/2026_08_30_11_20_47/tool/environment/.venv/bin/activate

python tool/operations/adopt_single_data_tree.py \
  --source-dataset /data/mengh/LOOK/2026_08_30_11_20_47/dataset \
  --source-preprocess-cache /data/mengh/LOOK/2026_09_02_00_00_00/cache/preprocessed_pairs

# Review the dry-run, then repeat with --execute.
python tool/operations/adopt_single_data_tree.py \
  --source-dataset /data/mengh/LOOK/2026_08_30_11_20_47/dataset \
  --source-preprocess-cache /data/mengh/LOOK/2026_09_02_00_00_00/cache/preprocessed_pairs \
  --execute

# Routine verification starts here.

python pipeline/12_build_balanced_4class_dataset.py --execute
python pipeline/13_verify_balanced_4class_dataset.py
bash pipeline/15_run_tests.sh --gpus 0,1
python pipeline/16_run_graph_smoke.py --gpus 0,1
python pipeline/17_run_pipeline_smoke.py --gpus 0,1
```

Start the complete baseline selection in a detached tmux session:

```bash
bash tool/operations/start_detached_validation.sh \
  --session look-fourclass-baseline --gpus 0,1 --mode baseline-selection
```

Monitor after disconnecting or closing the MacBook:

```bash
ssh ws
cd /home/mengh/LOOK/2026_09_03_08_30_00
bash tool/operations/check_detached_validation.sh \
  --session look-fourclass-baseline
```

Use `--follow` for live logs or `--attach` to enter tmux. Detach with `Ctrl-b`, then `d`.

## Baseline Selection

Step 19 `--mode baseline-selection` performs:

1. Feature-fusion calibration at seed 3407 for three pretrained/new-layer LR pairs and
   dropout 0.0/0.2, six configurations total.
2. Stage A comparison of seven linear fusion positions with the selected profile.
3. Stage B confirmation of the top three positions at seeds 3407, 3408 and 3409.
4. OCT-only and CFP-only references under the selected training profile.
5. Ranking by mean Macro-F1, balanced accuracy, Macro-AUROC, ECE and Macro-F1 stability.

The candidate is written below `runs/baseline_selection/candidates/`. If its three-seed
mean Macro-F1 is below 0.70, `quality_gate_audit.json` is generated and approval is
blocked. Passing still requires explicit scientific review:

```bash
python tool/operations/approve_baseline_candidate.py \
  --candidate /data/mengh/LOOK/2026_09_03_08_30_00/runs/baseline_selection/candidates/<candidate>.json \
  --reviewer-note "Reviewed labels, per-class errors, stability and modality baselines." \
  --execute
```

## LOOK Validation And Test

After baseline approval, run the frozen architecture/seeds with normalized-mean and
paired-cGAN filling:

```bash
python pipeline/19_run_study_sweep.py \
  --mode full-study --phase validation --gpus 0,1 \
  --baseline-selection-manifest /data/mengh/LOOK/2026_09_03_08_30_00/runs/baseline_selection/<approved>.json
```

Freeze completed validation artifacts:

```bash
python pipeline/19_run_study_sweep.py \
  --mode full-study --phase freeze --gpus 0,1 \
  --baseline-selection-manifest /data/mengh/LOOK/2026_09_03_08_30_00/runs/baseline_selection/<approved>.json
```

Open balanced and natural-distribution tests only with that frozen study manifest:

```bash
python pipeline/19_run_study_sweep.py \
  --mode full-study --phase test --gpus 0,1 \
  --baseline-selection-manifest /data/mengh/LOOK/2026_09_03_08_30_00/runs/baseline_selection/<approved>.json \
  --frozen-manifest /data/mengh/LOOK/2026_09_03_08_30_00/runs/freezes/<freeze>/frozen_configuration_manifest.json

python pipeline/20_aggregate_matrix_analysis.py
```

## Explicit Portable Example

Another Linux user can run the cohort builder without editing files:

```bash
python pipeline/12_build_balanced_4class_dataset.py \
  --project-root /srv/research/LOOK/home \
  --data-root /scratch/user/LOOK \
  --image-root /datasets/ukb/ophthalmology \
  --cohort-root /scratch/user/LOOK/dataset/cohorts/retinal4 \
  --labels-csv /scratch/user/LOOK/dataset/cohorts/retinal4/balanced/reference_labels.csv \
  --natural-labels-csv /scratch/user/LOOK/dataset/cohorts/retinal4/natural/reference_labels.csv \
  --preprocess-cache-root /scratch/user/LOOK/cache/preprocessed_pairs \
  --source-labels-csv /datasets/ukb/ophthalmology/reference_labels.csv \
  --execute
```

The same path arguments can be forwarded to Step 19 and the detached helper.

## Outputs And Recovery

- `runs/backbones/<backbone_id>/last/`: resumable trainer checkpoint.
- `runs/backbones/<backbone_id>/best/`: best trainer checkpoint.
- `runs/backbones/<backbone_id>/best.pt`: portable frozen graph state.
- `runs/backbones/<backbone_id>/history.json`: train and complete-validation evidence.
- `runs/sweeps/<plan>/progress.json`: current configuration and completed cases.
- `runs/experiments/<experiment_id>/`: predictions, filling, LOOK and metrics.
- `cache/pipeline_state/`: stage fingerprints, locks and completion manifests.
- `cache/quarantine/`: invalid artifacts with reasons.

Re-running an unchanged configuration validates and reuses completed artifacts. Partial
training resumes from `last/`. Scientific parameter or label changes create a new
deterministic ID. Test results report Macro-F1, weighted F1 (descriptive only), balanced
accuracy, Macro/per-class AUROC and AUPRC, sensitivity, specificity, ECE, confusion
matrices, and participant-bootstrap 95% confidence intervals.

The author server's two RTX 5000 Ada GPUs require the standard NCCL fallback
`NCCL_P2P_DISABLE=1`; launchers apply it only as a default. A deployment with a healthy
peer-to-peer path can explicitly set `NCCL_P2P_DISABLE=0`. This setting changes the
transport path, not the model, batches, gradients, or scientific configuration.

## Data Restriction

UK Biobank data are controlled access. This code release contains schemas and processing
logic only. Reproducers must obtain authorization and supply their own image/phenotype
sources. Original mounted media are never modified.
