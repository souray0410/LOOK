# LOOK

LOOK studies linear latent correction for incomplete paired CFP-OCT classification.
An ImageNet-pretrained ResNet50 is first trained on complete pairs and then frozen.
Normalized-mean or independently trained paired-cGAN filling is followed by LOOK;
there is no missing-input classifier fine-tuning.

This release uses MHD Framework V4. ResNet forward Feature Messages and backward
Gradient Messages share the declared hypergraph topology. Classifier training enters
the reverse computation exclusively through `MHD_Graph.backward(...)`; AMP scaling,
gradient accumulation, optimizer updates, and DDP synchronization retain their native
PyTorch semantics. The independently trained cGAN remains an ordinary PyTorch module
because it is a filling baseline rather than the MHD study backbone.

## Default Paths

All defaults are in `project.json`:

```text
project  /home/mengh/LOOK/2026_09_02_00_00_00
data     /data/mengh/LOOK/2026_09_02_00_00_00
dataset  /data/mengh/LOOK/2026_08_30_11_20_47/dataset
venv     /home/mengh/LOOK/2026_08_30_11_20_47/tool/environment/.venv
cache    <data>/cache
runs     <data>/runs
```

Thus only the validated dataset remains under the preceding timestamp. New
checkpoints, monitors, predictions, sweeps, and matrix analyses use the new timestamp.
For another host, pass `--project-root`, `--data-root`, `--dataset-root`,
`--cache-root`, and `--runs-root`, or set their `LOOK_*` environment variables.
Resolution is `CLI > environment > project.json`.

## Install And Kernel

From the Mac release root:

```bash
cd /Users/mengh/Downloads/LOOK/2026_09_02_00_00_00/home
bash pipeline/2_sync_project_to_remote.sh
ssh ws
cd /home/mengh/LOOK/2026_09_02_00_00_00
bash pipeline/3_create_environment.sh
```

Step 3 reuses the configured environment when it is complete, installs the current
LOOK package into it, verifies the scientific stack, and registers the Jupyter kernel
`LOOK`. Pass `--venv-path` only when another host uses a different environment. Open this `home/`
folder in the VS Code Remote window, open Step 18, and select `LOOK` if VS Code
does not select it automatically.

## Current Experiment Workflow

The validated dataset already exists, so Steps 4-14 do not need to be rerun for this
iteration. First verify the release:

```bash
bash pipeline/15_run_tests.sh --gpus 0,1
/home/mengh/LOOK/2026_08_30_11_20_47/tool/environment/.venv/bin/python pipeline/16_run_graph_smoke.py --gpus 0,1
/home/mengh/LOOK/2026_08_30_11_20_47/tool/environment/.venv/bin/python pipeline/17_run_pipeline_smoke.py --gpus 0,1
```

Step 18 is the interactive entry:

```text
pipeline/18_UKB_LOOK_ResNet50_MHD.ipynb
```

Edit its single configuration cell and run all cells. The only compute selector is:

```python
GPU_DEVICES = [0]     # one process on physical GPU 0
GPU_DEVICES = [1]     # one process on physical GPU 1
GPU_DEVICES = [0, 1]  # two-process DDP
```

No batch, `torchrun`, rank, sampler, or device setting changes with this list. The
per-device micro-batches remain fixed; gradient accumulation is derived from world size
so the classifier and cGAN effective global batches remain fixed. If the list is changed
after PyTorch has already been imported in the notebook kernel, restart that kernel and
run all cells again.

The notebook orchestrator creates graph metadata with a batch-one CPU graph and frees
it before DDP starts. During classifier and cGAN training, each selected GPU therefore
holds one equal-sized worker model; the parent kernel does not retain an extra model on
the first GPU. Post-training LOOK fitting and scenario evaluation are sequential frozen-
model stages and use the first selected GPU after the DDP workers have exited.

Before changing formal global batch size on a new GPU model, measure the worst-case
late-fusion training peak with `tool/operations/calibrate_ddp_batch.py` under `torchrun`.
Choose a batch with material headroom rather than the largest batch that merely avoids
OOM; record the measured per-rank peak with the experiment configuration.

The current RTX 5000 Ada two-GPU profile was measured with real forward/backward and
optimizer steps. The classifier uses effective global batch 256 and batch 128/GPU, with
16.609 GiB peak allocation for worst-case feature fusion. The baseline search compares
two conservative discriminative-learning-rate profiles: `3e-5/3e-4` and `1e-4/1e-3`
for ImageNet-pretrained/new parameters. The paired cGAN
uses effective global batch 448 (224/GPU), with 13.858 GiB peak allocation and 18.170 GiB CUDA
reservation; its Adam rate remains `2e-4` for adversarial stability.
Both formal loaders use 16 workers per rank on the current 128-CPU host. Live profiling
showed that increasing this to 32 workers per rank saturated all host CPUs and delayed
the first batch. Deterministic CFP ROI extraction plus CFP/OCT square-pad and resize
outputs are therefore cached losslessly under `cache/preprocessed_pairs/`; random paired
augmentation remains dynamic and seed/epoch controlled. The cache is derived,
version-keyed, atomically written, safe to delete, and never modifies source images.

The launcher disables NCCL P2P by default because the current `ws` GPU pair reports P2P
support but stalls on the P2P/CUMEM collective path. Shared-memory collectives completed
the same two-rank V4 check with identical state SHA-256, parameter fingerprints, and
Gradient Messages. A different server may explicitly set `NCCL_P2P_DISABLE=0`;
notebook configuration and the GPU-list interface remain unchanged.

For DDP startup, every rank constructs or restores the model on CPU from the same seed,
and every parameter plus persistent buffer is covered by a cross-rank SHA-256 check.
Only after equality is proven does LOOK disable PyTorch's redundant initial model
broadcast and move one model to each GPU. Gradient all-reduce remains enabled normally.

The committed notebook opens as the initial single-configuration `validation` pilot
(`feature`, seed 3407, normalized mean, two GPUs). Running all cells starts or resumes
that pilot. Expand the lists only after its outputs have been checked.

Use these execution modes in order:

```text
dry_run     resolve IDs and save a plan without training
validation  train/resume, fit LOOK, select configurations, save validation results
freeze      hash the selected validation artifacts and emit a frozen manifest
test        load that manifest and evaluate the sealed test split without training/fitting
```

For command-line sweeps, Step 19 uses the same `StudyGrid` implementation. Its default
`full-study` mode remains the complete filling/LOOK protocol:

```bash
PY=/home/mengh/LOOK/2026_08_30_11_20_47/tool/environment/.venv/bin/python
$PY pipeline/19_run_study_sweep.py --dry-run --phase validation --gpus 0,1
$PY pipeline/19_run_study_sweep.py --phase validation --gpus 0,1
$PY pipeline/19_run_study_sweep.py --phase freeze --gpus 0,1
$PY pipeline/19_run_study_sweep.py --phase test --gpus 0,1 \
  --frozen-manifest /data/mengh/LOOK/2026_09_02_00_00_00/runs/freezes/<freeze_id>/frozen_configuration_manifest.json
$PY pipeline/20_aggregate_matrix_analysis.py
```

For a run that must survive SSH disconnection or a sleeping client computer, start the
complete-modality baseline search in a remote `tmux` session:

```bash
./tool/operations/start_detached_baseline_search.sh
./tool/operations/check_detached_baseline_search.sh
./tool/operations/check_detached_baseline_search.sh --follow
```

This dedicated search runs 56 complete-pair classifier configurations on GPUs 0 and 1:
seven fusion positions, two discriminative learning-rate profiles, two classifier
dropouts, two label-smoothing values, and seed 3407. It uses natural sampling without
replacement and Balanced Softmax. The training loss adjusts each class logit by the
logarithm of its complete training-split count; inference uses the unadjusted logits.
There is no manually tuned class-balance beta. Fusion is
strictly `concatenate -> 1x1 Conv/Linear -> BatchNorm/LayerNorm`; the fusion operation
contains no activation, attention, or gate. Filling, cGAN, LOOK, random missingness, and
the sealed test split are not entered during this search. Attach with
`tmux attach -t look-baseline-search`; detach without stopping by pressing `Ctrl-b`, then
`d`. The 46 GiB deterministic preprocessing cache is reused rather than rebuilt.

The check command is intentionally diagnostic rather than merely a process heartbeat.
At any time it reports the exact active hyperparameters, epoch, train loss/accuracy,
complete validation metrics, per-class F1/sensitivity, confusion matrix, best epoch so
far, top-five completed configurations, grouped hyperparameter summaries, GPU load and
memory, disk space, and recent logs. The underlying evidence remains available after an
interruption in:

```text
runs/backbones/<backbone_id>/run_config.json
runs/backbones/<backbone_id>/history.json
runs/backbones/<backbone_id>/{last.pt,best.pt,training_complete.json}
runs/sweeps/validation__<plan_id>/{study_plan.json,progress.json}
runs/sweeps/validation__<plan_id>/{leaderboard.csv,baseline_search_results.json}
runs/sweeps/validation__<plan_id>/baseline_search_diagnostics.json
runs/logs/look-baseline-search_<timestamp>.log
```

The search can be resumed with the same start command. Completed valid configurations
are reused, an interrupted classifier resumes from `last.pt`, and a scientific or code
change creates a different content-fingerprinted run ID. After the 56 configurations,
inspect the validation ranking before freezing any design or running the full
filling/LOOK study.

The mandatory post-search selection and evaluation order is defined in
`tool/research/BASELINE_LOOK_EXPERIMENT_PROTOCOL.md`. Briefly, the 56-run single-seed
screen is followed by a three-candidate, three-seed stability confirmation. The winning
hyperparameter setting and all three seed-specific complete-modality checkpoints are
then frozen before any formal missing-modality/LOOK comparison. The sealed UKB test is
accessed only after those choices are fixed; external testing remains a separate evidence
requirement.

## Resume And Outputs

The same normalized configuration resolves to the same deterministic run ID. A rerun
reuses complete artifacts and resumes classifier/cGAN epoch checkpoints. Changing a
scientific parameter or world size creates another ID and does not overwrite the old
run. Sweep progress is saved after every configuration.

```text
/data/mengh/LOOK/2026_09_02_00_00_00/
|-- cache/{pipeline_state,partial,quarantine,preprocessed_pairs}/
`-- runs/
    |-- backbones/<backbone_id>/       checkpoints, history, graph monitor, curves
    |-- generators/<generator_id>/     cGAN checkpoints, monitor, curves
    |-- experiments/<experiment_id>/   predictions, metrics, LOOK and matrix analysis
    |-- sweeps/<phase>__<plan_id>/      plan, progress, ranking and diagnostics
    `-- freezes/freeze__<id>/           sealed configuration manifest
```

The MHD graph contains `label_gt`, differentiable `loss`, and diagnostic
`batch_accuracy` nodes. Formal results use full-split predictions and report macro,
weighted, and per-class F1; macro AUROC; balanced accuracy; sensitivity/specificity;
calibration; Brier score; kappa; confusion matrices; and participant-level uncertainty.
After every classifier epoch, rank zero prints the principal validation metrics and
atomically refreshes `training_curves.csv/png/pdf` plus
`validation_per_class.png/pdf` and `validation_confusion_matrix.png/pdf`; these files
remain inspectable during training and survive interruption. The JSONL history retains
the full validation record, including the confusion-matrix counts.

## Fresh Rebuild On Another Dataset

Authorized users can execute Steps 1-14 in order with explicit source and runtime
paths. Steps 8, 12, and 14 require a dry run followed by `--execute`. Source disks are
never modified. See `PIPELINE_LEDGER.md` for each step and `STRUCTURE.md` for structure
before and after processing.
