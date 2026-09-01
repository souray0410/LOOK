# LOOK

LOOK studies linear latent correction for incomplete paired CFP-OCT classification.
An ImageNet-pretrained ResNet50 is first trained on complete pairs and then frozen.
Normalized-mean or independently trained paired-cGAN filling is followed by LOOK;
there is no missing-input classifier fine-tuning.

## Default Paths

All defaults are in `project.json`:

```text
project  /home/mengh/LOOK/2026_09_01_16_06_11
data     /data/mengh/LOOK/2026_09_01_16_06_11
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
cd /Users/mengh/Downloads/LOOK/2026_09_01_16_06_11/home
bash pipeline/2_sync_project_to_remote.sh
ssh ws
cd /home/mengh/LOOK/2026_09_01_16_06_11
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

No batch, `torchrun`, rank, sampler, or device setting changes with this list. Global
classifier and cGAN batch sizes stay fixed and are divided internally per GPU. If the
list is changed after PyTorch has already been imported in the notebook kernel, restart
that kernel and run all cells again.

The launcher disables NCCL P2P by default because the current `ws` GPU pair requires
it. A different server may explicitly set `NCCL_P2P_DISABLE=0`; notebook configuration
and the GPU-list interface remain unchanged.

The committed notebook opens in a safe single-configuration `dry_run` state
(`feature`, seed 3407, normalized mean). It cannot begin training until
`EXECUTION_MODE` is explicitly changed to `validation`. Expand the lists only after
the singleton plan and pilot have been checked.

Use these execution modes in order:

```text
dry_run     resolve IDs and save a plan without training
validation  train/resume, fit LOOK, select configurations, save validation results
freeze      hash the selected validation artifacts and emit a frozen manifest
test        load that manifest and evaluate the sealed test split without training/fitting
```

For command-line sweeps, Step 19 uses the same `StudyGrid` implementation:

```bash
PY=/home/mengh/LOOK/2026_08_30_11_20_47/tool/environment/.venv/bin/python
$PY pipeline/19_run_study_sweep.py --dry-run --phase validation --gpus 0,1
$PY pipeline/19_run_study_sweep.py --phase validation --gpus 0,1
$PY pipeline/19_run_study_sweep.py --phase freeze --gpus 0,1
$PY pipeline/19_run_study_sweep.py --phase test --gpus 0,1 \
  --frozen-manifest /data/mengh/LOOK/2026_09_01_16_06_11/runs/freezes/<freeze_id>/frozen_configuration_manifest.json
$PY pipeline/20_aggregate_matrix_analysis.py
```

## Resume And Outputs

The same normalized configuration resolves to the same deterministic run ID. A rerun
reuses complete artifacts and resumes classifier/cGAN epoch checkpoints. Changing a
scientific parameter or world size creates another ID and does not overwrite the old
run. Sweep progress is saved after every configuration.

```text
/data/mengh/LOOK/2026_09_01_16_06_11/
|-- cache/{pipeline_state,partial,quarantine}/
`-- runs/
    |-- backbones/<backbone_id>/       checkpoints, history, graph monitor, curves
    |-- generators/<generator_id>/     cGAN checkpoints, monitor, curves
    |-- experiments/<experiment_id>/   predictions, metrics, LOOK and matrix analysis
    |-- sweeps/<phase>__<plan_id>/      plan and progress
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
