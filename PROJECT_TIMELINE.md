# LOOK Project Timeline

This document summarizes the purpose, principal changes, observed outcome and disposition
of each timestamped LOOK source snapshot. Detailed reproduction instructions remain in
the corresponding snapshot.

## 2026_08_30_11_20_47

- Established the first reproducible LOOK project layout around MHD Framework V3.
- Exported paired UK Biobank color fundus photographs and central 2D OCT slices from
  read-only source media while preserving relative structure.
- Matched images to phenotype records, removed unpaired observations and built the first
  single-label five-class weak-reference table.
- Implemented the initial paired CFP/OCT ResNet50, MHD and LOOK notebook. Its verified
  processed image collection became the single physical dataset reused later.

## 2026_09_01_16_06_11

- Extended execution from one GPU to an explicit GPU list, with DDP for multiple devices.
- Added graph-visible monitoring, complete-validation metrics, persistent checkpoints,
  resumable grids and detached server execution.
- Kept the existing dataset and environment reusable through explicit paths.

## 2026_09_02_00_00_00

- Migrated the model to MHD Framework V4 with explicit graph backward levels while
  retaining PyTorch optimizer updates.
- Verified ImageNet V2 ResNet50 weights in active MHD hyperedges and kept fusion linear.
- Refined Trainer/Monitor responsibilities for full-validation criteria and DDP.
- Explored the imbalanced five-class task and cRT; best Macro-F1 remained about 0.315,
  motivating cohort and label review. Large superseded checkpoints were removed.

## 2026_09_03_08_30_00

- Defined a participant-level bilateral four-class record-derived phenotype task:
  Normal, diabetes-related eye disease, Glaucoma and AMD.
- Combined available self-report, HES/ICD timing, diagnosis age and procedure evidence;
  post-imaging diagnoses were isolated as incident cases.
- Built a 2,735-participant balanced cohort and a 71,938-participant natural cohort with
  complete bilateral CFP/OCT and zero participant leakage.
- Repaired persistent-worker augmentation state and DDP-safe final-batch accounting.
- The baseline still overfit while validation Macro-F1 remained about 0.33-0.50. The
  image/record phenotype mismatch made this task unsuitable as the main LOOK benchmark.

## 2026_09_03_19_35_04

- Reframed the benchmark as binary record-derived glaucoma classification using bilateral
  CFP and bilateral central 2D OCT B-scans.
- Added seven prespecified validation-only task profiles under one fixed split and short
  protocol to assess UKB task usability without touching sealed tests.
- `glaucoma_all_evidence` ranked first among eligible profiles: 925 cases, 925 matched
  controls, validation AUROC 0.6948 and Macro-F1 0.6402 in the task scout.
- Recorded the reviewed task decision in append-only Step 32 and updated downstream
  defaults without freezing a baseline or opening test data.
- Began full baseline qualification: six LR/dropout profiles, OCT-only and CFP-only
  references, seven linear fusion stages and Top-3 three-seed confirmation.
- Removed LOOK residual-strength alpha. LOOK remains direct residual correction with
  global spatial factors 4, 8 and 16, PCA latent correction and Ridge lambda selected
  without missing-input backbone fine-tuning.

## Current Direction

- Added Step 33 as a bounded overnight continuation: six regularization profiles only
  when baseline gates fail, uniform stage comparison if improved, and an optional
  single-seed zero/mean LOOK validation pilot if unchanged gates pass. No test access.

`2026_09_03_19_35_04` is active. Task selection is complete and formal baseline
qualification is running. LOOK starts only after explicit review and baseline freeze;
balanced and natural sealed tests remain unavailable until then.

## Git Snapshot Map

Each timestamp above is preserved in the private repository as:

```text
snapshot/<YYYY_MM_DD_HH_MM_SS>
<YYYY_MM_DD_HH_MM_SS> tag
```

`main` always points to the latest reviewed source release. Snapshot branches are
read-only historical records and do not require pull requests or merging.
