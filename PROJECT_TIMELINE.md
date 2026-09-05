# 2026_09_05_16_44_54 — ws02 runtime recovery

Pin NCCL transport and file limits inside tmux children. Preserve both immutable study sources and all scientific configuration. Formal-sized isolated generator checks passed in both directions; 158 regression tests passed. See home/RUNTIME_RECOVERY_20260905.md for evidence and limits.

# 2026_09_05_09_12_55 — Correction-start supplement

Add pipeline 39: 243 suffix configurations, 30 strict source references and 213 fresh fits, queued after the original 52 stages. Validation chooses starts independently per missing direction; test remains sealed. Numerical LOOK, PCA, backbone and generator code are unchanged. New topology/decision provenance and consistent scientific reports distinguish eligible starts from enabled sites.

## 2026-09-04 — Step 37: Macro-F1 restart (current)

Release `2026_09_04_10_49_20` supersedes previous model/output protocols. Backbone
best epoch and early stopping now use full-validation Macro-F1; LOOK uses the
same primary endpoint. Fixed layer3 and three seeds are retrained from ImageNet,
without architecture search. New checkpoint identities and PCA banks prevent old
AUROC artifacts from being reused. All factors and random missing ratios are
reported independently. Test cohorts remain sealed. Steps 33–36 and their
launchers are archived on `archive/superseded-auroc-2026-09-03` at `383c3c8`.
The following older entries are historical and do not define current acceptance
or launch rules. See `JOINT_LOOK_PROTOCOL.md` in the home tree for current rules.

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

- 2026-09-04: Step 33 completed all six regularization profiles, two unimodal controls,
  seven fusions and Top-3 three-seed repeats. New feature AUROC/F1 was 0.7760/0.7047;
  original layer3 was 0.7835/0.6938. Neither passed all internal automatic gates.
- The researcher explicitly approved proceeding to LOOK with original layer3 under
  the existing AUROC selection rule, without additional backbone search. Step 34
  schedules zero/mean plus LOOK first, three-seed replication, then independent cGAN
  plus LOOK. One fusion configuration, nine outer cases, validation-only.
- Corrected PCA short-tail sample omission and the Ridge GCV residual/df calculation;
  no change to backbone training, MHD V4, dataset or existing checkpoints. Added
  numerical regression tests. No alpha, DCT or low-rank preprocessing is introduced.
- Later on 2026-09-04: replaced repeated per-missing-pattern PCA fitting with an
  explicit complete-training-feature PCA stage shared across missing directions,
  fillings and candidate dimensions. Dmax is independent (default 512); support
  both listed candidates and min/max/step search. Two quick seed-3407 diagnostic
  cases precede nine full cases. Step 35 records scoped deletion of the superseded
  partial LOOK run; original backbone checkpoints and all baseline results remain.

`2026_09_03_19_35_04` is active. The reviewed fixed-backbone LOOK development study is
the next stage. Failed thresholds remain recorded; they are not journal acceptance
criteria. Balanced and natural tests remain sealed pending final configuration review.

## Git Snapshot Map

Each timestamp above is preserved in the private repository as:

```text
snapshot/<YYYY_MM_DD_HH_MM_SS>
<YYYY_MM_DD_HH_MM_SS> tag
```

`main` always points to the latest reviewed source release. Snapshot branches are
read-only historical records and do not require pull requests or merging.

## 2026-09-04 — Joint LOOK protocol replacement

Souray approved replacing fusion-only mandatory correction with joint sequential
optional sites, retaining layer3 checkpoints and sealed tests. Step 36 schedules
9 main filling/seed cases and 2 ablations. Study e6d740a884be was stopped; scoped
cleanup preserves negative aggregates and all baseline/data/generator evidence.
See home/JOINT_LOOK_PROTOCOL.md and home/docs/joint_migration_evidence.json for
implementation, acceptance, deployment and live-state snapshot. Per-level weight
offloading was discussed as a possible implementation advantage, not implemented.


## Step 38 — 2026_09_04_19_18_07 unified protocol replacement

User requested pause of Step 37 and fresh consistent comparisons. Old processes were
stopped and old results preserved. Train all seven fusion positions and two unimodal
controls at seeds 3407/3408/3409. Checkpoint/architecture/LOOK endpoint is Macro-F1;
validation uses FP32, training AMP. Select top three fusion positions by three-seed mean,
then lower SD and declared order. Run 27 main LOOK cases plus six ablations. Independent
GAN still uses training-internal reconstruction validation. Random masks now use shared
hash ordering, exact rounded participant counts, nested sets and fixed missing direction.
All factors and metrics are reported; both test cohorts remain sealed. Step 37 and the
old method descriptions are historical and do not govern this release.

Current specification: configs/unified_study.json; current entrypoint: pipeline/38_run_unified_study.py.
Technical evidence belongs under the new runtime runs/maintenance; runtime status requires
live verification. No earlier checkpoint, generator, PCA or experimental result is reused.
