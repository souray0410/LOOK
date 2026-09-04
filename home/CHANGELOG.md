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

# Current Iteration

## 2026_09_03_19_35_04

- Replaced the weak four-class benchmark with a binary glaucoma task.
- Initially defined a 716-case high-confidence glaucoma profile, then compared it with
  six prespecified alternatives in a fixed validation-only task scout.
- Added deterministic 1:1 controls matched within participant splits by age, sex, and
  assessment centre, plus a natural-prevalence sensitivity cohort.
- Kept bilateral CFP and one central 2D OCT B-scan per eye; no 3D OCT is used.
- Changed bilateral aggregation from mean-plus-max to a parameter-free linear mean.
- Changed baseline ranking and checkpoint selection to complete-validation AUROC, with
  Macro-F1, sensitivity, specificity, calibration, and unimodal comparisons as gates.
- Added raw-zero as a distinct filling baseline beside normalized-mean and paired cGAN.
- Removed residual-strength alpha from LOOK. Ridge GCV lambda remains the sole
  correction regularization parameter.
- Preserved MHD Framework V4 unchanged and retained single- or two-GPU execution through
  one GPU-list setting.
- Preserved superseded Steps 12-20 in a timestamped, non-executable history area and
  continued active development at Steps 21-32.
- Added seven validation-only task candidates spanning evidence quality and sample size,
  with a fixed short feature-fusion scout before any formal baseline commitment.
- Added a validation-only UKB data/task usability summary and explicit parent-process
  CUDA cleanup between scout candidates.
- Selected `glaucoma_all_evidence` for formal baseline qualification: 925 cases and 925
  matched controls; scout AUROC 0.6948 and Macro-F1 0.6402 under the fixed short budget.
- Added a hash-checked Step 32 selection manifest. The decision does not freeze a
  baseline, approve LOOK, or access sealed test data.

## 2026-09-04: Step 36 joint LOOK

Joint input-through-layer3 correction, optional sequential acceptance, stable logit metrics,
shared per-channel PCA and atomic decision recovery replace the old protocol.
Preserve exact backbone training identity and checkpoints; add scoped cleanup and
real-checkpoint verification. See JOINT_LOOK_PROTOCOL.md and migration evidence.


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


## 2026-09-04: unimodal post-training evaluation repair

The shared LOOK input reset assumed both OCT and CFP input nodes existed. This
crashed after the first OCT-only training run completed, before validation_result
was written. Reset now writes only input nodes present in the graph. Training,
checkpoint selection, graph topology, residual fitting and statistical definitions
are unchanged. The existing verified checkpoint must be reused, not retrained.

Regression coverage now executes evaluate_missing for all seven fusion positions
and both unimodal controls, checks exact original-forward logits and a partial tail
batch, and checks that an unused modality cannot affect a unimodal model. A real
296-participant validation pass through ExperimentRunner also matches the saved
Macro-F1 exactly. Verification and scoped failure cleanup commands are in
tool/operations/verify_unimodal_evaluation.py and cleanup_failed_unimodal_evaluation.py.
The latter requires --execute, permits only the known failed attempt, preserves
training artifacts and writes a small audit before deletion.

The source-code fingerprint creates a new study ID inside the SAME release and
unchanged specification. The failed attempt is archived as a maintenance audit;
its incomplete evaluation/sweep/state directories are removed. The original
release tag remains immutable; the repair receives a distinct patch tag. Read the
latest live summary rather than relying on the initial study ID.
