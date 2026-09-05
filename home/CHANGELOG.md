# Idle graph parking during GAN workers

Operational release 2026_09_05_19_14_09. The unified launcher executes the original scientific entrypoint through an audited adapter. During paired-cGAN preparation only, the idle frozen graph is moved to CPU and restored after workers finish. Model weights, node states, RNG and logits passed GPU round-trip equivalence checks. Original batch size, optimizer, seeds, PCA and data roles are unchanged. This prevents the parent graph from competing with GAN workers for about 13 GiB of GPU memory; it is not an edge-deployment method claim. Use this release operations/start_unified_study.sh with the original --project-root and --execute.

# 2026-09-05: bounded method evidence

Release 2026_09_05_18_49_49. Adds SSF, independent-fit and missing-only controls: 9 cases after original 52 stages and suffix 243 cases. Original numerical kernels unchanged; test sealed. Protocol and comparator deviations: docs/METHOD_EVIDENCE.md. Advisor report is generated from completed, traceable validation results.

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
stopped and old results preserved. Screen all seven fusion positions at seed 3407,
select the top three by validation Macro-F1 with declared position order as the tie
breaker, then add seeds 3408/3409 only for selected positions. OCT-only and CFP-only
remain auxiliary three-seed controls outside fusion selection. Validation uses FP32,
training uses AMP. Run 27 main LOOK cases plus six ablations. Independent
GAN still uses training-internal reconstruction validation. Random masks now use shared
hash ordering, exact rounded participant counts, nested sets and fixed missing direction.
All factors and metrics are reported; both test cohorts remain sealed. Step 37 and the
old method descriptions are historical and do not govern this release.

Current specification: configs/unified_study.json; current entrypoint: pipeline/38_run_unified_study.py.
Technical evidence belongs under the new runtime runs/maintenance; runtime status requires
live verification. No earlier checkpoint, generator, PCA or experimental result is reused.

## 2026-09-04: restore staged fusion screening

The first Step 38 scheduler incorrectly expanded every fusion position to all three
seeds before selection. The intended state machine screens every position once at seed
3407 and then replicates only the selected three. The scheduler, plan, resume identity
and tests now enforce that order. The superseded study is stopped and removed under a
scoped cleanup audit; none of its outputs enter the replacement study.


## 2026-09-04: unimodal post-training evaluation repair

The shared LOOK input reset assumed both OCT and CFP input nodes existed. This
crashed after the first OCT-only training run completed, before validation_result
was written. Reset now writes only input nodes present in the graph. Training,
checkpoint selection, graph topology, residual fitting and statistical definitions
were unchanged in that repair. Its checkpoint belonged to the subsequently superseded
all-position three-seed attempt and is no longer a valid input to the current study.

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
