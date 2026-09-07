# Compact-start and frozen-prefix supplement (2026_09_07_10_44_19)

Pipeline 45 audits nine existing starts (1/4/7) with fusion fixed at layer3, then replays only the original accepted 3407/OCT-missing path. No refitting or test access. Prior 52 and 15+3 stages are complete; 182 historical starts stay deferred. See docs/COMPACT_START_PREFIX_PROTOCOL.md.


## 2026-09-06 bounded continuation (current)

Release 2026_09_06_15_48_18 contracts are documented in docs/ADVISOR_QUESTIONS.md.
The original 52-stage study is unchanged. The full 243-start queue is deferred
after a controlled cutover: retain every complete result, verify the 27
layer3/normalized_mean/three-seed starts, finish at most the cutover active case,
and then run the existing 15 method cases followed by three self-input cases.
Do not restart the historical full suffix or old waiting controllers.

Step 43 is the serial budget supervisor (GPU0/1 allowed, 14 GiB per GPU aggregate,
12 GiB torch allocator plus context margin). Step 44 produces a CPU-only cohort
provenance and validation sensitivity audit. Scientific configs remain pinned.

The legacy cuDNN TF32 setting causes measurable batch-dependent logits. Keep
original feature/evaluation batches; only graph allocation and SSF training
microbatches may shrink. Real-data acceptance reproduces both original full
validation missing-direction baseline logits exactly. Do not claim arbitrary
batch-size bitwise equivalence or silently change historical precision.

MHD_MODEL_ADAPTER_DESIGN.md is a future interface design, not an implemented
model migration. Additional architectures/data can change empirical conclusions.

Runtime pointer: /data/mengh/LOOK/maintenance/bounded_queue_current.json.

# Idle graph parking during GAN workers

Operational release 2026_09_05_19_14_09. The unified launcher executes the original scientific entrypoint through an audited adapter. During paired-cGAN preparation only, the idle frozen graph is moved to CPU and restored after workers finish. Model weights, node states, RNG and logits passed GPU round-trip equivalence checks. Original batch size, optimizer, seeds, PCA and data roles are unchanged. This prevents the parent graph from competing with GAN workers for about 13 GiB of GPU memory; it is not an edge-deployment method claim. Use this release operations/start_unified_study.sh with the original --project-root and --execute.

# Step 40: method evidence

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

# Pipeline Ledger

| Step | Purpose | Durable output | Recovery and integrity |
|---:|---|---|---|
| 1 | Initialize roots and manifests | runtime skeleton | create missing paths only |
| 2 | Sync source | remote source tree | excludes data, environments and runs |
| 3 | Verify/create environment and kernel | environment lock, LOOK kernel | reuses complete environment |
| 4 | Verify source mounts | source audit | requires read-only raw media |
| 5 | Export CFP and central OCT | four image directories | per-file resume; source unchanged |
| 6 | Verify export | image manifest | counts and image readability |
| 7 | Audit bilateral pairs | paired/unpaired reports | participant/visit/eye keys |
| 8 | Remove unpaired derived files | paired export | dry-run, allowlist, explicit execute |
| 9 | Verify paired images | paired verification | exact references and zero extras |
| 10 | Extract selected fields | paired visit map and matched fields | streaming, source hashes |
| 11 | Build record candidates | evidence-level candidate table | source, timing, code and exclusion reason |
| 12-20 | Historical four-class iteration | `pipeline/history/2026_09_03_08_30_00/` | superseded; provenance only, never active |
| 21 | Build glaucoma and task-bank cohorts | profile-specific primary/natural/incident tables | one global split, source fingerprint and deterministic matching |
| 22 | Verify glaucoma and task-bank cohorts | verification and data manifest | paths, classes, shared split, SMD and leakage |
| 23 | Clean partials | clean derived metadata tree | verification gate and explicit execute |
| 24 | Run unit tests | test report | blocks formal execution on failure |
| 25 | Run graph smoke | nine topology checks | binary shape, ImageNet mapping, forward/backward |
| 26 | Run pipeline smoke | three filling arms and LOOK | DDP, resume and sealed-test guard |
| 27 | Interactive experiment | standard runner artifacts | one configuration cell |
| 28 | Baseline/LOOK sweep | checkpoints, rankings and freezes | deterministic IDs and last/best resume |
| 29 | Matrix analysis | aggregate CSV and figures | selected LOOK banks only |
| 30 | Validation-only task scout | task leaderboard and resumable checkpoints | seven prespecified profiles, no test/LOOK access |
| 31 | Summarize task usability | JSON and Markdown audit | validation evidence, phenotype caveats and no automatic winner |
| 32 | Select formal task | hash-checked selected-task manifest | explicit review; validation-only evidence; test remains sealed |
| 33 | Bounded overnight continuation | `runs/overnight/<fingerprint>/summary.json`, grids and checkpoints | waits for successful predecessor; six regularization profiles at most; resume by scientific IDs; machine-gated validation pilot only |
| 34 | Reviewed fixed-backbone LOOK study | `runs/pca/<bank_id>/`, `runs/reviewed_look/<fingerprint>/`, 2 quick + 9 full cases | shared complete-train PCs first per stage; independent Dmax/list/range; exact backbone reuse; validation-only; failed gates retained; incremental PCA-entry and correction-bank resume |
| 35 | Clean superseded reviewed LOOK outputs | `runs/maintenance/cleanup_look__<id>.json` | maintenance only: stopped study manifest allowlist, dry-run then execute; preserve baselines/data/cache; rerun Step 34 afterward |

| 36 | Joint sequential optional LOOK | `runs/joint_look/<id>/`, 9 main cases + 2 ablations | frozen exact layer3 checkpoints; shared full-train PCA; logit ranking; atomic on/off decisions; both tests sealed; see JOINT_LOOK_PROTOCOL.md |

Steps 34 and 35 are retained historical entries; the active replacement is Step 36.

Pipeline numbers are append-only. Superseded executed steps move to the timestamped
history directory; active replacements receive new numbers and are never renumbered.

## Task Scout State Machine

1. Build all candidate labels from the same evidence table and global participant split.
2. Match strict controls 1:1 within each split, age, sex and assessment centre.
3. Run the four main quality-volume profiles first, followed by three diagnostic profiles.
4. Hold model, fusion position, seed, optimizer and short training budget fixed.
5. Rank validation evidence only; never open test and never run LOOK during scouting.
6. Review performance jointly with case count, evidence quality, balance and modality gain.
7. Promote an eligible task explicitly before the full baseline state machine begins.

The current reviewed selection is `glaucoma_all_evidence`: 925 prevalent cases and 925
matched controls, with train/validation/test case counts of 632/148/145. Step 32 records
the decision but does not qualify the baseline or authorize LOOK.

## Baseline State Machine

1. Verify the Step 32 task manifest and label hashes.
2. Calibrate three pretrained/new-layer LR pairs and dropout 0.0/0.2 at feature fusion.
3. Train OCT-only and CFP-only references under the selected profile.
4. Search seven linear fusion positions at seed 3407.
5. Confirm the top three at seeds 3407, 3408, and 3409.
6. Rank by mean AUROC, then Macro-F1, balanced accuracy, ECE, and AUROC stability.
7. Require AUROC 0.80, Macro-F1 0.70, sensitivity/specificity 0.65, and no multimodal
   AUROC deficit.
8. Require explicit approval before LOOK.
9. Freeze all validation-selected artifacts before any test access.

## Recovery Contract

- Existing files are reused only after configuration, code, input, and artifact checks.
- Long training resumes from `last/`; `best/` is preserved separately.
- One sweep case is committed before the next starts.
- Corrupt or mismatched outputs move to quarantine.
- A changed phenotype rule, labels file, model setting, filling strategy, or LOOK setting
  creates a new deterministic run ID.
- Validation and test predictions use separate paths.
- Natural and incident cohorts cannot influence model selection.
- Cleanup never targets `image_root`, `preprocess_cache_root`, or mounted source media.


## Step 38 — 2026_09_04_19_18_07 unified protocol replacement

User requested pause of Step 37 and fresh consistent comparisons. Old processes were
stopped and old results preserved. Screen all seven fusion positions at seed 3407,
select the top three by validation Macro-F1, then add seeds 3408/3409 only for those
positions. OCT-only and CFP-only remain auxiliary three-seed controls outside fusion
selection. Validation uses FP32 and training uses AMP. Run 27 main LOOK cases plus six ablations. Independent
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


## Step 42: self-input supplement

Release 2026_09_05_23_25_06: append exactly 3 self_input_missing_only cases after the unchanged 15-case method queue. Total 283 stages. Same levels and shared PCA, train-mean clamp before PCA, missing-only writeback, no new start search. See docs/SELF_INPUT_EVIDENCE.md. Test remains sealed.
