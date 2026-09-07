# 2026_09_07_11_18_41: explicitly resumed full start analysis

The user authorized completing the previously deferred remainder on 2026-09-07.
Pipeline 47 verifies and freezes 61 existing results, then queues exactly 182 new
start cases through the two independent GPU workers in pipeline 46. The four already
selected contexts are reused after verification; the remaining 23 selected-context
evaluations follow their nine starts. Those evaluations are not extra start fits.
Full scope: three frozen fusion positions × three fillings × three seeds × nine starts.
No backbone/PCA/generator retraining, new hyperparameters or test access.
Historical studies and failed/interrupted records are retained unchanged. The prior
empty pending plan describes the earlier idle state; this explicitly authorized
continuation uses configs/resume_full_starts.json and its generated frozen plan.

# 2026_09_07_11_03_06: independent case scheduling

The current default for future LOOK cases is pipeline/46_run_dual_gpu_queue.py:
one independent seed/case per physical GPU 0/1, up to two cases, 14 GiB LOOK memory
per card. The user replaced the previous single-case restriction. Case dependencies,
scientific parameters and sealed-test boundaries remain fixed. The compact supplement
is complete; configs/dual_gpu_pending.json is empty and deferred searches stay deferred.
See [dual-GPU execution contract](docs/DUAL_GPU_QUEUE.md).

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

# Operational import identity guard

Release 2026_09_05_19_20_22 preserves idle-graph parking and checks launcher PYTHONPATH against the requested immutable project. The running 19:14 adapter is numerically identical and may finish; use this version for future recovery.

# Idle graph parking during GAN workers

Operational release 2026_09_05_19_14_09. The unified launcher executes the original scientific entrypoint through an audited adapter. During paired-cGAN preparation only, the idle frozen graph is moved to CPU and restored after workers finish. Model weights, node states, RNG and logits passed GPU round-trip equivalence checks. Original batch size, optimizer, seeds, PCA and data roles are unchanged. This prevents the parent graph from competing with GAN workers for about 13 GiB of GPU memory; it is not an edge-deployment method claim. Use this release operations/start_unified_study.sh with the original --project-root and --execute.

# Current method-evidence supplement

Release 2026_09_05_18_49_49. Adds SSF, independent-fit and missing-only controls: 9 cases after original 52 stages and suffix 243 cases. Original numerical kernels unchanged; test sealed. Protocol and comparator deviations: docs/METHOD_EVIDENCE.md. Advisor report is generated from completed, traceable validation results.

# Runtime handoff — 2026-09-05

Latest observed parent state: failed at 27/52, raw_zero/feature/3409, at 06:27:56 UTC. DataLoader ancillary-descriptor transfer failed; the shell soft file limit is 1024. The supplement correctly exited without fitting any new case. Do not claim either queue is running. Parent recovery requires user authorization; source and accepted results remain intact. The new suffix launcher sets a 65536 open-file limit in its tmux child. See START_STUDY.md and refresh remote status before reporting.

# Correction-start supplement: 2026_09_05_09_12_55

The latest release adds the suffix-start study after the unchanged parent 52-stage queue. See [START_STUDY.md](START_STUDY.md). Parent paths and status snapshots below are historical; use the read-only monitors for current status.

# LOOK handoff — unified fusion and LOOK study

Current release `2026_09_04_19_18_07`. Respond in Chinese beginning “好的，Souray。”
All local shell commands use /tmp; only temporary local checkouts. GitHub main is
current source; ws02 is the workstation (ws remains an SSH alias). Read in order:

1. project.json and configs/unified_study.json.
2. JOINT_LOOK_PROTOCOL.md for the exact method, selection and missingness definitions.
3. PIPELINE_LEDGER.md Step 38, CHANGELOG.md and parent PROJECT_TIMELINE.md.
4. runs/maintenance technical evidence and the previous release pause audit.
5. Refresh tool/operations/check_unified_study.py and read the current summary/log.

Source: /home/mengh/LOOK/2026_09_04_19_18_07
Runtime: /data/mengh/LOOK/2026_09_04_19_18_07
Environment/data/preprocessing are explicit shared dependencies in project.json.
Results from superseded study `7eb81db4949c` are forbidden inputs. Previous release is
paused and retained as historical evidence.

Step 38 is the current formal entrypoint. Seven fusion positions are screened once at
seed 3407 with FP32 validation Macro-F1 checkpointing. Select the first three by F1,
using declared position order only for exact ties, then add seeds 3408 and 3409 for
those positions. OCT-only and CFP-only are auxiliary three-seed controls outside
selection. Then run 27 main
LOOK cases (three positions x three seeds x mean/black/independent cGAN) and six mean
seed3407 ablations. Preserve all factor/scenario metrics and negative results. GAN
selection uses training-internal reconstruction validation, never outer validation.
Both test cohorts stay sealed. No threshold tuning or missing-input fine-tuning.

Random masks are shared across every case, exact-count nested and direction-fixed.
LOOK fits only training residuals. Validation is used to select checkpoint/architecture/
LOOK decisions, so validation gains are exploratory and not an independent test claim.
Training AMP remains enabled; evaluation is FP32. MHD core remains unchanged.

Source deployment and tests must be verified before formal launch. Any runtime status
in this document is a setup description and must be refreshed live. Start/resume:
`bash tool/operations/start_unified_study.sh --gpus 0,1 --execute`.
Check: `python3 tool/operations/check_unified_study.py`.
There is no automatic status notification; Souray requests live status on demand.

Historical fusion comparisons used a different selection protocol and cannot justify
claiming layer3 is the Macro-F1-best fusion position. Earlier fixed-layer3 LOOK gains
remain exploratory evidence, not invalid solely because architecture selection changed.
Do not describe all earlier historical results as erroneous. Outputs from stopped study
`7eb81db4949c` are protocol-invalid and removed by scoped audit. Per-level weight streaming
is deferred. Every new technical failure blocks progress; negative efficacy does not.


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


## Latest supplement

Release 2026_09_05_23_25_06: append exactly 3 self_input_missing_only cases after the unchanged 15-case method queue. Total 283 stages. Same levels and shared PCA, train-mean clamp before PCA, missing-only writeback, no new start search. See docs/SELF_INPUT_EVIDENCE.md. Test remains sealed.
