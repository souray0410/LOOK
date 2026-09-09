# Current release: 2026_09_07_16_28_16 — test after validation

Step 49 is an inference-only test follow-on. It waits for the unchanged validation
queue, freezes 87 comparison configurations, verifies all validation replays, and
then evaluates the existing 290-participant test. Natural test is withdrawn.
It inherits the current arbitrary-GPU policy and 14 GiB LOOK-only per-card limit.
No test is opened while validation is running. Results include all seeds, negative
findings, paired intervals and calibration metrics. See [test queue](docs/TEST_QUEUE.md).

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

# Correction-start supplement: 2026_09_05_09_12_55

The latest release adds the suffix-start study after the unchanged parent 52-stage queue. See [START_STUDY.md](START_STUDY.md). Parent paths and status snapshots below are historical; use the read-only monitors for current status.

# LOOK 2026_09_04_19_18_07

Read [HANDOFF.md](HANDOFF.md), [the fixed study specification](configs/unified_study.json)
and [method and operations](JOINT_LOOK_PROTOCOL.md). Step 38 first screens the seven
fusion positions at seed 3407 and selects the top three by validation Macro-F1. Only
those three positions receive seeds 3408 and 3409 before joint sequential LOOK with
mean, black-image and independent cGAN filling. OCT-only and CFP-only are auxiliary
controls and never enter fusion selection.

Main is current source. Previous fixed-layer3 results are retained as exploratory history;
their queue is paused. Data, splits, preprocessing, ImageNet normalization and MHD V4
are shared unchanged. Superseded all-position three-seed results are not inputs.
Tests stay sealed. Stage-wise weight streaming is not implemented.


## Self-input supplement

Release 2026_09_05_23_25_06: append exactly 3 self_input_missing_only cases after the unchanged 15-case method queue. Total 283 stages. Same levels and shared PCA, train-mean clamp before PCA, missing-only writeback, no new start search. See docs/SELF_INPUT_EVIDENCE.md. Test remains sealed.
