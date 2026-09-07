# Current release: 2026_09_07_15_04_51 — flexible device execution

Pipeline 48 supports arbitrary explicitly selected single/multiple GPUs and live
drain/interrupt/add transitions. The existing study continues against its immutable
2026_09_07_11_18_41 worker source and unchanged scientific plan. The per-card 14 GiB
budget covers LOOK processes only. See [execution guide](home/docs/FLEXIBLE_GPU_EXECUTION.md).
No new scientific cases or test access are introduced.

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

# Historical release: 2026_09_07_11_03_06

Independent case parallelism: GPU 0 and GPU 1 each run one case, at most 14 GiB LOOK memory per card. No pending scientific jobs are added by this release. See home/docs/DUAL_GPU_QUEUE.md.

# Historical release: 2026_09_06_15_48_18

Bounded LOOK continuation: representative start evidence, existing 15+3 controls,
14 GiB/device resource supervision, cohort provenance/sensitivity audit, and
future MHD model-adapter design. See home/docs/ADVISOR_QUESTIONS.md and
home/docs/MHD_MODEL_ADAPTER_DESIGN.md. Historical runs are preserved.

# Runtime recovery: 2026_09_05_16_44_54

See [runtime recovery and validation](home/RUNTIME_RECOVERY_20260905.md). This operations-only update resumes the existing parent and supplement through explicit --project-root paths; it does not start a new scientific study. The isolated tests passed, but the intermittent NCCL hang was not deterministically reproduced.

# Correction-start supplement: 2026_09_05_09_12_55

The latest release adds the suffix-start study after the unchanged parent 52-stage queue. See [START_STUDY.md](home/START_STUDY.md). Parent paths and status snapshots below are historical; use the read-only monitors for current status.

# LOOK 2026_09_04_19_18_07

Current unified Macro-F1 research release. Read [the handoff](home/HANDOFF.md) first.
Screen seven fusion positions with seed 3407 by validation Macro-F1; replicate only
the selected layer3, feature and layer2 positions with seeds 3408 and 3409. Run all
three filling strategies with joint sequential LOOK, then ablations and six unimodal
controls (52 stages). Keep all factor/scenario results and negative findings.

Deploy home/ to /home/mengh/LOOK/2026_09_04_19_18_07; outputs belong in
/data/mengh/LOOK/2026_09_04_19_18_07. data/ is an empty runtime skeleton. Controlled
images, identities, checkpoints and runtime results are never committed to this repo.

See [fixed specification](home/configs/unified_study.json), [protocol](home/JOINT_LOOK_PROTOCOL.md),
[project standard](GENERAL_PROJECT_STANDARD.md) and [timeline](PROJECT_TIMELINE.md).
The paused previous release is preserved by snapshot/2026_09_04_10_49_20 and
2026_09_04_10_49_20. Older AUROC source remains in archive/superseded-auroc-2026-09-03.

### Simple-control supplement and advisor snapshots

Current method-evidence protocol v2 adds bias-only and train-fitted positive logit-affine controls (15 total method cases, 280 combined stages). See `home/docs/METHOD_EVIDENCE.md`. `home/pipeline/41_refresh_advisor_report.py` creates timestamped validation-only reports without waiting for the entire queue.
