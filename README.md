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
