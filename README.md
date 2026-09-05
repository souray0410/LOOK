# Correction-start supplement: 2026_09_05_09_12_55

The latest release adds the suffix-start study after the unchanged parent 52-stage queue. See [START_STUDY.md](home/START_STUDY.md). Parent paths and status snapshots below are historical; use the read-only monitors for current status.

# LOOK 2026_09_04_19_18_07

Current unified Macro-F1 research release. Read [the handoff](home/HANDOFF.md) first.
Seven fusion positions and two unimodal controls are trained at three seeds; choose
three fusion positions by mean validation Macro-F1 and run all three filling strategies
with joint sequential LOOK. Keep all factor/scenario results and negative findings.

Deploy home/ to /home/mengh/LOOK/2026_09_04_19_18_07; outputs belong in
/data/mengh/LOOK/2026_09_04_19_18_07. data/ is an empty runtime skeleton. Controlled
images, identities, checkpoints and runtime results are never committed to this repo.

See [fixed specification](home/configs/unified_study.json), [protocol](home/JOINT_LOOK_PROTOCOL.md),
[project standard](GENERAL_PROJECT_STANDARD.md) and [timeline](PROJECT_TIMELINE.md).
The paused previous release is preserved by snapshot/2026_09_04_10_49_20 and
2026_09_04_10_49_20. Older AUROC source remains in archive/superseded-auroc-2026-09-03.
