# LOOK 2026_09_04_10_49_20

Current Macro-F1 research release. Read [the handoff](home/HANDOFF.md) first.
`main` contains the current workflow; [archive/superseded-auroc-2026-09-03](https://github.com/souray0410/LOOK/tree/archive/superseded-auroc-2026-09-03)
preserves the superseded AUROC protocol for historical tracing only.

Complete-input layer3 backbones are trained at three seeds, selecting best
weights by full validation Macro-F1. Joint sequential optional LOOK selects its
sites and dimensions by the same endpoint. Every spatial factor and missing
scenario is reported. Test cohorts remain sealed. Controlled data are not included.

Deploy `home/` to `/home/mengh/LOOK/2026_09_04_10_49_20`; runtime outputs belong in
`/data/mengh/LOOK/2026_09_04_10_49_20`. `data/` is an empty runtime skeleton.
Existing data, cohort definitions, environment and preprocessing are shared via
explicit paths in `home/project.json`; no image copy is needed.

See [method and commands](home/JOINT_LOOK_PROTOCOL.md),
[fixed specification](home/configs/macro_f1_study.json),
[project standard](GENERAL_PROJECT_STANDARD.md) and [historical timeline](PROJECT_TIMELINE.md).
