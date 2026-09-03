# LOOK 2026_09_03_19_35_04

Portable two-tree release for the selected UK Biobank bilateral CFP/OCT
all-evidence glaucoma benchmark and LOOK missing-modality study.
Controlled UK Biobank data are not distributed.

```text
2026_09_03_19_35_04/
|-- home/  source, configuration, reusable packages, and numbered pipeline
`-- data/  empty runtime skeleton for a new deployment
```

Author deployment:

```text
home/ -> /home/mengh/LOOK/2026_09_03_19_35_04
data/ -> /data/mengh/LOOK/2026_09_03_19_35_04
```

The new runtime stores cohort manifests, state, checkpoints, and analyses only. It
references the existing immutable image export and preprocessing cache through explicit
paths in `home/project.json`; no 346 GB image copy or symlink is created.

The validation-only task scout selected `glaucoma_all_evidence` (925 cases and 925
matched controls) for formal baseline qualification. Selection does not freeze a
baseline, approve LOOK, or open either sealed test cohort.

Path precedence is explicit CLI argument, `LOOK_*` environment variable, then the
author defaults in `project.json`. Other authorized users can reproduce the workflow
by overriding their project, data, source-image, phenotype, and cache paths.

See [home/README.md](home/README.md) for commands and
[GENERAL_PROJECT_STANDARD.md](GENERAL_PROJECT_STANDARD.md) for the reusable engineering
standard.
