# LOOK 2026_09_03_19_35_04

Portable two-tree release for UK Biobank bilateral CFP/OCT task scouting and the
candidate glaucoma benchmark.
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

Path precedence is explicit CLI argument, `LOOK_*` environment variable, then the
author defaults in `project.json`. Other authorized users can reproduce the workflow
by overriding their project, data, source-image, phenotype, and cache paths.

See [home/README.md](home/README.md) for commands and
[GENERAL_PROJECT_STANDARD.md](GENERAL_PROJECT_STANDARD.md) for the reusable engineering
standard.
