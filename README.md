# LOOK 2026_09_03_08_30_00

Portable two-tree release for the UK Biobank CFP-OCT incomplete-modality study.
Controlled UK Biobank data are not distributed with this source bundle.

```text
2026_09_03_08_30_00/
|-- home/  source, configuration, numbered pipeline, reusable packages
`-- data/  empty runtime skeleton for an independent deployment
```

Author defaults:

```text
home/ -> /home/mengh/LOOK/2026_09_03_08_30_00
data/ -> /data/mengh/LOOK/2026_09_03_08_30_00
```

The author deployment keeps exactly one physical processed dataset and one preprocessing
cache under the current data tree. A new timestamp does not copy hundreds of gigabytes.
Other users may deploy both trees anywhere and pass explicit path arguments.

Path resolution is:

```text
explicit CLI argument > LOOK_* environment variable > home/project.json default
```

Every numbered entry supports the relevant subset of `--project-root`, `--data-root`,
`--dataset-root`, `--image-root`, `--cohort-root`, `--labels-csv`,
`--natural-labels-csv`, `--preprocess-cache-root`, `--cache-root`, and `--runs-root`.
Changing deployment paths never requires editing scientific source code.

Read [home/README.md](home/README.md) for exact commands and
[GENERAL_PROJECT_STANDARD.md](GENERAL_PROJECT_STANDARD.md) for the reusable engineering
standard. UK Biobank reproducers must obtain their own authorization and source data.
