# LOOK 1.0.0 Release Bundle

This repository is a two-tree Linux reproduction bundle. It contains source code and
an empty runtime schema; it does **not** redistribute UK Biobank data.

```text
2026_08_30_11_20_47/
|-- home/    copy these contents to a source root
`-- data/    copy these contents to a runtime-data root
```

The author deployment uses:

```text
home/ -> /home/mengh/LOOK/2026_08_30_11_20_47
data/ -> /data/mengh/LOOK/2026_08_30_11_20_47
```

All Linux paths, source mounts, disk UUIDs, SSH host, GPU index, and package index can
be overridden through documented command-line options. Defaults reproduce the author
environment, while code inside `home/` resolves project resources relative to
`project.json`.

Start with [`home/README.md`](home/README.md). It contains the complete numbered
rebuild, experiment, sealed-test, and analysis commands. The release acceptance gates
require read-only source data, resumable stages, manifest validation, single-GPU
execution, and leakage-free participant splits.

## Data access

UK Biobank data are controlled-access resources. Reproducers must obtain authorization
and provide their own source images and phenotype tables. The expected input schema and
field identifiers are documented, but no participant data are included here.

## Acceptance Gates

- `home/` and the deployed source tree have identical manifests.
- `data/` contains only a schema until authorized data are built or attached.
- Every completed stage verifies outputs before reuse and repairs incomplete products.
- Corrupt products are quarantined; valid completed products are never overwritten.
- No deprecated package alias, migration code, compatibility symlink, or hidden path
  setup is part of LOOK 1.0.0.
