# Pipeline Ledger

All steps are Linux entry points. Paths default to `project.json` and can be overridden.
Original UKB volumes are read-only; destructive flags apply only to derived data.

| Step | Host | Purpose | Principal Output | Reuse / Safety |
|---:|---|---|---|---|
| 1 | local/remote | Initialize source manifest and runtime roots | source/release manifests, empty roots | Standard-library bootstrap; atomic manifests; safe repeat |
| 2 | local | Push source or pull compact results | remote source mirror / local reports | `rsync`; excludes data, venv and caches |
| 3 | remote | Reuse/create environment, install current package and register kernel | lock, `look` kernelspec | Uses `venv_path` or `--venv-path`; validates imports and kernel target |
| 4 | remote | Verify image and label mounts | console audit | Requires expected UUIDs, files and read-only mounts |
| 5 | remote | Export CFP and middle OCT B-scans | field directories, export CSV | Per-file size check, `.partial`, atomic rename |
| 6 | remote | Compare export with source archives | verification text | Counts every field, partial and export error |
| 7 | remote | Audit strict CFP-OCT eye/time pairing | paired/unpaired CSVs, summary | Non-destructive deterministic rebuild |
| 8 | remote | Remove unpaired derived images | removal CSV | Dry-run, manifest allowlist, `--execute` |
| 9 | remote | Re-audit paired inventory | final pairing verification | Checks all references and source mount |
| 10 | remote | Extract all phenotype columns for matched EIDs | matched CSVs and manifest | Read-only source; atomic large-table writes |
| 11 | remote | Derive eye-level label candidates | candidate CSV and audit | Preserves ambiguous and multilabel status |
| 12 | remote | Build single-label five-class cohort | labels, mapping, flow, exclusions | Dry-run, fixed gates, `--execute` |
| 13 | remote | Verify final cohort end to end | `verification.json` | Labels, paths, hashes, counts and leakage |
| 14 | remote | Remove reproducible intermediates | final dataset-only tree | Dry-run, fixed top-level allowlist, `--execute` |
| 15 | remote | Run unit and governance tests | pytest result | Selected GPU visibility; no data mutation |
| 16 | remote | Smoke-test MHD fusion topologies and monitor nodes | smoke result | Seven fusion positions; first selected GPU |
| 17 | remote | Tiny dual-filling integration run | `runs/smoke/` | one- or two-GPU DDP; checkpoint resume |
| 18 | remote | Interactive complete study entry | deterministic experiment runs | One config cell; GPU list; validation/freeze/test |
| 19 | remote | Execute formal study grid | sweep plan/progress | Sequential configurations; DDP within training stage |
| 20 | remote/local | Aggregate matrix diagnostics | aggregate CSV | Reads completed runs; no model changes |

## State Machine

Reusable long-running components compute normalized configuration, input and code
fingerprints; acquire one process lock; verify completed outputs by size and SHA-256;
resume from progress or `last` checkpoint; write new files atomically; and quarantine
invalid artifacts. A changed fingerprint creates a distinct run ID. Existing valid
formal results are not overwritten.

Dataset pruning steps use a stricter contract because they intentionally change derived
data: inspect an allowlist, require `--execute`, then run the next verification gate.
If interruption leaves an inconsistent cohort, restore the affected derived stage from
the read-only source; never repair the source disk.

## Parameter Families

- Runtime: project, data, dataset, cache and runs roots.
- Source: image/label UUID, mount and root.
- Compute: physical GPU list, workers, global micro/effective batches and AMP. Per-device
  batches, ranks, samplers and `torchrun` launch are derived internally.
- Classifier: backbone, fusion stage, seed, epochs, patience, learning rates and decay.
- Filling: normalized mean or independently trained paired cGAN.
- Missingness: OCT missing, CFP missing and frozen-model missing ratios.
- LOOK: correction nodes, downsample factors, latent dimensions, ridge alpha grid,
  maximum PCA rank and primary validation metric.

Step 18 and Step 19 share `look_core.study_grid`; source code is never text-replaced to
create a run. The full resolved configuration is persisted with every experiment.
