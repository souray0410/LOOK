# General Reproducible Research Project Standard

This file is project-independent. Future projects may adopt it without inheriting LOOK,
UK Biobank, a particular account, server, model, or timestamp.

## 1. Source And Runtime

- Keep versioned source under `<home-root>/<project>/<release>`.
- Keep data, caches, checkpoints and results under `<data-root>/<project>/<release>`.
- Store a controlled dataset only once. New source iterations may adopt or move the
  verified data tree; never duplicate it merely to match a source timestamp.
- Keep controlled source media read-only. Derived manifests may reference media by
  relative path.
- Exclude datasets, secrets, virtual environments, caches and checkpoints from Git.

## 2. Portable Paths

- Centralize author defaults in one configuration file.
- Every executable accepts explicit roots relevant to its work.
- Resolve paths as CLI, then environment variable, then project default.
- Use package-relative paths internally. Do not add `sys.path` mutations or account-
  specific absolute paths in scientific modules.
- A different Linux user or server reproduces the workflow by supplying paths, without
  editing source.

## 3. Code Layout

- Number only ordered pipeline entries.
- Never renumber or silently delete a pipeline step that has been executed or released.
  Move superseded entrypoints into a timestamped, read-only `pipeline/history/` area and
  continue allocating numbers above the highest historical step.
- Keep reusable packages, model definitions, tests and operational helpers unnumbered.
- Put scientific logic in tested modules; notebooks and shell files are front ends.
- Use one canonical package name. Do not retain deprecated aliases, wrappers or symlinks.
- Keep framework/core code separate from project-specific model and task logic.

## 4. Artifact And Resume Contract

- Identify a run by normalized scientific configuration, input hashes and relevant code
  hashes, not by output path.
- Reuse complete artifacts only after schema, size and hash checks.
- Resume partial work from durable checkpoints. Generate missing components only.
- Write `.partial`, validate, then atomically rename.
- Quarantine corrupt or fingerprint-mismatched artifacts with a reason.
- Never silently overwrite a valid formal result.
- `--restart` affects only the declared stage/run; deletion additionally requires an
  explicit execution flag and an allowlisted root.

## 5. Experimental Separation

- Training fits model parameters and training-only transformations.
- Validation controls stopping, architecture, hyperparameters and method selection.
- Freeze and hash the reviewed configuration before opening test.
- Test never changes a model, threshold, preprocessing rule or correction matrix.
- A secondary natural-distribution test may complement a balanced primary evaluation,
  but neither may be used for model selection.
- A quality target is a review gate, not permission to tune on test or manipulate labels.

## 6. Training And Compute

- One explicit GPU list is the compute selector: one item means one GPU; multiple items
  mean one process per GPU under the declared distributed strategy.
- Record per-device batch, global batch, accumulation, precision, seeds, GPU model,
  CUDA, PyTorch and package lock.
- Verify initial model state across ranks and aggregate the complete validation split
  before checkpoint selection.
- Preserve model, optimizer, scheduler, scaler, epoch and best/last state for recovery.
- Verify that epoch-dependent dataset state reaches worker processes. Persistent workers
  must use shared state; otherwise deterministic augmentation can silently repeat across
  every epoch. Recreating workers is a correct but slower fallback.
- Account for every training row. If distributed sharding or `drop_last` excludes rows,
  report the count and justify it; retain an equal-sized final partial batch when safe.
- Treat collective timeout as a symptom; inspect the first failed rank or worker.
- Keep host-specific NCCL transport fallbacks explicit, recorded and environment-
  overridable; verify them first with a minimal native PyTorch DDP test.

## 7. Evidence And Monitoring

- Save epoch history, predictions, primary/secondary metrics, per-class metrics,
  confusion matrices, calibration, confidence intervals and manifests.
- Monitor named model/graph states, but compute validation criteria only from complete
  validation predictions.
- Detached jobs must survive client disconnect and expose progress, current config,
  validation evidence, GPU use, disk use and logs through one documented command.
- Preserve enough evidence to explain why a baseline was selected or rejected.

## 8. Acceptance Gates

Before a formal run:

1. Verify dataset schema, counts, references, duplicates and participant leakage.
2. Run unit tests, every supported graph topology, single/multi-GPU smoke, resume smoke,
   end-to-end tiny pipeline and deterministic grid planning.
3. Confirm test access is blocked before freeze.
4. Confirm local and remote source manifests match.
5. Confirm cleanup cannot target source data or paths outside the declared runtime.
6. Confirm documentation commands work from a fresh checkout with explicit paths.

Before an expensive architecture sweep, require a baseline qualification stage:

1. Demonstrate that a small subset can be intentionally overfit and that held-out metrics
   are independently reproduced from saved predictions.
2. Verify pretrained parameter mapping and compare framework gradients/updates with native
   autograd on a representative path.
3. Run unimodal controls before interpreting multimodal fusion gains.
4. Inspect train-validation gaps and evidence-defined subgroups to distinguish optimization
   failure from label/input mismatch.
5. Change one scientifically meaningful component at a time and invalidate prior artifacts
   through code/config fingerprints.

The active source tree contains current code only. Superseded entrypoints may remain in
an explicitly non-executable history area for provenance; compatibility imports, aliases,
wrappers and obsolete runtime branches remain forbidden.

## 9. Source-Control And Deployment Flow

- Treat the private remote Git repository as the canonical source of truth while a
  study is unpublished. A workstation is a deployment, not the only source archive.
- Make changes in a clean clone or worktree, whether temporary or persistent. Commit
  every accepted scientific or engineering change before deployment.
- Tag reproducible milestones with their release timestamp or declared study milestone.
- Push source, configuration, tests, documentation and empty data schemas to a private
  remote repository while a study is unpublished.
- Preserve each released timestamp as a `snapshot/<YYYY_MM_DD_HH_MM_SS>` branch and a
  matching immutable tag. Keep `main` on the latest reviewed release; snapshot branches
  are archival and must not be merged back into `main`.
- Never commit controlled data, extracted participant labels, secrets, environments,
  caches, checkpoints, predictions or runtime logs.
- Deploy committed source from the canonical repository to compute servers. If an urgent
  server-side edit is necessary, commit it on a branch and push it before treating the
  deployment as reviewed.
- Before a formal run, compare repository and compute-server identities by commit and
  source manifest. Compare a local clone too when one is being used.
- Finish each update as one transaction: edit in a clean clone/worktree, test,
  commit/tag, push, deploy, verify hashes, then start or resume computation.
- Local clones may be removed after push and independent fresh-clone verification. Their
  removal does not replace the need to back up controlled data and irreplaceable runtime
  artifacts outside Git.
