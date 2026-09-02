# General Reproducible Research Project Standard

This document defines the project-management and execution contract shared by future
research repositories. Domain-specific methods, datasets, models, and metrics belong in
each project's own protocol; the rules below are intentionally general.

## 1. Release Layout

Use one immutable timestamp or release identifier and separate source from runtime data:

```text
<project>/<release>/
|-- README.md
|-- project.json
|-- tool/       reusable packages, tests, environment and research protocols
`-- pipeline/   numbered executable stages

<runtime>/<project>/<release>/
|-- dataset/    validated derived dataset or an explicit external dataset pointer
|-- cache/      reproducible intermediate artifacts and stage state
`-- runs/       checkpoints, predictions, metrics, sweeps and frozen manifests
```

The source tree must remain small enough to version and distribute. Controlled data,
virtual environments, model checkpoints, caches and credentials must not enter Git.

## 2. Paths And Portability

- Put author-friendly defaults in one configuration object only.
- Every executable accepts explicit project, dataset, cache and runs roots.
- Resolve values in this order: CLI argument, environment variable, project default.
- Internal modules and resources use package-relative paths, never user-specific paths.
- The same release must run under another Linux account or server by changing paths,
  without editing scientific source code.
- Source datasets are read-only. Derived outputs must stay inside the declared runtime
  root, with boundary checks before copying or deleting.

## 3. Code Organization

- Number only stages that users execute in order: `1_...`, `2_...`, and so on.
- Keep reusable libraries, operations, tests and model definitions unnumbered.
- Entry scripts parse arguments and call library functions; scientific logic does not
  live in shell scripts or notebook-only cells.
- Use one canonical package name and direct imports. Do not retain deprecated aliases,
  compatibility symlinks or `sys.path` mutations in a clean release.
- Notebooks are transparent interactive front ends to the same tested library used by
  command-line sweeps.

## 4. Artifact And Resume Contract

- A stage checks artifact schema, size, count and fingerprint before reuse.
- Complete valid artifacts are reused; partial valid work resumes from its last durable
  cursor or checkpoint; missing pieces are generated individually.
- Write new files to `.partial`, validate them, then atomically rename them.
- Quarantine corrupt, truncated or fingerprint-mismatched artifacts with a reason.
- Never silently overwrite a complete formal result.
- Long stages use a process lock and persist progress after every recoverable unit.
- A clean restart deletes only the selected run/state scope and never source data.
- Resume tests must use a new process or newly constructed trainer, not only reload into
  the object that wrote the checkpoint.

## 5. Deterministic Identity

- Derive run IDs from normalized scientific configuration, data manifest, code hash and
  execution topology that can change numerical results.
- Separate component hashes when appropriate. Reporting-only changes must not invalidate
  a trained model; training-logic changes must create a new model identity.
- Record seeds, package versions, CUDA/runtime versions, GPU topology, input hashes and
  output hashes with every formal run.
- Repeating an identical completed configuration must not retrain or change result hashes.

## 6. Training And Distributed Execution

- Use one explicit GPU list as the compute selector. One item means single GPU; multiple
  items mean one process per GPU under the declared distributed strategy.
- Preserve the intended global batch size and document per-device batch size and gradient
  accumulation.
- Initialize model state deterministically and verify identical initial parameters across
  ranks before training.
- CUDA-aware DataLoader workers use `spawn`; do not fork workers after CUDA initialization.
- Aggregate validation tensors and metrics across all ranks before checkpoint selection.
- Save model, optimizer, scheduler, scaler, epoch, training history and distributed state
  needed for exact recovery.
- Enable actionable distributed diagnostics, but treat collective timeout as a symptom:
  inspect the first failed rank and upstream worker/process error.

## 7. Experimental Separation

- Training data fit model parameters and training-only transformations.
- The development/validation split selects architecture, stopping point and hyperparameters.
- Freeze and hash the selected configuration before opening the sealed test split.
- Treat an automatically top-ranked candidate as review-required. If every candidate is
  scientifically inadequate, revise a declared module and create a new fingerprint rather
  than certifying the least-bad result.
- Test data never influence model, preprocessing, thresholds, correction parameters or
  stopping decisions.
- Preserve unsuccessful preregistered configurations and complete prediction tables for
  audit and figure regeneration.
- Report multiple seeds or confidence intervals when selection instability is plausible.

## 8. Monitoring And Evidence

- Persist machine-readable progress, epoch history, complete validation metrics,
  per-class results, confusion matrices and checkpoint criteria.
- Monitoring observes the same named graph/model states used by training; it must not
  invent proxy validation values from minibatches.
- A detached server run must survive client sleep or disconnect and expose one documented
  status command showing configuration, progress, metrics, GPU state, disk state and logs.
- Keep the primary endpoint fixed in advance and retain secondary metrics needed to
  understand imbalance, calibration and failure modes.

## 9. Verification Gates

Before a formal clean start, require:

1. Clean version-control state and matching local/remote source manifests.
2. Dataset schema, counts, references, hashes and split-leakage audit.
3. Unit tests for configuration, identity, artifacts, metrics and resume behavior.
4. Graph/model smoke tests for every supported topology.
5. Real single- and multi-GPU smoke tests, including a new-process checkpoint resume.
6. End-to-end tiny pipeline execution and deterministic sweep planning.
7. Confirmation that sealed test access is technically blocked until freeze.
8. Confirmation that cleanup targets only declared run/state roots.

## 10. Destructive Operations

- Default to dry-run for deletion, pruning and source cleanup.
- Require explicit execution authorization and a manifest or root allowlist.
- Stop all writers and confirm GPU/process release before deleting artifacts.
- Log what was deleted and why when the deletion is part of an audited experiment reset.
- Never alter the original acquisition or controlled source storage.

## 11. Documentation Contract

Every project README must provide:

- purpose, scientific scope and explicit limitations;
- source/runtime tree and path overrides;
- environment creation and verification commands;
- exact fresh-build, resume, clean-start, validation, freeze and test commands;
- stage inputs, outputs, integrity checks and recovery behavior;
- experiment-selection rule and primary endpoint;
- monitoring commands and artifact locations;
- data-access and redistribution restrictions.

## 12. Release Gate

A release is ready only when source manifests match across deployment targets, tests and
smokes pass from documented commands, no deprecated implementation remains, the runtime
tree is outside version control, and a new user can reproduce the workflow by supplying
only authorized data and explicit deployment paths.
