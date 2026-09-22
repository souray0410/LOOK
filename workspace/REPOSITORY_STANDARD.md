# Souray repository standard, version 2

LOOK and Radon_Bridge are reference research repositories. Their root directories,
common commands and deployment configuration schema must match; domain packages and
scientific protocols may differ. MHD_Framework is an independently installable toolbox.
Its own release layout has no research workspace registry; applications pin one installable API release.

## Research skeleton

- `src/<package>/`: reusable implementation, with native MHD network adapters separate from communication modules.
- `configs/deployment/`: portable host roots; scientific configurations are declared per approved study.
- `scripts/`: identical bootstrap, check, test and preflight entry points.
- `tests/unit/`, `tests/integration/`: synthetic correctness and explicit integration verification.
- `docs/`: architecture, development, provenance and interpretation.
- `experiments/<timestamp>/`: protocol, status, aggregate evidence and artifact index; no restricted data.
- `third_party/MHD_Framework`: exact Git submodule revision, installed as a separate dependency.
- `workspace/`: identical shared management rules and validators.
- `.github/workflows/`: automated CPU installation and correctness checks; no research data or GPU jobs.

A new project should start from either research skeleton, change its project identity,
package, scientific code and protocol, and register its deployment roots. Do not copy
historical experiment outcomes or inherit authorization to access test data or submit jobs.

## Versioning and dependencies

Use `YYYY_MM_DD_HH_MM_SS` (Asia/Riyadh, creation time) as each research-cycle branch
name and experiment directory. It is a cycle identifier, not one branch per code edit.
Commit fixes within the active branch. At closeout preserve the exact accepted commit
and an annotated `archive/<timestamp>` tag; never move an existing archive tag.
`main` is the reviewed common baseline. A timestamp alone is not a dependency lock.
Pre-migration branches with an explanatory suffix are special historical snapshots.

MHD has its own release history. Projects independently pin an exact submodule commit,
explicit installed API version (V5 here), critical-file SHA256, and environment dependency lock.
No startup `git pull` or floating main dependency is permitted. Install MHD from the
pinned local checkout; its development package version is not a replacement for Git SHA.
Each project has a separate environment. Multiple projects may legitimately use different
MHD versions. Upgrade on a new commit only after output, gradient, checkpoint and adapter
compatibility checks. Record old/new SHA and deviations; do not rewrite historical locks.
Each installed release contains one implementation. Applications select V4 or V5 by pinning the release commit; no runtime compatibility switch is used.

## Active storage and historical evidence

Servers retain the active research cycle plus required environments and shared datasets.
Historical source, protocols, aggregate tables, figures and public-safe provenance belong
on GitHub. Restricted raw data, participant predictions/features and large checkpoints
belong in authorized artifact storage; a private GitHub repository is not that storage.
GitHub stores only non-identifying artifact references and checksums.

Before retiring an old server work copy: verify the pushed immutable code reference,
copy all necessary artifacts (including optimizer/RNG state when required), verify file
count, byte sizes and SHA256, and test representative checkpoint reload and report reads.
Resolve parent checkpoints/bases from the archive before removing their active copies.
Keep original files and historical records unchanged until acceptance. Never delete the
only copy, active-job dependencies or another project's data. Shared original datasets
are not old run results. No cleanup is performed merely by running the template commands.

## Evidence

Installation, synthetic tests and deployment are not new training or clinical evidence.
Every study declares data roles, selection, metrics and resource limits before execution.
Reports separate development and test, statistical precision and practical relevance.

## Portable paths

Project-owned configuration paths are relative to the discovered repository root,
not the process working directory. External data and results are configured through
explicit machine profiles or environment overrides; absolute external roots are valid.
Never embed a developer's home/server directories into reusable scientific code.
Historical original paths remain evidence; relocate them only through explicit maps.

## Model and training execution records

Follow [MODEL_RUN_STANDARD.md](MODEL_RUN_STANDARD.md) for all current and future
models and studies. Architecture, recipe, execution and accepted artifact are
distinct identities. Each model training execution has its own timestamp; study
metadata and scheduling queues do not substitute for that execution identity.
