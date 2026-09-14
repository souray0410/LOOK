# Model and training-run standard — version 3

This is the common standard for current and future research projects, independent
of disease, dataset, architecture, framework release, server and scheduler. It
covers native-model training and project-specific training. LOOK and Radon_Bridge
are consumers, not exceptions or the scope limit. Numerical methods and clinical
protocols remain project-specific.

## Separate definitions, recipes, executions and artifacts

1. **Model definition:** maintain reusable architecture code once in its owning
   package. Identify architecture variant, spatial dimensions and exact source
   revision. A shared name such as ResNet50 does not equate 2D, inflated3D and
   another3D implementation. A complete task model also identifies its head and
   aggregation/adapter; backbone weights alone are not a complete task model.
2. **Recipe/configuration:** record model, task/loss, dataset and split identities,
   preprocessing, initialization, seed, optimization, batch/accumulation, precision,
   framework/environment, evaluation and selection rules. Record a full-config
   checksum and the workflow's explicit reuse/request identity. A timestamp is
   never proof of scientific equivalence. Separate artifact content identity from
   machine-specific file locations; do not rewrite historical identity schemes.
3. **Execution:** one training execution of one complete model/system receives one
   unique timestamp directory. Independently trained models each receive their own
   directory. A jointly optimized composite model is one execution, with named
   components and parent-run mappings, not contradictory independent checkpoints.
4. **Accepted artifact:** selected state plus the metadata needed to reconstruct
   and verify it. Copying an accepted artifact for another project's use does not
   create a new training execution. Preserve source_run_id and artifact checksums.

## Timestamp and directory rules

Use registration time `YYYY_MM_DD_HH_MM_SS_ffffff`, with timezone/UTC offset saved
in metadata. Microseconds and atomic exclusive directory creation prevent a shared
second from overwriting a run. Registration, actual start, stop and acceptance are
separate events. A reservation is not a completed model or a selected recipe.

```text
<artifact_root>/
  index.json                 execution identity and provenance map
  index.csv                  readable state snapshot
  index_status.json          snapshot generation time
  runs/<run_timestamp>/      one model-training execution
    README.md                human-readable model/task/configuration meaning
    run.json                 execution/source/parent identity
    configuration.json       registered configuration or explicit pending intent
    spec.json                actual executed configuration, when available
    best.pt                  selected complete model state
    last.pt                  full stopping/resume state while required
    status.json              current worker state
    accepted.json            verified acceptance, only after completion
    provenance/              per-launch observed environment and source receipts
    preflight/               non-scientific checks, if required
    ...                      logs, history, metrics, diagnostics and predictions
  queues/<queue_id>/         scheduling manifests, allocation logs and task index
```

This is a storage contract, not a mandated checkpoint serialization format. Other
frameworks may use a documented equivalent to .pt. Names and meaning must remain
clear, versioned and verifiable. Research-cycle branch names and study protocols
have their own identifiers; do not nest their timestamps into model identity.

An unchanged interrupted execution keeps run_id and appends attempt/recovery
records. A deliberate repeat, new seed or changed scientific recipe receives a new
run_id. Preserve prior failures and references. Exact same-seed repeats are allowed
when declared but are not extra independent seed evidence. Parameter changes are
not described as exact checkpoint continuation.

## Index, provenance and reuse

Index architecture/implementation/dimensions, input track, task, dataset/split and
preprocessing digests, framework/code revision, initialization, seed, hyperparameters,
parent/repeat links and result locations. Mark reserved, running, paused, failed,
needs-attention and accepted states distinctly. Snapshot tables include update time;
worker records and validated acceptance remain authoritative. Pending selected
recipes are visibly pending until the actual execution specification is available.

Resolve by the explicit scientific request and verified artifact content. Do not
choose whichever timestamp is newest or whichever filename looks similar. Project
consumers retain independent ownership of fine-tuned weights and map source models;
shared parent models are immutable within a locked study. Missing parents create an
explicit training request, never a silent random substitute.

## Shared queue and explicit completion markers

Every lane, including a project lane without ready project work, selects from the
same finite native queue. A single locked owner dispatches unique run IDs; a CSV
snapshot never grants ownership. The canonical queue lock is independent of the
owner output directory. Claimed run-directory aliases are excluded from selection.
Deliberate repeat executions are separate run IDs linked to the same recipe; do not
confuse such repeats with duplicate dispatch of one execution. Different queues
must have disjoint execution ownership or a separately accepted shared coordinator.

Index `training_complete` only after matching configuration, required receipt files,
SHA and plateau evidence have passed verification. Raw `state=completed` alone is
not accepted. Expose not-started, awaiting-selection, active, paused/checkpoint,
failed, acceptance-pending and evidence/liveness-needs-review states separately.
A stale heartbeat does not authorize stealing a task. A checkpoint's existence is
not successful strict resume validation. A protection cap or OOM is not completion.
Failed versions need scoped review; do not blindly retry them or restart accepted
models. Resource-compatible pending or safely resumable work may then be dispatched.

Keep frozen existing owners authoritative until safe replacement is accepted.
Introducing a new queue-lock implementation does not retroactively lock legacy
owners; never launch another owner for their queue. Existing single-owner lanes
already exclude in-flight IDs. New inventory readers are independent of training.

## Data processing and end-to-end reproducibility

Keep executable source for inventory, label derivation, exclusions, participant/visit/
eye pairing, dataset splitting, decoding/QC, deterministic preprocessing, train-only
fitting, stochastic augmentation, model training, evaluation and statistical reporting.
Keep the actual configuration and ordered invocation for every executed stage. A README,
a seed, a dataset name or a checkpoint alone cannot reproduce a result. A mutable branch
name alone cannot identify the source that generated a dataset or model.

Separate ownership, without breaking provenance:

- The framework owns architecture definitions and generic model interfaces; importing
  a model must not import a clinical cohort, create a split or launch preprocessing.
- A versioned data pipeline owns source inventories, label rules, split manifests and
  reusable derived datasets. Share immutable derived data only when definitions match.
- The independent training workflow consumes that dataset version and owns native
  training recipes, complete task-model checkpoints and their execution records.
- Each research project consumes explicitly identified parents and data, and owns its
  method, adaptations, fine-tuned weights, evaluation and analysis. Record every parent
  artifact and dataset edge; copying a parent does not make it a new trained model.

For each pipeline stage, retain its exact source commit and file SHA256, configuration,
actual start/finish/status, command or structured arguments, input and output manifests,
checksums, environment and upstream receipt references. Archive required uncommitted
source before execution; a dirty flag or checksum cannot replace the missing bytes.
Keep immutable source references available after branch deletion or server cleanup.
Store executable source once per content version; runs may reference that preserved
bundle rather than duplicating code or the dataset for every trained model.

Dataset receipts distinguish the source release/field meanings, label provenance and
time rules, participant counts and exclusions, sample/visit/eye mapping, split roles,
and deterministic processing parameters. The restricted manifests retain exact IDs
and input-file hashes; GitHub contains source/configuration and non-identifying aggregate
receipts only. Hashing participant IDs does not make their row-level manifests public.
Derived-data identity includes pipeline code, recipe, source manifests and any fitted
preprocessing artifact. Different task/model-specific transforms create new derived
versions instead of overwriting a shared cache.

Fit normalization/PCA/SVD and other learned preprocessing only on the protocol's training
partition. Store the fitted artifact and training-manifest digest; reuse it unchanged for
development/test. Fixed image decoding of test data is recorded separately from model
performance access. Neither pipeline construction nor this standard grants test access.
Keep class weighting, augmentation, sampler state and loss implementation with the recipe.

An acceptance audit follows this chain in both directions:
`raw source -> labels/cohort/splits -> processed data -> native training -> parent artifact
-> project training -> selected checkpoint -> predictions -> statistics/report`.
Verify hashes and reconstruct representative processed samples and selected-model outputs
before claiming reproducibility. Record tolerances and numerical limitations. Do not
claim a full raw-data regeneration from only a synthetic or small replay check.

## Declared and observed execution environment

Each start/resume records its own observed environment; preserve previous snapshots.
Record OS/architecture, Python, installed dependencies with a recreate/install lock,
exact framework/model/project source, PyTorch build, its compiled CUDA version, driver,
cuDNN, GPU model/UUID/memory, device topology and process/rank/world-size mapping when
applicable. The CUDA toolkit selected by a module or nvcc is distinct from PyTorch's
compiled CUDA and the driver's supported CUDA level; record them separately, with
unavailable fields explicit. A login-node/preparation snapshot is not a worker GPU record.

Keep observed precision, AMP/scaler, TF32, deterministic flags, cuDNN benchmarking,
batch/microbatch/accumulation, seeds, optimizer/scheduler, initialization checksums,
selection rules and resume checkpoint identity. Hardware/world-size changes are recorded;
a full resume state alone does not guarantee bitwise equivalence across such changes.
Capture only an allowlist of relevant environment settings, never tokens, passwords or
an entire environment dump. Package-version inventory alone is not a dependency lock.
Do not backfill a historical runtime using today's versions: mark absent evidence unknown.

## One reproduction record; artifact-level access

Public/private are visibility attributes, not two model families, directory trees,
recipe schemas or training pipelines. Every execution uses the same architecture,
data definition and preparation, configuration, environment, weights, evaluation,
provenance and reproduction-command record. Do not maintain public/private trainers
or omit scientific stages from the public description. Keep unpublished project
methods out of a generic model dependency, regardless of repository visibility.

Record access separately for each artifact: public, restricted or private. Record
availability separately: available, withheld or pending. Access is not availability,
scientific acceptance or permission to publish. A private repository may reference
public initialization weights and restricted data in the same execution. Approved
weights may be public while their training data remain restricted.

An additive distribution manifest references the original run ID and exact recipe
checksum. It lists artifact roles, content checksums when known, dependencies and
access/availability metadata. Changes in access or storage location do not create a
new model/run or rewrite scientific configuration, historical digests or receipts.
Local paths and credential-bearing URLs belong in separate authorized bindings,
not the portable identity record. Changing scientific settings creates a new recipe.

The complete canonical record stays in authorized storage. An authorized release
is a reviewed view of that record; it is not a second maintained implementation.
A withheld artifact remains an explicit dependency, with permitted acquisition or
reconstruction instructions. Missing required assets block the applicable replay;
never substitute synthetic inputs, another split, weights or automatic retraining
while claiming the same result. Public users with authorized assets run the same
stage commands and acceptance checks. No claim requires raw data to be public.

Preserve a private source-to-release mapping and identify withheld metadata honestly.
A redacted view cannot claim the byte checksum of the full record. Participant-level
IDs, manifests, predictions, internal paths and sensitive source metadata require
separate disclosure review; even hashes are not an automatic anonymization method.
No access flag authorizes upload, a visibility change, deletion or publication.

MHD_Models runtime/distribution.py validates this additive metadata contract;
it does not export files, resolve credentials, replace model loaders or certify a
release as safe. Existing frozen workers/receipts remain valid and untouched.
Numerical replay, disclosure review and publication are separate recorded gates.

## Retention and finalization

Retain complete recovery state for running, paused, failed and unfinished runs.
For accepted completed models, permanently retain selected state, reconstructing
configuration, code/environment and source identities, metrics/history, necessary
prediction evidence and acceptance/finalization records. Keep restricted predictions
and data only in authorized storage, regardless of repository privacy.

Redundant checkpoints, transient features and optimizer stopping state can be
removed only after selected-model reload/metric verification, dependency checks,
and a finalized-artifact format accepted by all downstream loaders. Record removed
file names and checksums. Preserve any scientifically required ensemble/trajectory
states. Do not erase files still required by a frozen receipt or claim that a cleaned
artifact remains exactly resumable. This standard itself launches no cleanup.

## Portability and long-term adoption

Configure external roots; repository-relative paths resolve from the repository,
not the current shell directory. Generic model/scientific code contains no user's
server paths. Bind devices through the execution environment, not saved host GPU
numbers. A server migration preserves execution identity and content hashes through
an audited location mapping; it is not a new scientific run.

New project templates reference this file and their model/run entrypoints declare
how they implement it. MHD_Models's runtime/run_registry.py is the current
reference execution registrar; it is independent of MHD core. A project may use a
different backend implementing the same contract, without forcing unrelated
frameworks through an MHD-only API. No model architecture or arbitrary model export
is implied by this organizational standard.

Keep the standard text identical in participating repository workspace directories.
Update its version and peer checks together. Preserve old checkpoints, pinned source
and immutable scientific records; do not mass-migrate archives merely to restyle them.
New studies inherit management conventions, not data access, GPU authorization,
clinical assumptions or another study's optimization/selection decisions.
