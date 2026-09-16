# Research execution audit standard

## General self-review principle

Apply this to every research and engineering task, not only previously observed
failures. Proactively challenge assumptions, check implementation against intent,
and distinguish observations from inferences. Ask what evidence could disprove
the current assessment and whether the inspection itself has blind spots.
Recheck after meaningful changes, at dependency transitions, after anomalies and
before declaring readiness or completion. A passing narrow test cannot certify
an entire system. Do not wait for the user to detect contradictions.

Investigate and repair within the existing authorization, then independently
verify the original symptom and downstream behavior. If evidence conflicts,
revise the assessment and approach instead of repeating reassurance or the same
failed action. Record uncertainty and unresolved work candidly. Do not silently
change scientific aims, data roles, comparison fairness or completion standards
in the name of automatic correction. Keep this principle applicable as models,
datasets, systems and failure modes evolve; finite checklists are aids, not proof
that every relevant issue has been covered.

A live process, successful CI, a checkpoint file, or an empty legacy
`remaining_implementation` field does not establish that an approved research
protocol is fully implemented, deployed, scientifically accepted, or complete.

For every progress report and scheduled review:

1. Read the latest approved protocol, repository handoff and live binding. Keep
   the declared matrix, implemented cases, admitted jobs, active workers,
   accepted matched groups, and published results as separate counts.
2. Check the exact GitHub commit and CI result; inspect failures and fix in an
   isolated checkout. A local pass cannot stand in for a clean CI pass.
3. Compare all required parent catalogs and case registries with the active
   feeds. Check immutable input/source hashes, claims and actual Slurm steps.
   A configuration for future owners is not deployed into existing owners.
4. Read fresh run status, log/history advancement, failure traces and scientific
   acceptance receipts. Verify recovery and data roles with their real acceptors.
   An accepted status string alone is not model acceptance.
5. Keep an independent gate checklist for the newest protocol. Classify each
   item as accepted (with a matching hashed receipt), running, waiting for
   dependencies, not implemented, not deployed, not validated, or failed.
6. For a repair: preserve healthy workers; record failure evidence; change an
   independent version; test; obtain CI and appropriate runtime acceptance;
   deploy safely; recheck the original failure; update the handoff. Do not mark
   the issue closed at the code-writing or push stage.
7. An unchanged implementation gap is still unfinished work. During an
   authorized maintenance turn, advance the next bounded missing component
   instead of only polling GPUs. Never relax a scientific gate to clear a status.

Reports must state operational health separately from protocol readiness and
scientific results. Avoid unqualified statements such as "everything is fine"
when an implementation, deployment or acceptance gate remains open. Existing
known blockers need not trigger repeated notifications, but stay visible in the
repair ledger. New failures, regressions, verified recovery and required user
input are actionable notifications.

`src/mhd_models/scheduling/research_audit.py` in MHD_Models checks configured controller/feed
coverage without importing models, touching test predictions or changing tasks.
It complements, and cannot replace, live Slurm inspection, GPU lifecycle tests,
scientific artifact acceptance and exact-SHA GitHub CI checks. Keep its report
and previous fingerprint to distinguish a new incident from a known blocker.

## Autonomous corrective maintenance — permanent execution rule

For authorized research and engineering workflows, detection is the beginning of
maintenance, not its deliverable. The responsible agent must independently follow
an issue through evidence collection, cause analysis, versioned repair, appropriate
acceptance, safe deployment, restoration and verification of actual progress. Do
not leave a repairable failure as an alert or wait for the user to repeat the task.
This rule applies to future projects, models, framework versions, data pipelines,
CI, scheduling, evaluation and reporting; it is not specific to one incident.

### Actions and limits

- Inspect dependency boundaries proactively, before allocation/deployment and after
  changes. A scheduler that is alive while science is stalled needs investigation.
- Isolate the affected path and preserve evidence; keep unrelated healthy workers,
  allocation owners and immutable scientific snapshots running. Protect against
  repeated submissions, duplicate claims and accidental checkpoint replacement.
- Prepare a bounded fix independently. Validate the original failure and relevant
  adjacent paths, including real input, resource and recovery checks when required.
  Passing static checks or CI cannot replace runtime acceptance.
- After acceptance, automatically deploy and resume the same authorized execution
  from its verified state. Recheck actual step liveness, fresh logs and committed
  training/evaluation progress. Successful submission or a running PID alone does
  not close an incident. Recheck for regressions after the handover.
- Temporary infrastructure faults permit bounded retries with recorded limits and
  backoff. Deterministic failures require a changed, validated repair; no blind
  restart loop, endless reallocation or fabricated workload to occupy a GPU.
- Preserve scientific identity, split/test roles, batch/precision/BN semantics,
  selection and stopping criteria, comparisons, resource ceilings and original
  artifacts. Do not improve a status by lowering scientific acceptance standards.
  New research choices or destructive/unauthorized actions require user direction.
  Explicit read-only or stop instructions still take precedence.
- If recovery awaits resources or a long validation, register an idempotent,
  monitored continuation that can actually perform the next action. A plan or
  promise alone is not an automated continuation. Never create a second owner.
  If safe automation is unavailable, keep the dependency explicit and notify the
  user when their action or an external change is required.

### Evidence and closure

Maintain a durable incident/repair record with detection time, symptom/evidence,
impact and affected scope, cause (confirmed or hypothesis), exact code/config/run
identities, attempts, acceptance receipts, recovery state, next action and owner.
Record transitions such as detected, isolated, repairing, validating,
waiting_dependency, restoring, verifying_progress and closed. Existing schemas may
use equivalent names; do not rewrite historical records merely to rename states.
Every non-closed issue must retain a concrete next action and continuation owner.

Close only after the original failure is resolved, required acceptance passes and
actual downstream progress is verified. A repair without deployment, a restored
scheduler without resumed work, or an unexplained remaining failure stays open.
Synchronize the handoff and monitoring context so another session can continue.
Report preparation, validation, deployment, restoration and scientific completion
separately. Notify on meaningful failure, verified recovery, completion or required
user action; unchanged healthy/waiting states remain quiet by default.

This is an obligation to investigate and advance safe recovery, not a claim that
all possible faults can be detected or repaired automatically. Re-examine the
checks themselves, investigate contradictory evidence, and correct inaccurate
status claims. Monitoring automation and source changes must respect the same
review, acceptance and resource boundaries as the workflows they maintain.

## Connected research and staged delivery

A supplement must connect to an existing research question before it is counted
as scope: record the hypothesis, matched control, exact changed and held-fixed
factors, predecessor evidence, reusable artifacts, target-cohort validation, and
which claim the result can or cannot support. Review interaction contrasts when
a new implementation changes more than one scientific factor. Do not mistake
a new module, protocol file, or server-specific queue for an independent finding.

Share validated operator definitions and provenance across environments, but
revalidate data roles, parents, dimensions, numeric policy and resource classes.
Small-cohort scores are not large-cohort evidence; UKB organ expansion is not an
independent external dataset. Resource validation and replication gates must not
depend on favorable method performance. Preserve complete matched groups when
prioritizing weekly outputs, then replicate seeds and expand the declared scope.

Before reusing an ablation on a changed parameterization, rederive its meaning
and test its actual input/output and gradient structure. A mask in latent factor
coordinates need not remove an edge between original sources. Keep unsupported
controls explicitly unavailable until their mathematical and runtime contracts
are established. For every open integration gap, distinguish implemented, tested,
deployed, queued, running and scientifically accepted states in the handoff.


## One current contract; explicit version migration

This is a permanent research and engineering acceptance rule across projects,
models, data processing, configurations, artifacts, evaluation, reporting and
scheduling. A new release implements one declared current contract. Do not make
its ordinary execution depend on whether an artifact came from WS02, Ibex or an
older layout. Site settings describe resource roots and hardware, not alternative
artifact schemas. Preserve real execution origin as provenance.

Keep old releases and their readers/executors reproducible at immutable source
pins. Let healthy old jobs finish under their original code, configuration and
recovery semantics. Do not hot-rewrite those jobs to enforce a new layout.
Historical reproduction is served by the historical release, not by accumulating
legacy branches in the current runtime.

Convert completed old artifacts through explicit, versioned migration tools into
the current contract before admitting them to new consumers. Importers may know
source versions and old layouts; normal current readers must not silently try a
new format and then an old one, guess a schema from a host/path, or retain a
permanent legacy-mode dispatcher. Reject unsupported inputs with the required
migration/version information. A catalog alias alone is not a migration.

Migration acceptance must record source and destination identities, schema and
framework pins, file hashes, conversion rules and acceptance evidence. Preserve
scientific configuration, original provenance and necessary recovery evidence;
verify tensor/state semantics, strict loading, relevant outputs/gradients and
Node IDs according to the change. Recheck dependent loaders, project copies,
schedulers, evaluation and reports before switching consumers. Do not relabel an
old framework artifact as a newer version, or call retraining a lossless migration.
Preserve the old reproducible copy until migration and downstream acceptance pass.
A format-only migration does not require retraining or authorize new test access.

### Required review and release evidence

For every structural/version change, record in the handoff or acceptance ledger:

1. The single target contract and its scope, including legitimate scientific
   variants (architecture, 2D/3D, task and data semantics). Such explicit variants
   remain necessary; this rule forbids legacy-format ambiguity, not model diversity.
2. The unchanged old execution sources and dependencies still in use, their drain
   condition, and the separate migration entrypoint/source-to-target mapping.
3. Evidence that current readers reject unconverted legacy input; migration tests
   cover supported source formats, state preservation, tampering/incomplete input,
   idempotence and downstream use where applicable. A regex or CI pass alone is
   not sufficient semantic acceptance.
4. The consumers audited and remaining transitions, separately marked planned,
   implemented, validated, deployed and accepted. Do not claim an entire platform
   migrated because the directory or one representative model was converted.

Review this boundary before adding a fallback or compatibility flag. Resolve the
version difference at migration/release boundaries instead of patching it into
normal operation. Keep this standard identical in the companion research repos
and make it a mandatory AGENTS.md instruction so future work follows it without
another user reminder. Framework V4-to-V5 is not authorized by this policy alone.

## Scientific comparison and conclusion review

Apply [SCIENTIFIC_REVIEW_STANDARD.md](SCIENTIFIC_REVIEW_STANDARD.md) before
design, deployment and reporting. Preserve a concrete scientific review record,
including unresolved algorithm/coverage gates; operational acceptance alone is
insufficient. This is a review requirement, not a claim of automated enforcement.
