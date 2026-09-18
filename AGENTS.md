# Repository instructions
Read workspace/REPOSITORY_STANDARD.md and docs/development.md before changes. This is a new research preparation branch, not a continuation of archived queues. Do not restore legacy folder layouts or copy historical reports into this branch. Preserve scientific operators and MHD Node semantics. Install the fixed third_party/MHD_Framework as its own package; do not embed it in this project wheel. New training or test access requires the applicable data and study protocol. Shared management files and scripts must match the peer research project. Validate installation, structure, imports, tests and source digests before deployment.

Read docs/architecture.md. src and tests/unit use data, models, methods, training, evaluation, analysis, runtime, studies. Preserve these roles and canonical modules in scripts/check_layout.py. Never put implementation modules at the package root. main holds the accepted unified layout; retire only explicitly superseded transition branches, retain historical reproduction references. Update the pinned MHD release only after strict application output/gradient/state_dict acceptance; never follow floating main.

The installed MHD release selects the API. Import `mhd_framework` / `mhd_framework.utils`; this application pins the V4 release, never floating main. Packaging paths changed; V4 tensor implementations are preserved.

Read workspace/MODEL_RUN_STANDARD.md for model/run identity, timestamps, provenance, reuse and retention. Apply it to all new model families and studies, not only current UKB work. Copying accepted parents is distinct from creating a new training execution. Preserve pinned scientific entrypoints and historical artifacts.

Read workspace/GPU_EXECUTION_STANDARD.md for whole-device budgets, measured concurrency and replacement safety. Project/native admission is separate from architecture correctness.

Read docs/handoff/README.md when continuing work or coordinating with another project. It is a dated snapshot: refresh live bindings and acceptance evidence before action. Update it after deployment, protocol or dependency changes, acceptance failures/fixes, phase completion or material blockers. Ibex is the default full-runtime acceptance environment; WS02 is optional, not a prerequisite. Keep healthy running snapshots unchanged.

Read workspace/RESEARCH_AUDIT_STANDARD.md for every progress report, scheduled review, repair and deployment. Compare approved scope against actual feeds and acceptance; operational health never certifies full research readiness. Advance authorized missing work and keep a verifiable repair ledger.
Proactively review assumptions, implementation and conclusions throughout all work, including the limits of your own checks. Investigate contradictions, correct authorized issues and independently verify the result; do not wait for user detection or generalize a narrow pass into overall readiness.

Autonomous corrective maintenance is mandatory for authorized workflows: follow workspace/RESEARCH_AUDIT_STANDARD.md through repair, acceptance, safe restoration and verified downstream progress. Detection or an alert alone does not complete maintenance; keep each open incident owned with a concrete automatic continuation where safe. Preserve healthy work and scientific standards.


Current-version uniformity is a mandatory design, review and release gate: follow
"One current contract; explicit version migration" in workspace/RESEARCH_AUDIT_STANDARD.md.
New readers, execution paths and outputs use one current contract; do not add
host-dispatched legacy schemas, try-new-then-old fallbacks or compatibility modes
to normal runtime. Keep old releases/executors pinned for old jobs and reproduction;
convert completed artifacts with separate versioned migration tools, validate state
and downstream consumers, then admit the canonical outputs. Preserve legitimate
architecture/dimension/task variants. Record migration and rollout evidence in the
handoff; CI or directory renaming alone does not establish migration acceptance.

Resource-lease continuity is a mandatory cross-project design/review gate. Read
"Resource leases are independent of scientific executions" in
workspace/GPU_EXECUTION_STANDARD.md before changing submission, expiry, checkpoints,
recovery, stage caching or completion. Enforce the same rule for neural training,
PCA/SVD/correction, evaluation and reporting; never equate a 48-hour lease with
scientific completion or claim deployed recovery from documentation alone.

Read workspace/DELIVERY_STANDARD.md before scheduling or reporting: prioritize a finite end-to-end delivery package, prepare real dependencies concurrently, reuse exact accepted artifacts, and distinguish single-configuration, matched-package and replication acceptance. No new scheduler or performance-dependent release. Documented rules are not runtime acceptance.

Scientific validity is a mandatory design, deployment and reporting gate. Read
workspace/SCIENTIFIC_REVIEW_STANDARD.md and record the question, changed/held-fixed
factors, executed algorithm, acceptance evidence and conclusion limits. Audit the
full selection policy independently from fixed-configuration ablations. Do not
wait for user detection; missing scientific coverage stays open even when CI and
workers are healthy. Documentation does not certify automated enforcement.

Maintain docs/handoff/status.json alongside its README after material evidence review. Preserve goal, phase/dependencies, exact source/evidence dates, separate planned/implemented/deployed/running/accepted states, limitations and traceable evidence. Repository-only checks cannot certify runtime or scientific completion. Do not renew verification timestamps just because a polling request succeeded; no restricted data or credentials belong in status records.

Read workspace/RESULT_CONTINUITY_STANDARD.md for incremental dependencies, cumulative reporting, and traceable attempts including failed, superseded and withdrawn configurations. Low performance does not invalidate evidence. Verify publication and archive acceptance; documentation alone is not deployed automation.

Read workspace/EFFICIENCY_STANDARD.md: proactively measure repeated computation, dependency stalls and resource bottlenecks before users report delays. Verify numerical/selection equivalence, measured cost improvement and actual downstream progress after safe deployment. Documentation alone is not an installed performance monitor.

For research reports, figures, cumulative pages and weekly delivery, apply workspace/SCIENTIFIC_COMMUNICATION_STANDARD.md. Independently review whether an unfamiliar reader can identify the question, comparison, evidence, limitations and next action. Mechanical checks do not certify scientific interpretation; update generators with presentation changes.
