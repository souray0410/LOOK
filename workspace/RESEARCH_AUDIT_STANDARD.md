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

`scheduling/research_audit.py` in MHD_Models checks configured controller/feed
coverage without importing models, touching test predictions or changing tasks.
It complements, and cannot replace, live Slurm inspection, GPU lifecycle tests,
scientific artifact acceptance and exact-SHA GitHub CI checks. Keep its report
and previous fingerprint to distinguish a new incident from a known blocker.
