# Allocation continuity audit — 2026-09-15

The current rule is 48-hour resource leases and continuous scientific executions.
The common contract is workspace/GPU_EXECUTION_STANDARD.md version 5, copied
byte-for-byte across MHD_Models, LOOK and Radon_Bridge and mandatory in AGENTS.md.

## Live evidence and limits

- All 24 observed active A100 allocations requested 48 hours; no allocation was
  extended, cancelled or interrupted for this audit.
- Native and project owners request graceful checkpoint retirement 900 seconds
  before EndTime. Native full checkpoints include optimizer/scheduler/RNG/data
  progress. Project dispatch verifies completed artifacts before skipping them.
- LOOK original host acceptance is reused independently from correction completion.
  Its complete-training PCA bank reuses verified completed entries. An interrupted
  entry can be recomputed; this is not intra-entry or batch-exact PCA recovery.
  Selected correction banks are reusable; partial evaluation/reporting may rerun.
- Project expired-claim reconciliation checks absence from current Slurm jobs and
  accounting TIMEOUT/PREEMPTED/NODE_FAIL before releasing claims. Unknown liveness
  remains protected. Native failed/liveness-review claims are not certified as
  automatically repaired by the refill loop; scoped recovery evidence is required.
- Existing family workers already allow fixed-epoch training across leases. The
  prepared common owner previously gated other recipes on a whole-run estimate,
  which could defer long checkpointable work indefinitely. New code uses the same
  lifecycle-gated segment window for fixed-epoch and plateau recipes and for both
  profiling admission and actual standalone/companion admission.
- New submission paths reject historical short/unguarded requests. The old pool
  and unguarded renewal cannot submit in the new release. Historical releases
  retain their original executors; this does not hot-patch running jobs.

## Remaining acceptance, without changing science

The common standard is not a blanket claim of fully deployed recovery. Validate
new management changes, synchronize source and CI, and bind only accepted future
owners to the new snapshot. Existing workers keep immutable source pins. A native
package migration must not silently substitute its trainer for a legacy run.

For LOOK, test interrupted PCA entry and progressive correction reuse under the
actual pinned executor and record its lost-work bound. A single entry that cannot
finish within a lease needs finer versioned checkpointing, not endless retries.
For each workflow, retain corrupt-state, hard-timeout ownership, resumed-update,
completed-task-to-next-task and real downstream-progress evidence. Pending native
liveness reviews remain owned incidents, not accepted configurations. No test
access, new scientific recipes, new allocation count or completion rule is granted
by this audit. Resource-policy targets remain LOOK10 / Radon_Bridge10 / generic4.

## Validation receipt

Ibex isolated management suite: **272 passed** (validation2, 2026-09-15).
The first full run found three obsolete test fixtures without the now-required
role policy; those fixtures were updated to exercise real role-budget validation,
and the failed attempt was preserved. The targeted suite had passed 62 tests.
No Slurm submissions or real scientific training were performed by these tests.
The two research-project structure checks and shared-document byte comparison
also passed. Full deployment and real expiry handover remain separate evidence.
