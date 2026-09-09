# Scope amendment on 2026-09-07

The user has now explicitly resumed all 182 deferred start cases. Pipeline 47 builds
a frozen plan from audited original cases; it never reruns the verified 61. Each
selected-context job depends on all its own unfinished starts, freezes the two
validation-selected directions, and evaluates the fixed random masks without searching
again by missingness ratio. Existing four complete selected contexts are reused.
The old empty plan remains a record of the earlier idle state, not the current scope.

# Independent single-GPU LOOK queue

Effective 2026-09-07: the user replaced the earlier one-case-at-a-time execution
restriction with up to two independent cases, one on physical GPU 0 and one on 1.
Each case exposes only its own card as logical cuda:0. This is case/seed parallelism;
it does not divide one seed across cards. Per-card LOOK process memory remains
14 GiB, counted across every LOOK CUDA process including data-loader descendants.
The torch allocator reserves at most 12 GiB; a 250 ms process-memory guard stops
this queue's offending worker. The sampler is a guard, not a hardware partition:
transient between-sample peaks cannot be ruled out. Other projects are never killed.

## Default launch

`project.json` declares pipeline/46_run_dual_gpu_queue.py as the default for future
independent LOOK cases. Entries 38–45 preserve their historical execution semantics
and are not auto-launched. This release's pending plan is empty: all authorized
compact experiments finished, and the 182 deferred search cases remain deferred.
There is no new scientific computation merely to occupy two cards.

Set `PYTHONPATH=<source>/tool`, use the recorded venv, then run:

```sh
python <source>/pipeline/46_run_dual_gpu_queue.py \
  --project-root <source> --plan <frozen-plan.json> --output <new-runtime>/queue \
  --lock-path /data/mengh/LOOK/maintenance/bounded_gpu.lock
```

The shared supervisor lock excludes historical LOOK queues. It is inherited by
children, so a parent crash cannot permit a second scheduler while old cases are
still running. Each case has a second exclusive lock, its own directory and a
completion receipt pinned to the complete plan and deployed source manifest.
Only the supervisor writes the queue summary. Resume verifies result, prediction,
configuration and dependency hashes before reusing a completion. Interrupted cases
resume their own existing scientific identity; physical-card assignment is only
execution metadata. A source/plan change requires a new runtime directory.

## Explicit plan, no implicit expansion

The JSON schema is `protocol: independent_single_gpu_cases_v1`, `test_access: false`,
`jobs: [...]`. Empty jobs means nothing is pending. Nonempty scientific plans also
require `acceptance_record: {path, bytes, sha256}` pointing to this release's complete
real-data two-card acceptance. Each job declares:

- `id`: unique safe case identifier; `kind`: method or suffix; `seed`: source seed.
- `source`: hashed JSON file containing the pinned original source inventory row.
- `case`: existing method or suffix case object; method jobs also include the full
  existing `spec`. No new correction formulas, PCA fits or scientific parameters.
- `after`: IDs that must all finish first. For a method wave, give every seed in the
  next method the three preceding method IDs. Self-input cases depend on all method
  cases. Cases in a wave can run on either card. This preserves method order while
  parallelizing seeds.

Case/source fusion and filling must agree. Suffix candidates must be an exact ordered
suffix of the source graph. Duplicate scientific jobs are rejected. No launcher
infers jobs from incomplete historical grids or chooses additional configurations.

A free card starts a case only if at least 14 GiB is available and no other LOOK
process occupies it. A busy card does not prevent the other card from progressing.
Each worker is isolated from inherited DDP rank/world-size settings. Source inference
batches, precision and search settings are preserved. Only SSF training OOM retries
use 8 → 4 → 2 → 1, via the existing effective-batch-128 accumulation and epoch/RNG
checkpoint recovery. The final failed attempt blocks the queue; no failed batch is
skipped. Non-SSF OOM blocks without changing source inference batch or refitting PCA.

## Acceptance

CPU regressions exercise actual concurrent subprocesses, dependency barriers,
resume without rerunning completed results, hash mismatch rejection, per-card
aggregate memory accounting and visibility isolation. The real GPU probe uses eight
validation participants per seed, frozen checkpoints and existing ON matrices in
both missing directions, with no new fitting and no test access. Compare parallel
and serial probe logits for each seed; record source manifest, per-card process
peaks and overlapping execution intervals. These are engineering checks, not new
scientific outcome estimates or a throughput benchmark of complete training jobs.
