# LOOK shared feature extraction — 17 September 2026

## Accepted computation change

The frozen host is walked once per train minibatch and missing-input/upstream state.
All eligible sites are projected immediately at their original MHD levels. Complete-input
projections are immutable and shared between missing-input states; missing-input moments
are keyed by the complete accepted upstream bank. CPU projection dtype, minibatch order,
PCA, GCV, ridge solving, candidate evaluation and the selection policy are unchanged.

Cached projected features remain restricted train assets on Ibex. Keys bind the host,
parent specification, PCA bank, extraction version, batch and data role. Shards include
participant order/count stamps and payload checksums. Statistics use atomic checkpoints
and exclusive writers. Resume validates the already committed prefix against its original
shards. A source-only conversion is an explicit offline tool, not a runtime fallback.
It refuses a live original execution, changed scientific settings, changed artifacts,
changed predictions or a new source without its exact numerical acceptance receipt.

## Evidence

- Eight shared-cache tests passed: exact moments and solutions, cross-pattern reuse,
  corrected upstream state, pause/resume, corruption, unfrozen rejection, changed data
  order/source identity, writer exclusion and full best-forward path equivalence.
  (Several assertions share a test.)
- 103 method/search tests passed on Ibex; three additional migration tests passed.
- Real frozen ResNet50 host, 32 train participants, nine sites, two missing-input states,
  and a downstream pass under a nonzero upstream correction: exact moments, GCV solutions,
  artifact parameters and logits. No test participants were read.
- Latest isolated oracle extraction times: 7.14 → 1.03 s; 7.15 → 0.86 s;
  corrected-downstream 6.82 → 0.86 s. These measure only extraction/statistics on a small
  warm probe with concurrent load. They do not establish whole-study speedup.
- The larger 512-train/32-train full search resource profile is a separate acceptance
  gate; follow its actual receipt rather than treating the small oracle as deployment.

Evidence root: `OPS/look_efficiency_20260917/`; immutable candidate source `review_v2`;
`probe_v2/accepted.json` binds exact source bytes and specification. Restricted probes
and predictions remain on Ibex. Healthy historical workers are not modified in place.

## Deployment state

Source `ec4dc8f05b20d38de89abfb7a39d5ec0784780bc` passed GitHub CI
[35220985827](https://github.com/souray0410/LOOK/actions/runs/35220985827).
The complete 512-train fit / 32-train evaluation profile finished in 101.9 s,
with GPU peak reserved 3,531,603,968 bytes and RSS 2,409,086,976 bytes. Both missing
states exactly matched the old profile's full candidate scores, decisions, selected
artifacts and predictions. The three-round CFP-missing probe exercised corrected
upstream refitting. These are technical checks, not development research findings.

Original formal step `51909172.10` exited normally after a pause request. Seven
completed candidates were verified and explicitly converted, with their artifact and
prediction SHA preserved, into run `2026_09_17_15_28_21_422026` in
`search16_q32_shared_20260917_v1`. Original data and source were not rewritten.
The old run is absent from the active dispatcher feeds and its standalone sequence
is marked superseded; the original paused evidence remains available.

The first new launch omitted an already accepted allocator environment; proactive
review caught this before a memory failure. It paused normally and restarted the same
new run/spec with `MALLOC_ARENA_MAX=1`, `MALLOC_TRIM_THRESHOLD_=131072` and
`MALLOC_MMAP_THRESHOLD_=131072`. Both attempts and the incident are retained.
The current immutable launch is `formal_config_v3.json`, manager `manager_v3`,
step `51909172.17`. Its formal development replay advanced to 151/782 batches;
allocation owner `.0` and DenseNet parent `.1` remained active. This verifies real
computation after handover, not full formal completion or a measured end-to-end speedup.

`look_active_workflow.json` now exposes `priority_search` and the current binding.
The finite existing-allocation manager owns this task. Central dispatch expansion,
automatic representative controls and the entire weekly package remain separate
unaccepted gates; the handover does not silently declare them complete.

The active search remains first-seed fixed16/q32 best-forward, with remaining candidates
fitted under the accepted upstream corrections and stopping only if no remaining
candidate strictly improves its development score. New seeds remain behind the weekly
matched delivery gate.

## Finite lease continuation and cumulative publication (15:54 Saudi)

The original `launch_bound.py` pauses before expiry and exits; its existing
`launch.json` deliberately prevents blindly restarting that manager. It was not
by itself an automatic next-allocation recovery mechanism.

`finite_search_continuation_v3.py` and `continuation_binding_v3.json` now provide
one bounded entrypoint for the existing allocation owner. They pin the original
configuration, specification, scientific source bytes, manager and shared-claim
implementation. The caller supplies only its allocated job ID. Live or unknown
old-step state, an occupied continuation/manager/run lock, or absence of a proved
recoverable lease transition cannot start a duplicate worker. Confirmed death
plus a clean paused claim or TIMEOUT/PREEMPTED/NODE_FAIL permits a new management
attempt with the same scientific run/spec/source/env. Each attempt retains the
old claim, dead-step evidence and pause request; the accepted finite manager
performs resource admission, fitting, both missing-state replay and reporting.
No allocation is submitted by this entrypoint.

Three isolated management tests passed on Ibex, including original run/env
preservation and distinct attempt creation. A real held-lock check returned
`already_owned`, and the live formal-step check started no new attempt. Receipt:
`OPS/look_efficiency_20260917/continuation_validation_v3.json`. These checks are
not a claim that the next real allocation expiry has already occurred. Integration
into the existing pending/future allocation-owner entrypoint is a separate
binding action recorded by that owner; the recovery script alone is not a scheduler.

A detached read-only cumulative observer (PID 605155 at deployment) uses the
existing `cumulative_delivery` collector/publisher. Its fixed output is
`OPS/look_efficiency_20260917/continuation_v2/publication/current`. It published
one configuration's coverage immediately and adds its four matched result cells
only after the original manager has accepted the per-run delivery. It does not
invoke the full representative-package report or release another seed. The
publication is idempotent and retains the previous valid release on failure.

The unchanged formal step `51909172.17` had 776 then 916 shared feature shards
in two reads 119 seconds apart, with the latest write less than one second old.
A node-level read observed worker RSS 2.08 GiB and GPU memory 2492 MiB; these are
snapshots, not complete-run peaks. Formal log contained no traceback and the
original owner and DenseNet worker remained present. Search-stage logging is
quiet during shared extraction, so log age alone does not imply a stalled worker.
The configuration still has no scientific `accepted.json`; complete two-state
search, final report, real next-lease resume and whole-week acceptance remain open.
