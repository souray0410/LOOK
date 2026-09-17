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

Implementation and small numerical acceptance are complete. Formal handover is pending
the complete resource profile, explicit checked candidate migration, new execution
receipt, claim/observer transfer and independently observed formal downstream progress.
No accepted formal speedup or new scientific outcome is asserted by this document.

The active search remains first-seed fixed16/q32 best-forward, with remaining candidates
fitted under the accepted upstream corrections and stopping only if no remaining
candidate strictly improves its development score. New seeds remain behind the weekly
matched delivery gate.
