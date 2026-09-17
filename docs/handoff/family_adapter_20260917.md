# Independent affine search adapter — implementation acceptance

This is a bounded implementation checkpoint, not deployment or a scientific result.

The adapter in `look.methods.family_greedy` fits each of the eight registered affine
operators independently at eligible nodes. It requires an explicit finite candidate
table; it never inherits the nodes, rank, regularization or on/off decisions selected
by the original PCA method. Its default is best-forward. Every subsequent round
refits downstream candidates using only the accepted upstream bank. Train fitting,
dev selection and explicitly labelled train-only probes have separate roles. Test
input is rejected.

Fitting statistics are checkpointed with method, basis, data identity and upstream
artifact digests. Serialization is followed by prediction replay. Resource refusal
does not silently compress the feature space or reduce the declared rank.
Only PCA-subspace and PCA-constrained-intercept operators require the corresponding
number of stored PCA components: a free residual/PLS subspace must not inherit that
extra constraint. A frozen train normalization basis remains a dependency, not a
claim that every operator acts in PCA coordinates.

The shared trajectory engine additionally checks cached baseline and rejected
candidate prediction files during resume. Rejected predictions are evidence for
keeping a correction off and must not escape integrity checks.

## Verification

On Ibex, an isolated CPU source snapshot and the exact locked MHD V4 source
`c0a27abb3e0f2153bfd273b1d05d5b7dae9784f0` passed **97 tests** across methods,
search-policy registration and the weekly delivery gate. The first focused run
passed 33 tests. An expanded run initially lacked the optional MHD model package;
after supplying the locked submodule, all 97 passed. No installed environment or
healthy running source was changed.

New tests cover all eight operators, accepted-upstream refitting, reload, statistics
pause/resume, corrupted prediction rejection, test refusal, resource refusal and
free versus constrained rank. Adapter tests use controlled feature/evaluation
fixtures; they are not a full-data GPU lifecycle certificate. Existing MHD node and
frozen-parameter tests remain in the passed suite. Layout and source-lock checks also
passed locally.

## Remaining mandatory integration

1. Bind each independently searched family to an explicit approved rank/penalty and
   spatial candidate table; preserve the old PCA-conditioned comparisons separately.
2. Add immutable case registration, formal runner, resource qualification and
   independent acceptance/reporting, using the existing task claims and scheduler.
3. Validate actual MHD/data extraction, complete-cohort memory and interruption/reload
   inside an admitted allocation before formal execution.
4. Bind the accepted cases to the complete seed-3416 weekly package. Do not clear
   `independent_affine_best_forward_full_pipeline_not_yet_deployed` merely because
   these unit tests pass. Release later seeds only after the entire weekly gate.

The currently running PCA/GCV search retains its original immutable source and
identity. It has completed the initial 782-batch dev traversal and entered full
train fitting; the independent family is still not formally deployed. No test data
or scientific comparison results were used by this implementation check.
