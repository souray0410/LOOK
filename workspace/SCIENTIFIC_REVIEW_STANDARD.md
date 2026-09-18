# Scientific review before design, deployment and reporting

This permanent rule applies to LOOK, Radon_Bridge and MHD_Models, across future
cohorts, models and framework versions. The executing agent owns the review;
user inspection is not an acceptance stage. Read this with RESEARCH_AUDIT_STANDARD.md.
Targeting TMI/MIA/NBE raises the required rigor; this checklist neither guarantees
publication nor proves clinical validity. Passing engineering checks is insufficient.

## Required review record

Before a new comparison is dispatched, after an algorithm/selection change, and
before reporting a scientific conclusion, record the following in the study's
versioned protocol or evidence ledger. Reference prior records when unchanged.

| Field | Required evidence |
|---|---|
| Question and estimand | Exact outcome, population, comparison direction and aggregation weights; a result that would contradict the hypothesis |
| Changed factors | Explicit difference between arms; joint changes and interactions acknowledged |
| Held-fixed factors | Participants/splits, parent checkpoints, preprocessing, representation, eligible nodes, seeds, fitting data, selection metric and applicable budget, with hashes |
| Legitimate differences | Parameters, effective capacity, rank, search space or cost that cannot be matched; never label approximate matching exact |
| Actual algorithm | Executed fit/select/apply order, upstream state, disabled path and restore behavior; code entrypoint and immutable source |
| Data roles | Train fitting, internal selection, dev selection and locked test use; participant/eye/visit and cross-task overlap audit |
| Required evidence | Unit checks plus representative real-model prediction/state/recovery acceptance; exact receipt and artifact hashes |
| Statistical claim | Matched participants, seed scope, uncertainty, comparison family, selection bias and practical margin |
| Coverage and next action | Planned/implemented/validated/deployed/running/accepted separately; open issue, owner and next executable step |

A missing field is an unresolved gate, not permission to infer a favorable answer.
Do not rename old forced/conditional results as full-method results. New analyses
made after observing outcomes must be labeled as amendments, not preregistration.

## Algorithm and control-variable checks

- Separate full-method comparisons from fixed-configuration mechanism ablations.
  Fixing another method's selected representation/sites can answer a conditional
  mechanism question, but does not evaluate each method's own selection policy.
- For sequential greedy algorithms, each method has its own trajectory. Compare
  candidate-on with candidate-off under the same already accepted upstream state.
  Rejected candidates must not influence subsequent fitting. Independent starting
  points and all-on trajectories must be refitted when upstream states differ.
- Distinguish a single-node intervention, an all-eligible-node trajectory, and a
  final whole-bank fallback. An end-of-bank fallback is not per-node greedy search.
- Where specified, strict improvement enables a candidate and ties disable it.
  Verify adverse, tied and beneficial cases, rejected-state leakage, restart
  equivalence and identity/no-op output; inspect actual recorded decisions too.
- A dev selection rule may guarantee nondecrease of its own dev primary metric.
  This is not independent improvement, a test guarantee, or a guarantee for each
  subgroup, missingness mixture or secondary metric. Test never reselects gates.
- Fairness does not require nonsensical parameters for methods that lack them.
  Prespecify applicable search spaces and report effective parameter/rank/cost
  differences. Do not expand search selectively after observing rankings.
- Confirm the relevant complete matched group before claiming a method wins.
  One seed can establish technical execution and preliminary evidence, not
  replicated superiority. Replication release depends on technical acceptance,
  not favorable scores. Preserve negative and inconclusive results.

## Review cadence and remediation

At dependency transitions and before every user-facing progress report, reconcile
approved scientific requirements against actual accepted artifacts, not merely
queue counts. Ask whether an alternative explanation could produce the reported
pattern and whether a necessary matched control is absent. Use exact receipts;
synthetic tests and a successful CI cannot establish real-data scientific acceptance.

On finding a contradiction: stop using the affected conclusion, preserve the
original evidence, identify affected groups/consumers, version a bounded repair,
validate it and verify downstream results before closure. Keep unrelated healthy
workers and immutable execution snapshots. Report known gaps without waiting for
the user to rediscover them. Prioritize a complete matched first-seed group for
weekly delivery, then replicated groups; retain the approved remaining scope.

Review has both machine-checkable invariants and scientific judgment. Documenting
this rule does not install a new automatic checker or repair a missing algorithm.
For any promised automatic check, name its executable, deployment and acceptance
receipt; otherwise explicitly mark it as a review obligation or implementation gap.

## Necessity and information gained before dispatch (2026-09-18)

The responsible researcher must review each proposed comparison before execution:
state the unresolved question and a result that could change the interpretation;
check whether definitions, algebraic equivalence or nested feasible sets already
imply the claimed conclusion; identify the smallest informative matched control;
audit exact evidence/cache reuse; and justify cost against the finite delivery.
Record the disposition: proof plus implementation invariant, accepted reuse,
necessary empirical experiment, retained later phase, or explicitly authorized
cancellation. Do not turn all possible combinations into mandatory experiments.

A theoretical search-set dominance result can remove a redundant selected-dev
performance experiment under its stated equal-operator/candidate/score/budget
conditions. It does not establish held-out generalization, finite-budget dominance,
wall time, calibration or clinical utility. Verify theoretical invariants with
synthetic tests and reserve real data for genuinely unresolved empirical claims.
Separate full independently selected pipelines from fixed-prefix/parameter
mechanism ablations; a small mechanism question need not spawn a full search tree.
Keep completed evidence and healthy workers. Scope/primary-comparison changes
follow current user authorization. Documenting this review is not a universal
automated scientific-judgment engine or proof of production deployment.
