# Staged research, checkpoint continuation and exact artifact reuse

Study stages define scientific scope and delivery. They do not reset the experiment
identity or clone accepted work. LOOK and Radon_Bridge share this management contract;
MHD model implementations remain independent of either research project.

## Identities and extension

- Architecture, full scientific recipe, training execution, accepted artifact,
  inference/diagnostic view and research stage are separate entities.
- A stage lists immutable scientific requests and hashes of predecessor stages.
  Later stages reference existing run IDs and artifacts. A new stage name, report,
  deadline or repository folder alone never creates a fresh training execution.
- Match the complete scientific spec: disease/label/data split/preprocessing,
  parent checkpoints/bases, model/MHD/Node mapping, seed, precision, batch and BN,
  optimizer/schedule/stopping/selection rules, scientific source and view.
- Different data, 2D/3D input, framework or training settings create a new scientific
  request. Deliberate warm-start/transfer is a new run with recorded parent, not an
  exact resume and not evidence that the old experiment has been reproduced.
- Multi-bridge/six-network/other-organ expansion reuses matching complete native
  parents. Changing an upstream host changes its feature distribution: refit
  dependent bases/corrections when the protocol requires it. Never reuse LOOK
  PCA/correction parameters across incompatible host checkpoints.
- Epoch/optimizer/RNG/data progress and selection state remain tied to the original
  run across interruptions, allocations and research-stage boundaries. Existing
  shared claim and Slurm-liveness managers are the only execution owners.

## Incremental work

For each request, resolve to one of: reuse accepted artifact; resume original paused
run; wait/review active, failed or unverified work; or create a genuinely new request.
A failed hash/receipt never silently falls through to a duplicate training run.
Completed training requires its original domain acceptance, not merely last.pt or
an epoch count. A paused training artifact requires model, BN, optimizer, scheduler,
RNG and data progress plus the domain verifier's full resume compatibility and
liveness check. The domain verifier also checks other original protocol state
(e.g. scaler, augmentation/missingness RNG, metric history and best checkpoint).

Predictions can be reused only for the same checkpoint, preprocessing, participant
manifest/order and actual inference state. Diagnostics include intervention/view
identity and RNG. Additional statistics may reuse predictions; they retain distinct
analysis protocol/multiplicity/selection provenance. No project phase averages
across differing training regimes as if matched.

## Current implementation boundary

`workspace/stage_registry.py` supplies:
- immutable, atomic, process-locked stage registration and predecessor hashes;
- exact scientific request fingerprints and root-relative artifact references;
- file/receipt SHA checks and explicit project live-verifier callback;
- reuse/resume/wait-or-review/new planning with unchanged original run ID.

It never launches training, steals a claim, copies a checkpoint, deletes evidence or
unlocks test. Scientific adapters must construct the full identity from their
existing canonical specs and bind `verify_live` to actual project/native acceptance
and current execution ownership; a lambda returning True is not production
acceptance. Existing native/project verifiers remain authoritative. The management
unit tests validate only this layer; they do not certify full training-resume
numerical equivalence or deployment into all production controllers.

Complete pipeline integration still requires immutable prepared snapshots and
real end-to-end acceptances: same requested stage executed twice adds zero training;
extension dispatches only genuinely new identities; restored training preserves
model/BN/optimizer/RNG/data/best/selection/Node state; existing wrong/corrupt evidence
is quarantined; stage changes never alter labels or unlock test. Only after these
pass should the corresponding production adapter be marked deployed.

## Stage-specific evaluation

A first-paper stage can be locked and evaluated while long-term experiments remain.
Lock its complete required models, comparisons and data-use audit before accessing
test. Keep a cross-project test-access ledger. Once viewed, that test is not unseen
for later method selection; new independent confirmation needs unused data or an
external cohort. Registration in this utility grants no access permission.

## Retention

Keep referenced parents, accepted artifacts needed for comparisons/reproduction,
original stage manifests, source/environment locks and active/paused resume state.
Cleanup is dependency-aware and separate from the scheduler. A higher later score
never authorizes deleting the only checkpoint underlying an earlier claim. Report
and Git references are not substitutes for restricted checkpoint storage.
