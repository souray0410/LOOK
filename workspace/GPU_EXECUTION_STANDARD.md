# GPU execution standard — version 5

## Capacity is a whole-device budget

For the current80GiB A100 allocation, keep approximately10GiB physically free after
all admitted workloads, i.e. total work may consume approximately70GiB. This is
neither a10GiB per-model limit nor a minimum memory-utilization target. On another
device/allocation recalculate the available budget, CPU and host-memory capacity;
never assume the same concurrent model count or ignore unrelated GPU users.

Use measured complete-workload peaks plus safety margin to admit concurrency.
Include initialization, optimizer state, forward/backward, evaluation, temporary
workspaces, CUDA contexts, and checkpoint transitions. A framework allocator limit
is not total physical memory accounting. Count simultaneously resident processes
and models across project/native workers on the device, not just one process.
Conservative admission requires current resident work plus the incoming peak and
transient margin to fit below the total budget; missing measurements require a
bounded probe with sufficient headroom, not an optimistic declaration of safety.

The current native2D companion starts from the observed approximately5.6GiB footprint
and adds workers sequentially, at most four total for eight CPU cores with two torch
threads per worker. This is an initial measured resource class, not a permanent
four-model ceiling for every server or architecture. More concurrency is useful only
if aggregate productive throughput improves within CPU, memory, I/O and GPU limits.
Record samples/sec, optimizer updates, phase, GPU utilization and total memory;
initial evaluation and training phases must not be compared as identical throughput.

Keep scientific batch/accumulation, precision, BN, seed and optimizer rules fixed
when changing only packing. If a larger batch or AMP is proposed, freeze and validate
a separate recipe with matched controls; do not silently alter a running study.
Every independent run owns its checkpoint, random state, logs and artifact identity.
No duplicate claims or swapping the selected parent based on unfinished searches.

## Safe replacement without an avoidable idle gap

1. Prepare versioned replacement code/config independently while old work continues.
2. Check the combined old/new peak, CPU, host memory and total-device free reserve.
3. If overlap cannot fit or capacity is uncertain, retain old work and report the
   deficit; do not stop the old task merely to try whether the new task fits.
4. If capacity permits, start a bounded new-worker probe and require actual useful
   forward/backward/update (or full inference/correction for non-training work),
   health and checkpoint checks. A launched PID alone is insufficient.
5. Only after acceptance, checkpoint and stop the superseded worker. New failure
   leaves old work running; cleanup and recovery belong to the failed new attempt.

Task replacement must not cancel the allocation owner or other projects. A Slurm
allocation ending, node failure or external cancellation can still terminate all
its steps; overlap is not a guarantee against infrastructure failure. Preserve full
resume state and the allocation/attempt record. Never use artificial occupancy or
unbounded duplicate runs as a fallback when real authorized work is unavailable.

## Independent projects and native parents

A device is a scheduling resource, not permanently reserved for a project name.
While project parents are unavailable, all allocated lanes can train finite,
approved native candidates. When project workloads become ready, use the above
admission and handover contract. Native-model source/weights can be reused by exact
identity, while each project owns its adaptations, fine-tuned states and evidence.

Profile actual complete project paths before admission. LOOK requires its real
feature extraction, missingness/correction/PCA and evaluation path. Radon_Bridge
requires the complete two-branch graph, compressed projection/mixer/return, loss,
backward and optimizer state, including the largest admitted bridge configuration.
Single-native-model memory does not certify either project. Old workstation caps
remain historical behavior; new production entrypoints must explicitly bind the
current resource profile and cannot inherit a silent9/10/12GiB cap.

Current project-native adapters still require large-cohort production acceptance.
The measured companion controller is separate from frozen scientific workers;
its accepted resource profiles do not certify a universal model/project handover. Announce deployed capability
and pending acceptance separately.

## Measured dynamic companion admission

MHD_Models/scheduling implements finite companion ownership inside existing
allocations. No replacement allocation is required. Profile initialization, five
warmup plus twenty optimizer updates, validation, checkpoint save and reload, and
compare the resumed next update exactly with uninterrupted execution. Profiles
key architecture, input/max eyes, micro/effective batch, precision, optimizer,
framework/trainer and hardware; LR-only variants may share a mapped profile.

Use B=min(0.875*T,T-10GiB). Require conservative existing resident peaks plus
1.2 times incoming measured peak plus2GiB to fit B. Include unknown physical GPU
load once. Preserve original two threads/worker, allocated CPU, and15% host working
memory headroom. Record full cgroup charge separately; only clean inactive file
cache may be discounted as reclaimable, never dirty/writeback/anonymous memory.

Add one execution at a time. Compare three120-second stable training windows before
and after addition using aggregate committed participant progress divided by each
run's train size and wall time. Require at least5% median improvement; otherwise
checkpoint and pause the newest execution. Unstable observations time out after
15minutes. GPU utilization or memory occupancy alone is not the objective. Resource
limits, not a permanent four-worker constant, bound concurrent admissions.

Use shared atomic claims with immutable run/config identity and allocation/step.
Never steal stale claims. Reconcile actual step death, original acceptance receipts
and plateau evidence before reuse. Preserve healthy primary owners, freeze their
queue ownership, and retire companions ahead of unprofiled primary transitions.
Pause only failed/newest owned work for scoped faults or reserve pressure. Begin
checkpoint retirement900seconds before the allocation deadline. Manager restart
reconciles live steps; missing launch identities require review. Keep all handover,
resource, throughput and recovery receipts without altering scientific run IDs.

## Current allocation contract (2026-09-13)

All new LOOK, Radon_Bridge and native-model allocations request at least48hours,
by default48hours, with16CPU cores and128GiB host memory per GPU unless a separately
accepted full-workload profile requires more.12h/24h manifests remain historical
reproduction records; they do not authorize new submissions. Existing healthy
allocations finish normally. Slurm's time limit is never a training completion rule.

Project work and its explicitly registered prerequisite models take priority over
generic model exploration. Under the latest 2026-09-15 user instruction, preserve four generic-model GPU
slots within the verified 24-GPU budget. LOOK and Radon_Bridge each have a
ten-GPU target cap. Allocate only against current executable demand; all unused
project capacity is available to generic models. One project does not automatically
borrow the other project's ten-slot target. Reduced actual account limits constrain
new admissions; live over-budget jobs retire normally, never by forced cancellation.
Do not count unimplemented experiment positions as executable GPU demand.
Count actual requested/allocated GPUs, not job count. Running, pending and unresolved
submission intents all consume the local budget. Read actual Slurm association,
QOS and account inventory; a24GPU local ceiling is not a universal school policy.
Future admission must use the shared account lock, ownership journals and role policy.

Keep at least15% of allocated host memory free. Native formal workers require at
least40GiB step memory, increased by actual profiling. Co-resident work requires
measured throughput benefit; a high memory footprint alone is never an objective.

A complete source-parallel experiment uses the least feasible number of GPUs inside
one allocation. Never join unrelated allocations into one model. Placement and
transfer transport are part of recovery provenance. Validate loss, BN, gradients,
updates and full recovery against a runnable single-device fixture. Do not silently
change effective batch or precision to make a configuration fit. The production
resource gate covers every participating device and the complete model lifecycle.

Healthily running source snapshots remain immutable. The current research expansion
must pass source-specific runtime gates before it can displace native work. A protocol
registry, a submitted allocation, a unit test or a source-parallel toy fixture does not
certify real-data scientific completion or production memory admission.

## Acceptance environment and handoff (2026-09-14)

Ibex is the default and authoritative target for complete runtime acceptance of
Ibex workloads. WS02 is optional development/debugging, not a prerequisite.
Keep clean-environment GitHub installation/unit checks: they detect undeclared
dependencies that an established server environment may hide. A CPU CI pass does
not certify GPU, data, recovery, production deployment or scientific completion.

Use an admitted independent validation step inside an appropriate allocation,
never a GPU workload on a login node. Verify actual input dimensions, forward,
backward, optimizer update, validation, checkpoint save/reload/resume, MHD nodes,
BN/RNG state, diagnostics and reporting. Record code/config/data/framework SHA,
hardware, CUDA/software environment and complete GPU/host peaks. Multi-device
checks must use the intended Ibex placement; CPU Gloo success or a WS02 check
cannot stand in for that acceptance. Respect data roles and the sealed test gate.
Protect healthy workers and allocation owners. A changed resource class needs
new acceptance; do not alter scientific batch/precision/stop rules to force it.

Each repository maintains docs/handoff/README.md, linked from its root README
and AGENTS.md. Update it after deployment, protocol/dependency changes, acceptance
failures/fixes, phase completion or material blockers. Record the checked time,
preparation and deployed versions separately, evidence links, remaining gates,
next actions and cross-project artifact contracts. Consumers refresh the actual
bindings, Slurm steps and acceptance records before acting; a handoff snapshot
is not a live status service or permission to modify another project's queue.
Only aggregate and non-identifying references belong on GitHub.


## Project-first demand and parallel experiments (2026-09-15)

Allocation admission and task granularity are separate requirements. Independent
configurations, matched controls, seeds and hosts must be individually claimable
and able to run concurrently across cards. Preserve real dependencies: accepted
parents before host/bridge work, accepted host before frozen LOOK fitting, upstream
progressive correction before downstream fitting, and complete matched results
before group statistics. Shared immutable inputs do not imply a serial dependency.
Do not change scientific identity, data order or statistical weights to parallelize.

Fresh project dispatch admission records include ready required parent training.
A single CPU policy publisher may update the existing role-budget input under the
shared submission lock; it must not submit, claim, cancel or become another GPU
owner. Existing executors still enforce source, resource, recovery and test gates.
Running and pending allocations remain occupied; healthy work is not cancelled to
rebalance quotas. With ready demand, future released slots go to projects first.
If demand, ownership or account state is uncertain, hold new generic submissions
and retain healthy execution while investigating. Generic candidates retain their
original run and checkpoint when a validated safe handover pauses them.

Report allocated project slots, actual project work, required-parent work and generic
models separately. A dynamic admission budget does not prove that all research
arms have been split into independently dispatched tasks. Record unimplemented
parallel case boundaries and resource handovers in the acceptance ledger.


## Resource leases are independent of scientific executions

This is a permanent design and release requirement for every project, native
model, data-fitting pipeline, evaluation and future framework version. Under the
current allocation policy, request 48 hours. A resource lease expiring does not
complete, reset or create a scientific execution. Reaching a protection cap is
not acceptance either. Historical request manifests remain evidence and cannot
be replayed as new shorter or unguarded submissions.

Use one durable task state machine across allocations: eligible -> atomically
claimed -> running -> checkpointed/paused or acceptance-pending -> accepted.
Failed/unknown states require scoped diagnosis. On a new grant, re-read the finite
approved queue, dependency receipts, live claims and recovery evidence. Verify and
skip accepted tasks; resume eligible unfinished executions with the same run ID,
configuration, seed and source; otherwise claim the next ready task. A new
allocation/attempt ID is provenance, not a new replicate. Missing required recovery
state must not silently trigger fresh initialization.

Begin graceful retirement at least 900 seconds before the actual Slurm EndTime,
earlier when measured save/validation latency requires it. Keep regular durable
checkpoints during work: a final signal alone cannot protect against node failure.
At an optimizer boundary persist model/BN, optimizer, scheduler/selection/platform
state, AMP scaler when applicable, random streams, sampler/data position and
accumulation state or a verified empty-accumulation boundary. Atomically publish a
complete checkpoint; retain a verified predecessor until replacement is durable.
Changing hardware or placement requires recorded numerical/resource acceptance,
not an unqualified claim of bitwise reproducibility.

Non-training work also has recovery units: validated PCA/SVD entries, progressive
correction candidates/sites, prediction shards, diagnostics and report stages.
Bind every completed unit to inputs, configuration, source and output hashes.
Reuse only accepted units with satisfied dependencies. An interrupted unsaved
unit may be recomputed from its last valid boundary; explicitly report its
recovery granularity and maximum lost work. Do not claim batch-exact fitting
resume from an entry-level cache. If a unit repeatedly exceeds a lease without
committing progress, isolate that problem and implement accepted finer recovery;
do not repeatedly allocate and redo it without progress.

Duration admission asks whether a validated useful checkpointable work segment
fits, including load/probe/save/exit reserves, not whether the entire model will
converge within one lease. Long tasks must not be permanently starved by full-run
estimates. This does not lower the original epoch, plateau or performance rules.
Resource feasibility and minimum useful segment measurements remain required.

Before reassignment establish old allocation/step death using the scheduler and
accounting; stale heartbeats or a missing local client alone are insufficient.
Unknown submission/ownership blocks new claims. A verified TIMEOUT, preemption or
node failure can resume the unchanged run from valid state. OOM, corrupted state,
scientific protection-cap and deterministic software failures require their own
reviewed repair; no blind restart. A failure of one task does not cancel healthy
workers or the owner. Release resources when no real safe work is available.

Acceptance must exercise interruption/restart at training and non-training stage
boundaries, exact next-update/state equivalence where applicable, completed-unit
reuse, corrupt/missing checkpoints, terminal-versus-unknown ownership, concurrent
claim races, manager restart, and complete-task-to-next-task dispatch. Record
implemented, tested, deployed, live-restored and scientifically accepted separately.
Tests or a policy document alone never certify all current executors. Preserve old
immutable workers until validated retirement; use explicit versioned migration for
changed contracts. Keep a stage-by-stage evidence ledger and automatic continuation
for unfinished recovery gates. Apply this review before every new workflow release.
