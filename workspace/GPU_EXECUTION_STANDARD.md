# GPU execution standard — version 4

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

Reserve at least four GPU slots for ready finite native-model work. Allocate the
remaining verified account budget between ready LOOK and Radon_Bridge tasks, with
round-robin fairness; idle project reservations can be borrowed by native work.
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
