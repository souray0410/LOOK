# GPU execution standard — version 1

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
The bounded packed companion is implemented; a universal model/project handover
controller is not certified merely by this policy. Announce deployed capability
and pending acceptance separately.
