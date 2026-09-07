# Flexible LOOK execution

Pipeline 48 schedules one independent frozen case per selected physical GPU. There
is no fixed number of cards and no requirement to use indices 0 or 1. A single
case still sees exactly one CUDA device. Extra cards increase case concurrency;
they do not change model batch size, seeds, training objectives, or LOOK selection.

## Runtime control

Set `PYTHONPATH` to the **frozen worker release** `tool` directory. The controller
entrypoint may reside in a newer release. Always retain the same worker root,
scientific plan and queue output when changing device allocation.

```
python CONTROLLER/pipeline/48_run_flexible_gpu_queue.py \
  --control RUNTIME/devices.json --set-devices 3
python CONTROLLER/pipeline/48_run_flexible_gpu_queue.py \
  --control RUNTIME/devices.json --set-devices 1 3
python CONTROLLER/pipeline/48_run_flexible_gpu_queue.py \
  --control RUNTIME/devices.json --set-devices 0 2 3
```

These are alternative examples, not automatic requests to occupy these cards.
The current ws02 policy explicitly selects 0 and 1. Device indices are validated
and their NVML UUIDs pinned; unavailable or duplicate indices are rejected.

The default `--mode drain` stops assigning new cases to removed cards and releases
them after their active case completes. `--mode interrupt` releases removed cards
by stopping only controller-owned process groups. A valid atomic completion wins a
simultaneous interruption; unfinished work resumes from durable checkpoints. Work
since the last durable checkpoint can be repeated. Retained cards continue without
interruption. Adding cards admits ready cases automatically. An empty
`--set-devices` list pauses new dispatch; combine with interrupt for urgent release.

```
python CONTROLLER/pipeline/48_run_flexible_gpu_queue.py \
  --worker-root WORKER --plan FROZEN_PLAN --output EXISTING_QUEUE \
  --lock-path SHARED_LOCK --control RUNTIME/devices.json \
  --execution-acceptance EXECUTION_VERIFICATION \
  --collect-resumed-starts
```

Use `--collect-resumed-starts` only for the pipeline 47 combined study. Historical
pipeline 46 retains its fixed 0/1 contract. Scientific implementations are imported
from WORKER, with source manifest checks; the new execution adapter is audited
separately. Devices and controller identity are excluded from scientific identity.
Existing receipt hashes and dependencies are validated before reuse.

## Project memory and recovery

14 GiB is **LOOK's aggregate process memory per card**, not total card usage.
All identifiable LOOK processes, including children and other LOOK queues, count.
A shared virtual-environment executable does not make another project LOOK.
NVML totals are partitioned into LOOK, other-project and unknown ownership. Other
projects affect available memory only. Unknown ownership prevents new dispatch but
does not trigger a LOOK over-budget kill. The controller never signals unrelated
projects. A new case requires at least 14 GiB free and no existing LOOK worker on
that card. The torch allocator is capped at 11 GiB to leave 3 GiB for contexts and
non-allocator memory; actual LOOK NVML occupancy remains the enforcement criterion.
Polling is a watchdog, not hardware memory isolation: a transient overshoot between
samples cannot be excluded. Violations stop the owned case and preserve evidence.

SSF retains its bounded 8/4/2/1 execution-microbatch retry sequence, persisted in
queue status. Frozen inference batch sizes are retained because changing them can
change TF32 outputs. Unrecoverable OOM blocks the queue without raising the budget,
changing PCA/search ranges, or silently skipping samples. Pausing for device removal
is recorded separately from OOM. Invalid control files prevent dispatch while active
cases may finish. Atomic status, an append-only event log and pre-restart snapshots
record all transitions; completed results are never relabelled or overwritten.

## Validation scope

CPU subprocess regressions cover arbitrary/noncontiguous single, dual and three-card
schedules, live drain and interrupt, pause/resume, unchanged retained worker PID,
receipt reuse, UUID mismatch and project-specific memory attribution. Actual GPU
checks on ws02 cover available GPU 0/1, two missing directions, serial/parallel
prediction equality and changing selected devices. More than two physical cards are
not available on ws02; that scheduling path is verified with CPU worker simulation.
Test cohorts remain sealed. New hardware may require a small numerical acceptance
check; bitwise equality across GPU architectures is not promised.
