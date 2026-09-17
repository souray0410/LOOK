# Bounded recovery of the registered q32 delivery

This is a one-off, explicitly bound infrastructure migration, not a general queue
or alternate scientific reader. `recover_allocator.py` verifies the old terminal
Slurm step, exact failure text, run lock, all source/dependency hashes, profile
receipt/files, and shared claim owner/state before resuming that same run. It
keeps the old scientific source and resource ceilings. A second invocation cannot
launch another worker. Its only worker-environment change is libc allocator
configuration; numerical, batch and dataset settings remain unchanged.

The prior small profile did not predict full-cohort host memory. The recovered
formal worker passed all782 development batches and advanced beyond2000 training
fit batches at the last verified checkpoint. That is verified downstream progress,
not a completed fit or proof all subsequent configurations fit memory.

`watch_delivery.py` is a read-only30second observer; it does not own GPU work.
Fresh manager heartbeat cannot hide stale experiment logs/artifacts. Terminal
failures, manager death, resumable pause and600second lack of progress produce
an actionable incident. The existing ukb maintenance automation owns diagnosis
and bounded restoration; unknown deterministic failures are never blindly retried.

Scientific source b6bbc2727255f4cd7c63c2108bb19a8073f137c2; launch-bound profile-reuse
change4fbfbaefdedd8655062c8297399d4c651da1674f. Current read-only observer b72124e.
Full scope and live acceptance boundaries are in docs/handoff on main. This
experiment script intentionally retains exact external operational paths as
recovery provenance; do not copy those paths as another project's defaults.
