# Validate before switching a running experiment

Prefer ws02 for development GPU checks when the necessary cards are available.
Host availability is not an execution prerequisite: use another authorized GPU or
an explicitly scoped Ibex validation step when ws02 is busy. CPU checks come first;
CPU success is not GPU acceptance. Do not stop other users' jobs to obtain a test GPU.

Keep the accepted experiment running while preparing and validating new code when
resources permit. Freeze the validated code/configuration and source hashes, then
checkpoint and exit only the affected old task. Deploy to a new attempt and perform
the actual-device preflight before the long workload. Do not edit running source,
change a running experiment's parameters in place or swap locked parent weights.

When validation requires the same exclusive GPU, pause the user's own baseline lane
at a safe checkpoint boundary and use a finite, declared validation step. Follow the
allocation's resource boundaries. Do not presume reserved VRAM permits an arbitrary
concurrent Slurm process on someone else's GPU. Retain the other project unchanged.

The developer host and requested GPU count are configuration, not hardcoded protocol
requirements. Full checkpoint/recovery remains necessary for walltime, worker/node
failure and deliberate interruption even with careful pre-deployment validation.
Single-GPU validation does not certify distributed recovery or a different hardware
precision path. Re-run only acceptance affected by a change; preserve all attempts.
