# GPU development and batch execution

## Agreed workflow

Use ws02 for bounded functional debugging on explicitly available GPUs. Use Ibex for A100/NCCL acceptance and finite, approved experiment batches. A ws02 pass does not certify A100 memory, software versions or multi-GPU behavior. Each application retains its own source pin, environment, protocol, checkpoints and results.

`sbatch file.sh` submits a new allocation request each time. Running an ordinary Python/shell command does not request Slurm resources. An `srun` step inside an existing allocation uses that allocation; outside one it can request resources. Avoid nested `sbatch` for each trial when the intended execution is a single sequential batch.

For the current two-GPU preparation, both projects use two GPUs sequentially. The completed runtime acceptance already executed LOOK then Radon_Bridge in one allocation. Future scientific batches should similarly contain a finite manifest of ready tasks and immediately dispatch the next eligible task after acceptance. This is a scheduling policy, not authorization for an unspecified training matrix.

## Batch readiness and lifecycle

Before submission, validate source/framework pins, independent project output roots, dataset acceptance, protocol, commands, available storage and retry policy. Estimate useful walltime rather than automatically requesting 48 hours for a short check. Once allocated, run an A100 preflight followed by ready tasks without waiting for manual intervention. A failed infrastructure preflight stops the dependent batch; do not loop on the same deterministic error.

Do not reserve GPUs while waiting for data copying, code edits or an undecided study. Release the allocation when its finite runnable work is finished. Do not generate dummy computation to avoid idle-resource reclamation. Small bounded interactive checks are valid when actively used; long debugging belongs on ws02 where practical.

The batch driver must retain per-task attempts and acceptance receipts. Before the time limit, stop dispatching work that cannot complete safely or cannot checkpoint. A training resume is valid only when its optimizer, scheduler, RNG, sampler/data position and BN state are supported and verified; otherwise preserve the interrupted attempt and restart under the approved protocol. This document does not claim that a general scientific resume dispatcher is implemented.

Use independent job arrays, with explicit concurrency caps, when tasks need independent allocations. A single array submission creates multiple scheduled jobs, not one shared two-GPU allocation; total possible GPU use is GPUs per task times concurrent tasks. Increasing concurrency is a resource decision, not an automatic consequence of adding configurations.

## Provenance

Record allocation/job ID, task ID, source and framework commits, environment, rank count, selected devices, effective/microbatch, timing, peak project memory and acceptance state. Keep outputs project-local. Do not conflate successful runtime checks with trained-model performance or introduce holdout access through a debug command.

References: [Slurm sbatch](https://slurm.schedmd.com/sbatch.html), [job arrays](https://slurm.schedmd.com/job_array.html).
