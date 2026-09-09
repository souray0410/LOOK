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

## Interactive validation preference (2026-09-09)

The user prefers an interactive `salloc` allocation for the next development/validation session. Launch compute commands through `srun` inside that allocation. The same resource, readiness, walltime and release rules apply to interactive and batch jobs; an interactive shell does not make allocation duration unlimited.

For finite MHD V4 versus native PyTorch verification, select microbatch from actual GPU capacity. About 50% of total device memory is an initial workload sizing target, never a required minimum occupancy. Measure full forward/backward/optimizer peaks with transient headroom and at least 10 GiB free. Keep the reference and MHD inputs, weights, batch and numerical settings matched. Record the selected batch separately for each device class; this is not permission to alter scientific training batches. Memory allocation alone does not demonstrate useful GPU work.

New explicit user tasks take priority over the finite framework verification list. Handoff must occur at a bounded test boundary or a verified checkpoint, and the old process must release its CUDA context. Suspending a process without freeing GPU memory is insufficient. Completed cases are not repeated to retain resources. The adaptive validator and cooperative priority controller still require implementation and acceptance before use; the JSON policy records intended behavior, not a running service.
