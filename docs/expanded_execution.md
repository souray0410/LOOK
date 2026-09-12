# Expanded-cohort execution gates (2026-09-10)

**2026-09-12 update:** the authorized LOOK expansion and current gating order are
in [the three-dataset protocol](../experiments/2026_09_10_11_11_31/MULTIDATASET_PROTOCOL.zh-CN.md).
The earlier allocation shape below is historical. Current requests are independent
single-A100 48h jobs, with account-wide running plus pending GPUs capped at24.
The native prerequisite screen and observed-eye consumer do not yet constitute
accepted expanded-cohort fusion/LOOK execution. Do not use the old bilateral-only
loader or call a native validation job a LOOK experiment.

The user authorized preparation and automatic GPU submission after readiness. Shared
raw-data audit and native-model workflows are maintained in the private Model_Training
repository (`expanded/PROTOCOL.md`, initial implementation 0c6a63b). Research methods,
accepted data manifests, parents, fitted bases and results remain project-owned.

First CPU-only inventory: 93,127 unique imaged participants; 85,664 with at least one
same-visit CFP/OCT paired eye; 84,215 with bilateral pairs. These are archive metadata
counts before phenotype, image-quality, consent/withdrawal and final eligibility
checks, not accepted disease-cohort counts. Both label exports are audited separately.

GPU submission is gated on accepted data/cache, executable frozen task manifests,
preloaded weights, environment/resource checks and tested checkpoint-safe dispatch.
Allocation must immediately execute actual ready tasks. Project work and its exact native-model prerequisites have priority. When a project
ends, is user-paused/cancelled or fails, its freed GPU group automatically starts an
already validated task from the finite baseline-model queue; the other project is
unaffected. When that project is ready again, checkpoint and exit the baseline, then
return the group to the project. Prepare fallback before allocation, preserve its
independent complete model/recovery artifacts, and never change locked project parents
or repeat accepted baselines. Release only when no ready project or baseline task
remains; no placeholder occupancy or indefinite input waiting.
The periodic assistant monitor is not the mechanism that starts an allocated GPU.

Use new participant-level splits and newly trained native parents. Do not initialize
from old UKB models if they were trained on participants in the new test. Preserve
historical exposure metadata; this is internal validation, not an external cohort.
At least one paired eye is proposed; validate masked-eye aggregation before admitting
single-eye participants. Lock finite development tuning and then parent checkpoints
before main project comparisons. Later background candidates do not change an active
comparison generation. No test performance is accessed during preparation or tuning.

LOOK: preserve the accepted missing-modality/correction question and project-specific
loss, selection and missingness protocol.

Current cycle is `2026_09_10_11_11_31`. Allocate three A100 80GB GPUs for
at most72h with salloc: one exclusive GPU per project and one for a finite native
models queue. Freed project lanes may backfill from that queue; returning project
work checkpoints/exits only its own fallback. Do not change locked parent models.
Each Slurm step must have explicit GPU, CPU and host-memory resources; allocated
GPU UUIDs must be distinct. Multi-GPU models require separately validated windows.

The independent Model_Training runtime passed the WS02 two-lane finite acceptance
recorded in experiments/2026_09_10_11_11_31: real Radon_Bridge ResNet18 CFP/OCT3D,
native ResNet50, and ResNet34 fallback. Project pause/return and native SIGKILL
recovery matched uninterrupted full states and logits exactly. This is operational
evidence, not a scientific result or a full LOOK/expanded-cohort trainer acceptance.
CPU tests also validate supervisor crash reattachment without duplicate workers.

The latest native-first protocol in Model_Training `expanded/LAUNCH_PROTOCOL.md`
supersedes the earlier all-adapters-first launch gate. Native parents can start after
their own data/trainer acceptance, while both project adapters remain separately gated.
The frozen candidates cover record-derived glaucoma, cataract and macular degeneration,
with CFP2D, central OCT2D and complete128-plane OCT3D inputs; no balanced subsampling.
Formal observed-eye ResNet50 training supports safe model/BN/optimizer/scheduler/RNG
and sampler recovery. Allocation startup uses concurrent Slurm GPU-isolation checks
and full-resolution resource preflights before scientific training. Job IDs and current
state belong to execution receipts; this document is not evidence that a GPU is running.

Project admission still requires the grouped comparison manifest and an adapter that
consumes the complete participant model, including valid-eye feature aggregation.
Do not pass these new artifacts to the historical fixed-two-eye loader, or silently
substitute a per-eye graph. Accepted parents are copied into project-owned storage.
Historical comparison groups remain required; they are not replaced by the native HPO
matrix. Old UKB parents/test-selected configurations are not used as new native weights.

See [validate before switching](validation_workflow.md) for the host-independent
development process. ws02 is preferred when available, never a mandatory dependency.

Unexpected project termination is also an automatic fallback trigger: observe process
exit, block immediate retry, and dispatch a ready model on the freed lane. A diagnosed
project resumes by a once-only retry command; do not repeatedly restart faulty code.
