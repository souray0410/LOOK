# Expanded-cohort execution gates (2026-09-10)

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
Allocation must immediately execute actual ready tasks. Project work and required
native parents take priority over a finite approved background model queue. Empty
queues or errors exit/release; no placeholder occupancy or indefinite input waiting.
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
