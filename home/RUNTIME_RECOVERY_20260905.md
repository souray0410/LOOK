# ws02 cGAN runtime recovery

The original study `63561eb169f2` stopped at 31/52 stages on 2026-09-05
10:34:57 UTC during `paired_cgan_layer3_3407`, direction CFP to OCT. NCCL's
watchdog stopped making progress; rank 1's Python stack was inside Adam's
multi-tensor update. That stack is not sufficient to identify Adam as the cause.
No formal generator checkpoint or completed cGAN case existed at failure.

This operational mitigation pins the ws02 launch environment inside the actual
tmux child: P2P disabled, SHM enabled, device and host cuMem disabled. It also
sets the child's open-file soft limit to 65536 in both launchers and preserves
flight-recorder diagnostics. Explicit caller environment overrides are honored.
Watchdog monitoring and production timeouts are unchanged.

NVIDIA documents disabling cuMem host allocations as a workaround for affected
runtime configurations:
https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2273/user-guide/docs/troubleshooting.html
This supports testing the mitigation; it does not establish this failure's root
cause. The original transport also passed isolated runs, so the original hang
has not been reproduced deterministically.

## Validation

- Original source, formal 64-channel networks, 224-pixel bilateral images,
  112 participants per GPU, two GPUs, accumulation 2, effective batch 448,
  eight loader workers, original training-only internal split and seeds.
- Original transport: one epoch and a separate three-epoch run passed.
- cuMem disabled: a three-epoch run passed. Its generator and discriminator
  state tensors and complete loss history were exactly equal to the original
  three-epoch run.
- Both directions passed ten consecutive epochs without tracing, using only
  training data and its internal reconstruction holdout. Probe outputs are
  isolated maintenance artifacts, not formal generators or scientific results.
- Launcher regression tests execute the generated command through a simulated
  older tmux server with stale transport settings and a 1024-file limit; they
  also check explicit overrides.

## Deployment and provenance

The new operations release is `/home/mengh/LOOK/2026_09_05_16_44_54`.
Use its launchers with `--project-root` pointing at the original study's source
or the existing supplemental source. Neither immutable study source is edited.
All scientific Python modules, math, candidate search, seeds, batches,
checkpoints and test sealing remain unchanged. The original 52-stage and
supplemental 213-new-case identities remain separate.

The recovery audit and isolated probe payloads/logs are stored under
`/data/mengh/LOOK/2026_09_05_09_12_55/runs/maintenance/gan_recovery_20260905`.
Before resume, record hashes of all 31 completed stage results, accepted
backbone checkpoints and the immutable source files. After resume, verify
those hashes and the actual worker environment and file limits. The supplement
still waits for all original 52 stages to complete.

A successful recovery and bounded validation are evidence for the mitigation,
not proof that every possible future NCCL failure has been eliminated.
