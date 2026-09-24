# LOOK V5 monitor runtime pin

Source commit `bd670b463009669784522f6cd72a6fae1283ae03` adds a small
bootstrap verifier and a deterministic package builder for the long-lived
Ibex V5 monitor. Before importing the monitor, the bootstrap checks the exact
binding and every file named by its runtime manifest. The prepared Ibex
package is under
`/ibex/user/mengh/LOOK/look_v25_runtime_pinned_bd670b4_20260924/candidate/look_cataract_middle_v5_feature_chain_sharded_v25`.

The focused local tests passed, and the remote package verifier passed with
binding SHA256
`02034f5b93e7e21b32886b08edae5c38d9a84ba52c4a9ac87dadc4fcf0370c78`.
Changing a source file after package construction makes verification fail.

This package supports the held-first zero-GPU monitor transaction documented
in the private PHD acceptance receipt. It does not claim that the pending V5
GPU job has started or that the full feature cache and model training are
complete.
