# Unified model artifacts and version boundaries

The shared [model standard](../workspace/MODEL_RUN_STANDARD.md) is version 4.
Every new model artifact uses one canonical manifest/configuration/state format.
Server location does not select a loader. Old study releases finish their locked
work using their own code; conversion happens outside normal project runtime.

MHD_Models owns explicit source-to-canonical migration and its strict reader.
MHD_Framework retains exact architecture and API pins; no V5 upgrade is implied.
The project retains its own derived weights and an immutable parent-artifact map.

Transition acceptance requires: source completion and hashes, state-preserving
migration, exact framework reconstruction, original Node IDs, complete task head
and aggregation, matching development predictions, and preserved parent references.
No selected checkpoint, split, training recipe or test-access gate changes merely
because a package or directory changed. Restore/replay of historical projects uses
the archived project release, not inline legacy loaders in a new release.

Current running production snapshots have not been switched by this document.
New canonical artifact consumption must be accepted before promoting the corresponding
project executor. This is an explicit deployment gate, not an automatic fallback.
