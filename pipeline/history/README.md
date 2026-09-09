# Pipeline History

Executed or published pipeline steps are never renumbered or silently deleted. When a
scientific design is superseded, its entrypoints move into a timestamped directory here
with their original filenames. Historical steps are provenance records, not part of the
active command chain and must not be run against the current data tree.

The active pipeline always continues from the highest previously allocated number. Its
ledger identifies which steps are active, superseded, or diagnostic.
