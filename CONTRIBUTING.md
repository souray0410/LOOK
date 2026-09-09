# Research development workflow

1. Develop on the declared timestamp branch; archive prior source and evidence before structural migration.
2. Keep importable code under src, configuration under configs, commands under scripts and tests under tests. Do not reintroduce tool/pipeline/home/root-package alternatives.
3. Install the fixed MHD submodule independently; V4/V5 choice is explicit and source locked.
4. Use the same bootstrap/check/test/preflight commands as the peer research repository.
5. Separate synthetic implementation tests from data acceptance and scientific evaluation. Missing data or an unlocked protocol is not a completed experiment.
6. Commit, push the branch, deploy its exact source and verify digests. Keep datasets, checkpoints, participant predictions and environments outside Git.

See workspace/REPOSITORY_STANDARD.md for the shared contract. Old studies run from archived source and their original environment; they do not define the new study by default.
