# Repository instructions
Read workspace/REPOSITORY_STANDARD.md and docs/development.md before changes. This is a new research preparation branch, not a continuation of archived queues. Do not restore legacy folder layouts or copy historical reports into this branch. Preserve scientific operators and MHD Node semantics. Install the fixed third_party/MHD_Framework as its own package; do not embed it in this project wheel. New training or test access requires the applicable data and study protocol. Shared management files and scripts must match the peer research project. Validate installation, structure, imports, tests and source digests before deployment.

Read docs/architecture.md. src and tests/unit use data, models, methods, training, evaluation, analysis, runtime, studies. Preserve these roles and canonical modules in scripts/check_layout.py. Never put implementation modules at the package root. main holds the accepted unified layout; retire only explicitly superseded transition branches, retain historical reproduction references. Update the pinned MHD release only after strict application output/gradient/state_dict acceptance; never follow floating main.

The installed MHD release selects the API. Import `mhd_framework` / `mhd_framework.utils`; this application pins the V4 release, never floating main. Packaging paths changed; V4 tensor implementations are preserved.

Read workspace/MODEL_RUN_STANDARD.md for model/run identity, timestamps, provenance, reuse and retention. Apply it to all new model families and studies, not only current UKB work. Copying accepted parents is distinct from creating a new training execution. Preserve pinned scientific entrypoints and historical artifacts.

Read workspace/GPU_EXECUTION_STANDARD.md for whole-device budgets, measured concurrency and replacement safety. Project/native admission is separate from architecture correctness.

Read docs/handoff/README.md when continuing work or coordinating with another project. It is a dated snapshot: refresh live bindings and acceptance evidence before action. Update it after deployment, protocol or dependency changes, acceptance failures/fixes, phase completion or material blockers. Ibex is the default full-runtime acceptance environment; WS02 is optional, not a prerequisite. Keep healthy running snapshots unchanged.
