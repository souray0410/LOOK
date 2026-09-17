# LOOK priority owner management environment correction

The initial v3 CPU activation failed with `ModuleNotFoundError` for
`scheduling.project_priority`. Its environment had been reconstructed from an
older start script that omitted the configured management control source.
Checking pinned files and the allocation-command constructor did not exercise
the actual daemon cycle and was insufficient acceptance. The failed activation
is retained; scientific GPU workers and source snapshots were unchanged.

The immutable v4 owner and dispatcher under
`OPS/look_efficiency_20260917/priority_owner_v4/` put the configured
`source_roles_c187560` management package before the historical family package
in their management PYTHONPATH. All management scheduling Python files are
pinned. The original scientific continuation configuration, its environment
(including the three accepted MALLOC settings), and the scientific source are
unchanged. Both new CPU dispatch and the fallback GPU owner receive the corrected
management environment.

The isolated real daemon cycle used current production feeds, read-only claims,
scientific specifications and original source-binding checks. Only its output,
submission lock and journal copy were isolated; submission was prohibited, and
claim-release calls were intercepted without mutation. It returned
`workflow_allocations_active` with `error=null`, found 63 admitted tasks and 12
rejected entries, and resolved 40 admitted native source bindings. Four key
management modules resolved from the same pinned control source. No submission
or claim mutation occurred. The rejection count is preserved, not recast as full
matrix acceptance.

`cycle_validation.json` records that cycle and final cross-checks of all owner,
dispatcher and readiness pins. This verifies the management dependency path;
actual production activation, queued-owner launch and next-lease scientific
recovery have their own subsequent receipts. v4 is frozen after this acceptance;
future changes require another versioned snapshot.
