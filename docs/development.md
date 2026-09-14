# Development

Create and activate a dedicated Python 3.11 virtual environment, then run
`bash scripts/bootstrap.sh`, `python scripts/manage.py check`,
`python scripts/manage.py test`, and `python scripts/manage.py preflight --machine ibex`.
Preflight checks structure and paths only; it does not certify data, GPU capacity or training.
Integration scripts live under tests/integration and require explicit invocation.

See ../workspace/REPOSITORY_STANDARD.md for new-project reuse and archival rules.
Install the fixed MHD checkout separately; do not bundle framework classes in this wheel.

`python scripts/manage.py verify-env` verifies actual imported framework bytes against the lock and editable application origin.

## Internal architecture

Follow [architecture.md](architecture.md). Keep the source and test role directories identical to the peer application. Run `python scripts/manage.py check` before committing; this checks real modules and import targets, not just top-level folders. Run integration scripts as `python -m tests.integration.<module>` from the repository root. New method-specific files belong under the matching role.

## GPU execution

See [GPU development and batch execution](gpu_workflow.md): ws02 debugging, Ibex A100 preflight and finite batches within one allocation, with project-independent artifacts.

## Target-environment acceptance

Full GPU and end-to-end validation runs on Ibex after resource admission; WS02 is optional debugging, not a required gate. Keep clean-install GitHub checks alongside target-environment acceptance. Maintain the dated [handoff](handoff/README.md), and follow the shared GPU execution standard.
