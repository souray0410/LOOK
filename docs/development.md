# Development

Create and activate a dedicated Python 3.11 virtual environment, then run
`bash scripts/bootstrap.sh`, `python scripts/manage.py check`,
`python scripts/manage.py test`, and `python scripts/manage.py preflight --machine ibex`.
Preflight checks structure and paths only; it does not certify data, GPU capacity or training.
Integration scripts live under tests/integration and require explicit invocation.

See ../workspace/REPOSITORY_STANDARD.md for new-project reuse and archival rules.
Install the fixed MHD checkout separately; do not bundle framework classes in this wheel.

`python scripts/manage.py verify-env` verifies actual imported framework bytes against the lock and editable application origin.
