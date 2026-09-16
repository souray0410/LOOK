"""Resumable resource qualification, independent of the allocation lease."""
import json
import subprocess
from pathlib import Path
from look.runtime.state import atomic_write_json


def run_profile(command, environment):
    result = subprocess.run(command, env=environment, check=False)
    if result.returncode == 75:
        raise SystemExit(75)
    result.check_returncode()


def request_expiry_pause(run, attempt):
    # Different historical workers used attempt-local profiles. New qualification
    # lives under the scientific run and therefore survives attempt rollover.
    for root in (Path(run), Path(run)/'resource_profile', Path(attempt)/'profile'):
        atomic_write_json(dict(reason='allocation_expiry_checkpoint'), root/'pause.json')


def profile_pause_state(run, profile):
    """Do not turn an interrupted preflight into scientific failure/completion."""
    p = Path(profile)/'status.json'
    if not p.exists() or json.loads(p.read_text()).get('state') != 'paused':
        raise ValueError('Exit75 needs an actual paused profile receipt')
    atomic_write_json(dict(state='paused',stage='resource_profile',test_access=False,
                           scientific_acceptance=False),Path(run)/'status.json')


def adopt_profile(source, destination, spec):
    """Explicitly migrate a closed, identical preflight without rewriting paths.

    Candidate prediction paths are historical evidence. A canonical directory
    link preserves those bytes; no machine-specific read branch is introduced.
    The scientific writer's own lock must be free before migration.
    """
    import fcntl
    source, destination = Path(source).resolve(), Path(destination)
    with (source/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if json.loads((source/'spec.json').read_text()) != spec:
            raise ValueError('Qualification belongs to a different specification')
        status = json.loads((source/'status.json').read_text())
        if status.get('state') not in ('paused', 'completed'):
            raise ValueError('Qualification is not safely closed')
        if destination.is_symlink() and destination.resolve() == source:
            return  # Idempotent migration.
        if destination.exists() or destination.is_symlink():
            raise ValueError('Never overwrite an existing resource profile')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source, target_is_directory=True)
        atomic_write_json(dict(source=str(source), destination=str(destination),
            state='migrated', scientific_acceptance=False),
            destination.parent/'resource_profile_migration.json')
