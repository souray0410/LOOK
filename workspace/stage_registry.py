"""Immutable study-stage references; never trains, claims work, or unlocks test.

A stage is a research/analysis boundary, not a new training identity. Scientific
compatibility is exact and every reuse/resume requires the owning project's
live verifier. Existing claim/recovery managers remain the only execution owners.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

KINDS = {'training', 'basis', 'fit', 'prediction', 'diagnostic'}
IDENTITY = {'kind', 'scientific_spec', 'data_roles', 'parent_artifacts',
            'source_sha256', 'framework_sha256', 'view'}
RESUME_TRAINING = {'model', 'optimizer', 'scheduler', 'rng', 'data_progress', 'bn'}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def sha(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Invalid SHA256')
    return value


def identity(value):
    if set(value) != IDENTITY or value['kind'] not in KINDS:
        raise ValueError('Incomplete or unknown scientific identity')
    if not isinstance(value['scientific_spec'], dict) or not value['scientific_spec']:
        raise ValueError('Full scientific specification required')
    if not isinstance(value['view'], dict):
        raise ValueError('Explicit inference/diagnostic view required')
    if not isinstance(value['data_roles'], dict) or not value['data_roles']:
        raise ValueError('Data roles and manifests required')
    for role, digest in value['data_roles'].items():
        if role not in {'train', 'internal', 'dev', 'test'}:
            raise ValueError('Unknown data role')
        sha(digest)
    if not isinstance(value['parent_artifacts'], dict):
        raise ValueError('Explicit parent map required')
    for digest in value['parent_artifacts'].values():
        sha(digest)
    sha(value['source_sha256']); sha(value['framework_sha256'])
    # No paths, phase names or deadlines are added by this layer. The caller
    # supplies the full immutable scientific spec; it is never normalized away.
    return fingerprint(value)


def checked_file(root, ref):
    if set(ref) != {'path', 'sha256'}:
        raise ValueError('Artifact reference requires path and SHA')
    sha(ref['sha256'])
    relative = Path(ref['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Artifact paths must be relative to authorized root')
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError('Missing or external artifact')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != ref['sha256']:
        raise ValueError('Artifact digest mismatch')
    return path


def resolve(request, records, root, verify_live):
    """Plan reuse/resume only, with the current owner accepting exact provenance.

    verify_live(record, action) must recheck scientific acceptance or full resume
    compatibility AND shared claims/Slurm liveness. It must return literal True.
    A receipt alone, caller-supplied state, or expired heartbeat is insufficient.
    No unsafe record is silently converted into a fresh duplicate training run.
    """
    wanted = identity(request)
    matches = [r for r in records if identity(r['identity']) == wanted]
    if not matches:
        return {'action': 'new', 'identity_sha256': wanted, 'execution_authorized': False}
    if not callable(verify_live):
        raise ValueError('Project live acceptance verifier required')
    matches.sort(key=lambda r: (r.get('state') != 'accepted', r['run_id']))
    issues = []
    for record in matches:
        state = record.get('state')
        if state not in {'accepted', 'paused'}:
            issues.append({'run_id': record['run_id'], 'state': state})
            continue
        action = 'reuse' if state == 'accepted' else 'resume'
        try:
            if not record.get('files'):
                raise ValueError('Evidence files required')
            if action == 'resume' and request['kind'] == 'training':
                if not RESUME_TRAINING.issubset(record['files']):
                    raise ValueError('Full training resume state required')
            checked_file(root, record['receipt'])
            seen = set()
            for ref in record['files'].values():
                key = (ref.get('path'), ref.get('sha256'))
                if key not in seen:
                    checked_file(root, ref)
                    seen.add(key)
            if verify_live(record, action) is not True:
                raise ValueError('Live project verifier did not accept')
            return {'action': action, 'run_id': record['run_id'],
                    'identity_sha256': wanted, 'artifact_record_sha256': fingerprint(record),
                    'execution_authorized': False, 'copy_required': False}
        except (OSError, ValueError, KeyError) as exc:
            issues.append({'run_id': record['run_id'], 'state': 'needs_review', 'reason': str(exc)})
    return {'action': 'wait_or_review', 'identity_sha256': wanted, 'issues': issues,
            'execution_authorized': False}


def register_stage(directory, manifest):
    """Atomically register an immutable phase without cloning its artifacts.

    Test authorization intentionally remains in the existing evaluator. Different
    stages can reference identical scientific identities and the same old run IDs.
    """
    if set(manifest) != {'schema', 'stage_id', 'protocol_sha256', 'requests', 'previous_stages'}:
        raise ValueError('Invalid stage manifest')
    if manifest['schema'] != 'research_stage_v1':
        raise ValueError('Unknown stage schema')
    stage = manifest['stage_id']
    if not isinstance(stage, str) or not re.fullmatch('[a-zA-Z0-9_-]+', stage):
        raise ValueError('Unsafe stage identifier')
    sha(manifest['protocol_sha256'])
    requests = manifest['requests']
    if not isinstance(requests, list) or not requests:
        raise ValueError('Stage requests required')
    keys = [identity(r) for r in requests]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate request; preserve research questions separately')
    if not isinstance(manifest['previous_stages'], dict):
        raise ValueError('Previous stage mapping required')
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    target = directory / (stage + '.json')
    with (directory / '.stage_registry.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for previous, expected in manifest['previous_stages'].items():
            if previous == stage or not re.fullmatch('[a-zA-Z0-9_-]+', previous):
                raise ValueError('Unsafe or self-referencing previous stage')
            sha(expected)
            old = json.loads((directory / (previous + '.json')).read_text())
            if fingerprint(old) != expected:
                raise ValueError('Previous stage manifest changed')
        if target.exists():
            if json.loads(target.read_text()) != manifest:
                raise ValueError('Stage already locked; create an explicit amendment')
            return target
        fd, temporary = tempfile.mkstemp(prefix='.stage-', dir=directory)
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, target)
            fd = os.open(directory, os.O_RDONLY)
            try: os.fsync(fd)
            finally: os.close(fd)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return target
