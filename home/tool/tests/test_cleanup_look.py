import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest


def test_cleanup_manifest_scope_dry_run_and_repeat(tmp_path, monkeypatch):
    import look_core.cli as cli
    runs, cache = tmp_path / 'runs', tmp_path / 'cache'
    summary = runs / 'reviewed_look/abc/summary.json'
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({'status': 'failed', 'active_plan_id': '0123456789ab'}))
    sweep = runs / 'sweeps/validation__0123456789ab'
    sweep.mkdir(parents=True)
    (sweep / 'study_plan.json').write_text(json.dumps({'phase': 'validation',
        'grid': {'look_profiles': [{'enabled': True}]}, 'experiment_ids': ['old_look']}))
    experiment = runs / 'experiments/old_look'
    experiment.mkdir(parents=True)
    (experiment / 'old.pt').write_bytes(b'obsolete')
    protected = runs / 'backbones/keep.pt'
    protected.parent.mkdir(parents=True)
    protected.write_bytes(b'keep checkpoint')
    monkeypatch.setattr(cli, 'resolve_runtime_arguments', lambda a: SimpleNamespace(runs_root=runs, cache_root=cache))
    script = Path(__file__).resolve().parents[2] / 'pipeline/35_clean_superseded_look.py'
    main = runpy.run_path(str(script))['main']
    monkeypatch.setattr(sys, 'argv', [str(script), '--summary', str(summary)])
    main()
    assert experiment.exists()
    monkeypatch.setattr(sys, 'argv', [str(script), '--summary', str(summary), '--execute'])
    main()
    assert not experiment.exists() and not summary.exists()
    assert protected.read_bytes() == b'keep checkpoint'
    main()
    assert len(list((runs / 'maintenance').glob('*.json'))) == 1


def test_cleanup_rejects_nonreviewed_summary(tmp_path, monkeypatch):
    import look_core.cli as cli
    monkeypatch.setattr(cli, 'resolve_runtime_arguments', lambda a: SimpleNamespace(runs_root=tmp_path, cache_root=tmp_path))
    script = Path(__file__).resolve().parents[2] / 'pipeline/35_clean_superseded_look.py'
    monkeypatch.setattr(sys, 'argv', [str(script), '--summary', str(tmp_path / 'backbones/summary.json'), '--execute'])
    with pytest.raises(ValueError, match='reviewed LOOK summary'):
        runpy.run_path(str(script))['main']()
