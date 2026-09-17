import json
from pathlib import Path
import pytest
from look.analysis.cumulative_delivery import collect, publish
from look.runtime.state import file_sha256


def snapshot(n=0):
    return dict(cases=[dict(run='/approved/run', mode='best_forward', state='awaiting_configuration_acceptance')],
                rows=[], revision=n)


def no_figure(data, path):
    pass


def test_idempotence_and_cumulative_revision(tmp_path):
    assert publish(snapshot(), tmp_path, no_figure)['state'] == 'published'
    first = (tmp_path/'current').resolve()
    assert publish(snapshot(), tmp_path, no_figure)['state'] == 'unchanged'
    assert publish(snapshot(1), tmp_path, no_figure)['state'] == 'published'
    assert (tmp_path/'current').resolve() != first
    assert (first/'manifest.json').exists()


def test_render_failure_preserves_last_valid_view(tmp_path):
    publish(snapshot(), tmp_path, no_figure)
    before = (tmp_path/'current').resolve()
    def failure(data, path):
        raise RuntimeError('render failed')
    with pytest.raises(RuntimeError):
        publish(snapshot(1), tmp_path, failure)
    assert (tmp_path/'current').resolve() == before


def test_corruption_rejected_and_pointer_recovery(tmp_path):
    publish(snapshot(), tmp_path, no_figure)
    target = (tmp_path/'current').resolve()
    (tmp_path/'current').unlink()
    assert publish(snapshot(), tmp_path, no_figure)['state'] == 'recovered_publication'
    (target/'results.csv').write_text('altered')
    with pytest.raises(ValueError, match='Published result changed'):
        publish(snapshot(), tmp_path, no_figure)


def test_pending_cases_are_not_metric_rows_and_changed_config_rejected(tmp_path):
    run = tmp_path/'run';run.mkdir()
    spec = run/'spec.json'
    spec.write_text(json.dumps(dict(host={'seed':3416},mode='best_forward',spatial_factors=[16],latent_dims=[32])))
    config = tmp_path/'config.json'
    config.write_text(json.dumps(dict(task=dict(run_dir=str(run),spec=str(spec)))))
    sequence = tmp_path/'sequence.json'
    entry = dict(config=str(config),sha256=file_sha256(config))
    sequence.write_text(json.dumps(dict(tasks=[entry])))
    def forbidden(*args):
        raise AssertionError('Pending case cannot be accepted')
    data = collect(sequence, forbidden)
    assert not data['rows']
    assert data['cases'][0]['state'] == 'awaiting_configuration_acceptance'
    sequence.write_text(json.dumps(dict(tasks=[entry,entry])))
    with pytest.raises(ValueError, match='Duplicate run'):
        collect(sequence, forbidden)
    sequence.write_text(json.dumps(dict(tasks=[entry])))
    config.write_text('{}')
    with pytest.raises(ValueError, match='configuration changed'):
        collect(sequence, forbidden)


def test_accepted_delivery_accumulates_all_cells_and_renders(tmp_path):
    run = tmp_path/'run';run.mkdir()
    spec = run/'spec.json'
    spec.write_text(json.dumps(dict(host={'seed':3416},mode='best_forward',spatial_factors=[16],latent_dims=[32])))
    (run/'accepted.json').write_text('{"state":"accepted"}')
    config = tmp_path/'config.json'
    config.write_text(json.dumps(dict(task=dict(run_dir=str(run),spec=str(spec)))))
    sequence = tmp_path/'sequence.json'
    sequence.write_text(json.dumps(dict(tasks=[dict(config=str(config),sha256=file_sha256(config))])))
    delivery = run/'delivery';delivery.mkdir()
    results = [dict(pattern=p,method=m,metrics={'macro_f1':.6 if m=='host' else .7})
               for p in ('oct_missing','cfp_missing') for m in ('host','search')]
    (delivery/'results.json').write_text(json.dumps(results))
    for name in ('paired_statistics.json','diagnostics.json','comparison.svg','README.zh-CN.md'):
        (delivery/name).write_text('fixture')
    files = {p.name:file_sha256(p) for p in delivery.iterdir()}
    (delivery/'accepted.json').write_text(json.dumps(dict(state='accepted',test_access=False,
        input_sha256=file_sha256(run/'accepted.json'),files=files)))
    checked = []
    data = collect(sequence, lambda *args: checked.append(args))
    assert len(checked) == 1 and len(data['rows']) == 4
    assert data['cases'][0]['state'] == 'accepted_delivery'
    output = tmp_path/'publication'
    publish(data, output)
    assert (output/'current/comparison.svg').is_file()
    assert len((output/'current/results.csv').read_text().splitlines()) == 5
    (delivery/'results.json').write_text('[]')
    with pytest.raises(ValueError, match='evidence changed'):
        collect(sequence, lambda *args: None)
