from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import look.methods.operator as look


class Batches(list):
    dataset = SimpleNamespace(split='train', augment=False)
    drop_last = False


@pytest.fixture
def features(monkeypatch):
    nodes = {name: SimpleNamespace(feature_message=SimpleNamespace(current_state=None))
             for name in ['spatial', 'vector']}
    graph = SimpleNamespace(eval=lambda: None, get_node_by_name=nodes.__getitem__)
    data = torch.randn(19, 2, 8, 8, generator=torch.Generator().manual_seed(43))
    loader = Batches({'oct': x, 'cfp': x + 1} for x in data.split(4))
    calls = []
    def forward(graph, oct_tensor, cfp_tensor, artifacts=(), stop_node=None):
        assert not artifacts
        calls.append(stop_node)
        spatial = oct_tensor + cfp_tensor
        vector = spatial.mean((2, 3))
        nodes['spatial'].feature_message.current_state = spatial
        nodes['vector'].feature_message.current_state = vector
        return vector if stop_node in (None, 'vector') else spatial
    monkeypatch.setattr(look, 'forward_with_look', forward)
    return graph, loader, calls


def prepare(features, tmp_path, identity=None):
    graph, loader, _ = features
    return look.prepare_complete_pca_bank(graph, loader, ['spatial', 'vector'], [2, 4],
        2, torch.device('cpu'), tmp_path / 'pca', identity or {'checkpoint': 'a'}, tmp_path / 'quarantine')


def test_shared_pca_never_uses_missing_inputs_and_is_reused(features, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Missing forward/PCA refit is forbidden here')
    monkeypatch.setattr(look, 'iter_feature_pairs', forbidden)
    bank = prepare(features, tmp_path)
    assert set(bank) == {('spatial', 2), ('spatial', 4), ('vector', 1)}
    assert all(p.sample_count == 19 for p in bank.values())
    files = list((tmp_path / 'pca').glob('*/*.pt'))
    before = {p: (look.file_sha256(p), p.stat().st_mtime_ns) for p in files}
    monkeypatch.setattr(look, 'fit_complete_pca', forbidden)
    again = prepare(features, tmp_path)
    assert before == {p: (look.file_sha256(p), p.stat().st_mtime_ns) for p in files}
    assert [p.source_id for p in bank.values()] == [p.source_id for p in again.values()]


def test_shared_pca_repairs_only_missing_or_corrupt_entry(features, tmp_path, monkeypatch):
    bank = prepare(features, tmp_path)
    source = next(iter(bank.values())).source_id
    path = tmp_path / 'pca' / f'{source}.pt'
    path.write_bytes(b'truncated')
    calls = []
    original = look.fit_complete_pca
    def counted(*args, **kwargs):
        calls.append(args[2:4])
        return original(*args, **kwargs)
    monkeypatch.setattr(look, 'fit_complete_pca', counted)
    prepare(features, tmp_path)
    assert calls == [('spatial', 2)]
    assert list((tmp_path / 'quarantine').iterdir())
    calls.clear()
    path.unlink()
    prepare(features, tmp_path)
    assert calls == [('spatial', 2)]


def test_changed_checkpoint_gets_a_different_pca_bank(features, tmp_path):
    first = prepare(features, tmp_path)
    second = prepare(features, tmp_path, {'checkpoint': 'b'})
    assert first[('vector', 1)].source_id != second[('vector', 1)].source_id


def test_missing_directions_share_exact_pcs_without_refit(features, tmp_path, monkeypatch):
    pca = prepare(features, tmp_path)[('spatial', 2)]
    def pairs(graph, loader, node, pattern, factor, device, upstream, filler):
        for full, feature_shape, down_shape in look.iter_complete_features(graph, loader, node, factor, device):
            missing = full * (0.7 if pattern == 'oct_missing' else 0.5)
            yield full, missing, feature_shape, down_shape
    monkeypatch.setattr(look, 'iter_feature_pairs', pairs)
    monkeypatch.setattr(look, '_fit_incremental_pca', lambda *a, **k: pytest.fail('PCA refitted during Ridge'))
    graph, loader, _ = features
    a = look.fit_look_node(graph, loader, 'spatial', 'oct_missing', 2, [1, 2], 2, torch.device('cpu'), pca)
    b = look.fit_look_node(graph, loader, 'spatial', 'cfp_missing', 2, [1, 2], 2, torch.device('cpu'), pca)
    assert a[2].pca_source_id == b[2].pca_source_id == pca.source_id
    assert torch.equal(a[2].components, b[2].components)
    assert not torch.equal(a[2].weight, b[2].weight)


def test_shared_pca_rejects_validation_dataset(features, tmp_path):
    graph, loader, _ = features
    loader.dataset = SimpleNamespace(split='validation', augment=False)
    with pytest.raises(ValueError, match='training split'):
        prepare(features, tmp_path)
