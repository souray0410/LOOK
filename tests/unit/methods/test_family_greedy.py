"""Adapter contracts, not a substitute for admitted full-MHD runtime acceptance."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from look.methods import family_greedy as op
from look.methods.affine_family import ARMS
from look.methods.operator import FullFeaturePCA


def setup(monkeypatch, arm, *, basis_rank=3):
    graph = torch.nn.Linear(3, 3).double().eval()
    for p in graph.parameters():
        p.requires_grad_(False)
    g = torch.Generator().manual_seed(721)
    x = torch.randn(12, 3, generator=g, dtype=torch.float64)
    full = 1.2*x + .4
    calls = []

    def pairs(graph, loader, site, pattern, factor, device, upstream, **kw):
        calls.append((site, tuple(a.node_name for a in upstream)))
        missing = x.clone()
        for a in upstream:
            missing = a.apply_feature(missing)
        for xx, yy in zip(missing.split(4), full.split(4)):
            yield yy, xx, (3,), (3,)

    def evaluate(graph, loader, device, *, artifact_banks, **kw):
        bank = next(iter(artifact_banks.values()))
        z = x.clone()
        for a in bank:
            z = a.apply_feature(z)
        # Deliberate selection fixture, not a reported classification metric.
        return dict(participant_ids=np.arange(12), labels=np.zeros(12, dtype=int),
                    logits=z.numpy(), metrics=dict(macro_f1=.5 + .01*len(bank)))

    monkeypatch.setattr(op, 'iter_feature_pairs', pairs)
    monkeypatch.setattr(op, 'evaluate_missing', evaluate)
    monkeypatch.setattr(op, 'save_prediction_bundle',
                        lambda r,p: (p.parent.mkdir(parents=True, exist_ok=True),
                                     np.savez(p, **{k:r[k] for k in ('participant_ids','labels','logits')})))
    bases = {(n,1): FullFeaturePCA(n,1,(3,),(3,),torch.zeros(3),torch.ones(3),
                torch.zeros(3),torch.eye(3)[:basis_rank],torch.ones(basis_rank)/3,
                12,0.,0,'train-only-fixture') for n in ('a','b')}
    rank = 2 if arm in op.RANKED else None
    penalty = None if arm == 'orthogonal_alignment' else .1
    return dict(graph=graph,
        train_loader=SimpleNamespace(dataset=SimpleNamespace(split='train',augment=False),drop_last=False),
        dev_loader=SimpleNamespace(dataset=SimpleNamespace(split='development')),
        arm=arm, pattern='oct_missing', sites=['a','b'], factor=1,
        candidates=[dict(rank=rank,ridge_lambda=penalty)], pca_bank=bases,
        identity='locked-fixture', device='cpu', workspace_bytes=10**7), calls


@pytest.mark.parametrize('arm', ARMS)
def test_independent_family_refits_with_accepted_upstream_and_replays(arm, monkeypatch, tmp_path):
    kw, calls = setup(monkeypatch, arm)
    frozen = {k:v.clone() for k,v in kw['graph'].state_dict().items()}
    bank, result = op.fit_family_trajectory(output=tmp_path, **kw)
    assert result['mode'] == 'best_forward'
    assert calls == [('a',()),('b',()),('b',('a',))]
    assert [a.node_name for a in bank] == ['a','b']
    assert all(a.mapping.diagnostics['reference_configuration_inherited'] is False for a in bank)
    again, replay = op.fit_family_trajectory(output=tmp_path, **kw)
    assert calls == [('a',()),('b',()),('b',('a',))]
    assert result == replay
    assert [a.node_name for a in again] == ['a','b']
    assert all(torch.equal(frozen[k],v) for k,v in kw['graph'].state_dict().items())


def test_resume_sufficient_statistics_exact_and_changed_prediction_rejected(monkeypatch, tmp_path):
    kw, _ = setup(monkeypatch, 'residual_rrr')
    ticks = [0]
    def pause():
        ticks[0] += 1
        return ticks[0] == 5
    with pytest.raises(op.SelectionPaused):
        op.fit_family_trajectory(output=tmp_path/'resumed', should_pause=pause, **kw)
    resumed, _ = op.fit_family_trajectory(output=tmp_path/'resumed', **kw)
    fresh, _ = op.fit_family_trajectory(output=tmp_path/'fresh', **kw)
    for a,b in zip(resumed, fresh):
        for name in ('left','right','input_mean','output_mean'):
            torch.testing.assert_close(getattr(a.mapping,name),getattr(b.mapping,name),rtol=0,atol=0)
    p = next((tmp_path/'resumed'/'predictions').glob('*.npz'))
    p.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Prediction evidence'):
        op.fit_family_trajectory(output=tmp_path/'resumed', **kw)


@pytest.mark.parametrize('arm', ['residual_rrr','pls_svd_ridge'])
def test_free_subspace_rank_does_not_inherit_pca_rank(monkeypatch, tmp_path, arm):
    kw,_ = setup(monkeypatch, arm, basis_rank=1)
    bank,_ = op.fit_family_trajectory(output=tmp_path, **kw)
    assert bank[0].mapping.rank_budget == 2


def test_data_roles_and_resource_guard_precede_fitting(monkeypatch, tmp_path):
    kw,calls = setup(monkeypatch, 'residual_rrr')
    kw['dev_loader'].dataset.split = 'test'
    with pytest.raises(ValueError, match='Development'):
        op.fit_family_trajectory(output=tmp_path/'test', **kw)
    kw['dev_loader'].dataset.split = 'development'
    kw['workspace_bytes'] = 1
    with pytest.raises(MemoryError, match='workspace'):
        op.fit_family_trajectory(output=tmp_path/'ram', **kw)
    assert calls == []


def test_candidates_explicit_and_constrained_basis_not_clipped(monkeypatch,tmp_path):
    for arm,candidates in [('residual_rrr',[]),('residual_rrr',[dict(rank=True,ridge_lambda=.1)]),
            ('residual_ridge',[dict(rank=1,ridge_lambda=.1)]),
            ('orthogonal_alignment',[dict(rank=None,ridge_lambda=.1)])]:
        with pytest.raises(ValueError):op.validate_candidates(arm,candidates)
    kw,_ = setup(monkeypatch,'shared_pca_ridge',basis_rank=1)
    with pytest.raises(ValueError,match='infeasible'):
        op.fit_family_trajectory(output=tmp_path,**kw)
