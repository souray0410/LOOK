import importlib.util
from pathlib import Path
import copy
import numpy as np
import pytest
import torch
from look_core.supplement_diagnostics import (PROTOCOL,SEEDS,POLICIES,validate_scope,fold_assignment,
    fit_temperature,calibration,freeze)
from look_core.supplement_runtime import loader_for,tensor_rows
from look_core.dual_queue import atomic,record


def plan():
    return dict(protocol=PROTOCOL,test_access=False,refit_correction=False,change_test_roster=False,
        models=[dict(seed=s,policy=p,fusion='layer3',filling='normalized_mean',base=dict(checkpoint=s)) for s in SEEDS for p in POLICIES],
        gpu_jobs=[dict(id=f'seed_{s}',seed=s) for s in SEEDS])


def test_scope_is_fixed_and_paired():
    p=plan();validate_scope(p)
    for key in ('test_access','refit_correction','change_test_roster'):
        q=copy.deepcopy(p);q[key]=True
        with pytest.raises(ValueError):validate_scope(q)
    q=copy.deepcopy(p);q['models'][1]['base']={'checkpoint':'different'}
    with pytest.raises(ValueError,match='identical'):validate_scope(q)
    q=copy.deepcopy(p);q['models'].append(q['models'][0])
    with pytest.raises(ValueError):validate_scope(q)


def test_folds_are_participant_stable_and_order_independent():
    ids=np.asarray([str(i) for i in range(100)]);y=np.arange(100)%2
    f=fold_assignment(ids,y);r=np.random.default_rng(2).permutation(100)
    assert np.array_equal(f[r],fold_assignment(ids[r],y[r]))
    for fold in range(5):assert sum(f==fold)==20
    with pytest.raises(ValueError):fold_assignment(['a']*100,y)


def test_extreme_logits_calibrate_stably_without_changing_decisions():
    y=np.arange(100)%2;correct=y.copy();correct[::4]=1-correct[::4]
    z=np.column_stack((1-correct,correct))*10000.
    value,folds,scaled=calibration(np.arange(100).astype(str),y,z)
    assert np.isfinite(scaled).all()
    assert np.array_equal(z.argmax(1),scaled.argmax(1))
    assert value['crossfit']['negative_log_likelihood'] < value['raw']['negative_log_likelihood']
    assert value['crossfit']['macro_f1']==value['raw']['macro_f1']


def test_heldout_labels_cannot_change_its_temperature(monkeypatch):
    import look_core.supplement_diagnostics as m
    ids=np.arange(100).astype(str);y=np.arange(100)%2;z=np.column_stack((1-y,y))*2.
    fixed=fold_assignment(ids,y);monkeypatch.setattr(m,'fold_assignment',lambda *args:fixed)
    a,_,_=calibration(ids,y,z);changed=y.copy();changed[fixed==0]=1-changed[fixed==0]
    b,_,_=calibration(ids,changed,z)
    assert a['folds'][0]['temperature']==b['folds'][0]['temperature']


def test_temperature_zero_logits_and_nonfinite_guard():
    value=fit_temperature(np.array([0,1]),np.zeros((2,2)))
    assert value['temperature']==1
    with pytest.raises(ValueError):fit_temperature([0,1],[[np.nan,0],[0,1]])


def test_loader_rejects_test_before_opening_any_file(monkeypatch):
    import look_core.supplement_runtime as m
    monkeypatch.setattr(m,'UKBBilateralVisitDataset',lambda **kw:pytest.fail('Dataset opened'))
    with pytest.raises(ValueError,match='prohibit test'):loader_for({},'test',2)


def test_diagnostic_declaration_must_precede_test(tmp_path):
    q=tmp_path/'queue.json';atomic(dict(test_access=True),q)
    r=tmp_path/'request.json';atomic(dict(test_queue=str(q)),r)
    with pytest.raises(ValueError,match='too late'):freeze(r,tmp_path/'out','source')
    assert not (tmp_path/'out/diagnostic_manifest.json').exists()


def test_trace_pair_metrics_are_exact_and_read_only():
    from types import SimpleNamespace
    before=torch.tensor([[1.,2.],[3.,4.]]);after=before+1;target=after.clone()
    artifact=SimpleNamespace(factor=1,downsample_shape=(2,),mean=torch.zeros(2),std=torch.ones(2),pca_mean=torch.zeros(2),components=torch.eye(2))
    original=before.clone();r=tensor_rows(before,after,target,artifact)
    assert np.allclose(r['mse_before'],1) and np.allclose(r['mse_after'],0)
    assert np.allclose(r['latent_mse_before'],1) and torch.equal(before,original)
    with pytest.raises(ValueError):tensor_rows(before,after[:1],target,artifact)


def test_predecessor_cannot_be_bypassed(tmp_path):
    path=Path(__file__).parents[1]/'operations/supplement_queue.py'
    spec=importlib.util.spec_from_file_location('diagnostic_queue_test',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    q=tmp_path/'q.json';req=dict(test_queue=str(q),test_queue_identity={'fixed':1})
    atomic(dict(identity={'fixed':1},status='running_test'),q);assert m.predecessor_status(req)=='waiting_existing_queue'
    atomic(dict(identity={'fixed':1},status='complete',active={},completed={}),q)
    with pytest.raises(ValueError,match='Incomplete'):m.predecessor_status(req)
    atomic(dict(identity={'fixed':1},status='complete',active={},completed={str(i):{} for i in range(87)}),q)
    assert m.predecessor_status(req)=='ready'
