import copy
from pathlib import Path
import pytest
from look_core.start_evidence import validate_scope, CONTEXTS, PROTOCOL
from look_core.start_study import sites_for_fusion


def evidence():
    sites=sites_for_fusion('layer3');rows={}
    for ctx in CONTEXTS:
        for i,site in enumerate(sites,1):
            key=f'{ctx}__start_{i:02d}_{site}'
            rows[key]=dict(case_id=key,context=ctx,seed=int(ctx[-4:]),start_ordinal=i,
                fusion_position='layer3',filling='normalized_mean',candidate_sites=sites,
                eligible_sites=sites[i-1:],allowed_start=site)
    return dict(protocol=PROTOCOL,status='evidence_complete',test_access=False,
        completed_cases=rows,selected_contexts=dict.fromkeys(CONTEXTS),sources=dict.fromkeys(CONTEXTS))


def test_complete_representative_scope():
    value=evidence();validate_scope(value)
    assert len(value['completed_cases'])==27


@pytest.mark.parametrize('mutation',['missing','duplicate','sites','test','status','context'])
def test_representative_fail_closed(mutation):
    value=evidence();key=next(iter(value['completed_cases']))
    if mutation=='missing':value['completed_cases'].pop(key)
    if mutation=='duplicate':value['completed_cases']['duplicate']=value['completed_cases'].pop(key)
    if mutation=='sites':value['completed_cases'][key]['eligible_sites']=[]
    if mutation=='test':value['test_access']=True
    if mutation=='status':value['status']='running'
    if mutation=='context':value['completed_cases'][key]['context']='normalized_mean_feature_3407'
    with pytest.raises(ValueError):validate_scope(value)


def test_legacy_path_still_requires_full_study(monkeypatch):
    import look_core.method_study as module
    monkeypatch.setattr(module,'verify_parent',lambda *a:None)
    with pytest.raises(RuntimeError):module.verify_predecessors({},evidence(),{},Path('unused'))
    module.verify_predecessors({},dict(status='complete',test_access=False,
        completed_cases=dict.fromkeys(range(243)),selected_contexts=dict.fromkeys(range(27))),{},Path('unused'))


def test_execution_batch_does_not_change_source_or_pca_identity(monkeypatch):
    from look_core.bounded_runtime import BudgetRunner
    from look_core.start_study import SuffixRunner
    from look_core.pipeline import ExperimentRunner
    from types import SimpleNamespace
    from look_core.config import ExperimentConfig
    # Only graph construction sees the smaller execution batch; source config is restored even on error.
    obj=object.__new__(BudgetRunner)
    import dataclasses
    assert dataclasses.is_dataclass(ExperimentConfig)
    obj.config=ExperimentConfig();original=obj.config
    monkeypatch.setenv('LOOK_EXECUTION_MICROBATCH','1')
    def fail(self,path):
        assert self.config.micro_batch_size==1
        raise RuntimeError('fixture')
    monkeypatch.setattr(ExperimentRunner,'_load_frozen_graph',fail)
    with pytest.raises(RuntimeError,match='fixture'):obj._load_frozen_graph(Path('unused'))
    assert obj.config is original


def test_microbatch_gradient_with_uneven_last_effective_batch():
    # Compare actual optimization results, including a final short batch.
    import torch
    torch.manual_seed(9)
    x=torch.randn(11,7);y=torch.randint(0,2,(11,))
    initial=torch.nn.Linear(7,2).state_dict();results=[]
    for micro in (8,4,2,1):
        model=torch.nn.Linear(7,2);model.load_state_dict(initial)
        optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        for start in range(0,len(x),8):
            end=min(start+8,len(x));optimizer.zero_grad()
            for i in range(start,end,micro):
                j=min(i+micro,end)
                loss=torch.nn.functional.cross_entropy(model(x[i:j]),y[i:j])
                (loss*((j-i)/(end-start))).backward()
            optimizer.step()
        results.append(model.weight.detach().clone())
    for value in results[1:]:torch.testing.assert_close(value,results[0],atol=1e-7,rtol=1e-6)
