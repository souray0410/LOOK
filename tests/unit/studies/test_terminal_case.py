from types import SimpleNamespace
import pytest
import torch
from look.methods import operator as op
from look.studies.terminal_case import protocol, VERSION, cache, cached_rows, Paused


def test_cached_factories_preserve_original_numeric_stream(monkeypatch):
    g=torch.Generator().manual_seed(37)
    full=torch.randn(35,7,generator=g);missing=full*.5+torch.randn(35,7,generator=g)*.2
    def features(*args,**kw):
        for f in full.split(16):yield f,(7,),(7,)
    def pairs(*args,**kw):
        for f,m in zip(full.split(16),missing.split(16)):yield f,m,(7,),(7,)
    monkeypatch.setattr(op,'iter_complete_features',features);monkeypatch.setattr(op,'iter_feature_pairs',pairs)
    monkeypatch.setattr(op,'members',lambda n:(n,));monkeypatch.setattr(op,'member_shapes',lambda *a:((7,),))
    first=op.fit_complete_pca(None,None,'fusion_participant_feature',1,4,'cpu','identity')
    second=op.fit_complete_pca(None,None,'fusion_participant_feature',1,4,'cpu','identity',feature_factory=features)
    for name in ['mean','std','pca_mean','components','explained_variance_ratio']:
        assert torch.equal(getattr(first,name),getattr(second,name))
    args=(None,None,'fusion_participant_feature','oct_missing',1,[2,4],4,'cpu',first)
    original=op.fit_look_node(*args);cached=op.fit_look_node(*args,feature_pairs=pairs)
    for q in original:
        assert torch.equal(original[q].weight,cached[q].weight)
        assert original[q].ridge_lambda==cached[q].ridge_lambda


def test_cache_resume_identity_and_pair_order(tmp_path,monkeypatch):
    import look.studies.terminal_case as t
    class G:
        def get_edge_by_name(self,n):return SimpleNamespace(edge_operations=[SimpleNamespace(function=lambda x:x)])
    state={}
    def forward(graph,o,c,counts):state['x']=o+c;return state['x']
    monkeypatch.setattr(t,'forward_with_look',forward);monkeypatch.setattr(t,'read_site',lambda *a:state['x'])
    class Fill:
        def fill(self,o,c,p):return o,c
    monkeypatch.setattr(t,'NormalizedMeanFiller',Fill)
    batch=dict(participant_id=['a','b'],label=torch.tensor([0,1]),oct=torch.ones(2,2),cfp=torch.ones(2,2),counts=[1,1])
    ticks=[0]
    def pause():ticks[0]+=1;return ticks[0]==2
    with pytest.raises(Paused):cache(G(),[batch,batch],'cpu',tmp_path,'identity',pause)
    cache(G(),[batch,batch],'cpu',tmp_path,'identity',lambda:False)
    assert len(list(cached_rows(tmp_path)))==2
    with pytest.raises(ValueError,match='identity'):cache(G(),[batch,batch],'cpu',tmp_path,'other',lambda:False)
    (tmp_path/'000000.pt').write_bytes(b'bad')
    with pytest.raises(ValueError,match='changed'):list(cached_rows(tmp_path))


def test_scope_is_explicit():
    p=protocol();assert p['schema']==VERSION
    assert p['test_access'] is False and 'not_full_progressive' in p['scope']
    assert p['bootstrap_iterations']==10000
