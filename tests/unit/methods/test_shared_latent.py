from dataclasses import asdict, replace
from types import SimpleNamespace
import pytest
import torch
from torch.utils.data import Dataset, DataLoader
from look.methods import shared_latent as sl, operator as op


class Data(Dataset):
    split='train'; augment=False
    def __len__(self): return 12
    def __getitem__(self,i):
        return dict(oct=torch.tensor([i*.1,1.]),cfp=torch.tensor([.3,i*.2]),
                    label=i%2,participant_id=str(i))


class Graph(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.values={};self.calls=0
    def forward(self,levels):
        self.calls+=1
        level=levels[0]
        if level==0:self.values['a']=self.values['oct']+2*self.values['cfp']
        if level==1:self.values['b']=self.values['a']*3+.2


@pytest.fixture
def setup(monkeypatch):
    g=Graph().eval()
    monkeypatch.setattr(op,'_reset_inputs',lambda g,o,c,counts=None:g.values.update(oct=o,cfp=c))
    monkeypatch.setattr(sl,'site_level',lambda g,n:{'a':0,'b':1}[n])
    monkeypatch.setattr(sl,'read_site',lambda g,n:g.values[n])
    monkeypatch.setattr(sl,'write_site',lambda g,n,v:g.values.__setitem__(n,v))
    # Oracle uses original forward_with_look and single-site extraction.
    monkeypatch.setattr(op,'site_level',sl.site_level)
    monkeypatch.setattr(op,'read_site',sl.read_site)
    monkeypatch.setattr(op,'write_site',sl.write_site)
    bases={n:op.FullFeaturePCA(n,1,(2,),(2,),torch.tensor([.2,.3]),torch.tensor([1.1,.9]),
           torch.tensor([.1,-.1]),torch.eye(2),torch.tensor([.6,.4]),12,0.,0,'frozen') for n in ('a','b')}
    return g,DataLoader(Data(),batch_size=4),bases


def reference(g,loader,bases,upstream=(),pattern='oct_missing'):
    result={}
    for n,b in bases.items():
        s=op.LatentSufficientStatistics(2)
        for full,missing,_,_ in op.iter_feature_pairs(g,loader,n,pattern,1,torch.device('cpu'),upstream):
            project=lambda x:(((x-b.mean)/b.std)-b.pca_mean)@b.components.T
            s.update(project(missing),project(full))
        result[n]=s
    return result


def same(a,b):
    for n in a:
        for k,v in vars(a[n]).items():
            if isinstance(v,torch.Tensor):assert torch.equal(v,vars(b[n])[k]),(n,k)
            else:assert v==vars(b[n])[k]


def fitter(setup,tmp_path,**kwargs):
    g,loader,bases=setup
    return sl.SharedLatentFitter(g,loader,bases,2,torch.device('cpu'),tmp_path/'moments',tmp_path/'refs',{'source':'fixed'},**kwargs)


def test_exact_statistics_solve_and_reference_reuse(setup,tmp_path):
    g,loader,bases=setup;expected=reference(*setup)
    f=fitter(setup,tmp_path);actual=f.statistics('oct_missing',[],['a','b']);same(actual,expected)
    assert (f.metrics['full_forwards'],f.metrics['missing_forwards'])==(3,3)
    for n,b in bases.items():
        old=op.fit_look_node(g,loader,n,'oct_missing',1,[1,2],2,torch.device('cpu'),b)
        new=op.fit_look_node(g,loader,n,'oct_missing',1,[1,2],2,torch.device('cpu'),b,latent_statistics=actual[n])
        for d in old:
            assert sl.fingerprint(asdict(old[d]))==sl.fingerprint(asdict(new[d]))
    g.calls=0;same(f.statistics('oct_missing',[],['a','b']),expected);assert g.calls==0
    same(f.statistics('cfp_missing',[],['a','b']),reference(*setup,pattern='cfp_missing'))
    assert f.metrics['full_forwards']==3 and f.metrics['reference_hits']==3


def test_upstream_state_and_resume(setup,tmp_path):
    g,loader,bases=setup;b=bases['a']
    a=op.LOOKArtifact('a','oct_missing','normalized_mean',1,2,(2,),(2,),b.mean,b.std,b.pca_mean,b.components,
                     torch.eye(2)*.2,torch.ones(2)*.1,.1,0.,0.)
    f=fitter(setup,tmp_path);actual=f.statistics('oct_missing',[a],['b'])
    same(actual,{'b':reference(*setup,upstream=[a])['b']})
    tick=[0]
    def pause():tick[0]+=1;return tick[0]==2
    interrupted=fitter(setup,tmp_path/'resume',should_pause=pause)
    with pytest.raises(sl.SelectionPaused):interrupted.statistics('oct_missing',[],['a','b'])
    resumed=fitter(setup,tmp_path/'resume');same(resumed.statistics('oct_missing',[],['a','b']),reference(*setup))


def test_corrupt_cache_rejected(setup,tmp_path):
    f=fitter(setup,tmp_path);f.statistics('oct_missing',[],['a','b'])
    p=next((tmp_path/'moments').glob('*.pt'));r=torch.load(p,weights_only=False)
    r['payload']['batches']=99;torch.save(r,p)
    with pytest.raises(ValueError,match='Corrupt'):f.statistics('oct_missing',[],['a','b'])


def test_unfrozen_graph_rejected(setup,tmp_path):
    setup[0].train()
    with pytest.raises(ValueError,match='Frozen'):fitter(setup,tmp_path).statistics('oct_missing',[],['a'])


def test_reference_order_and_source_are_checked(setup,tmp_path):
    f=fitter(setup,tmp_path); f.statistics('oct_missing',[],['a','b'])
    g,loader,bases=setup
    from torch.utils.data import Subset
    class Reversed(Subset):
        split='train';augment=False
    altered=(g,DataLoader(Reversed(loader.dataset,list(reversed(range(12)))),batch_size=4),bases)
    with pytest.raises(ValueError,match='order changed'):
        fitter(altered,tmp_path).statistics('cfp_missing',[],['a','b'])
    other=sl.SharedLatentFitter(g,loader,bases,2,torch.device('cpu'),tmp_path/'other',tmp_path/'refs',{'source':'different'})
    other.statistics('oct_missing',[],['a','b'])
    assert other.metrics['full_forwards']==3


def test_paused_prefix_cannot_change_order(setup,tmp_path):
    tick=[0]
    def pause():tick[0]+=1;return tick[0]==3
    with pytest.raises(sl.SelectionPaused):
        fitter(setup,tmp_path,should_pause=pause).statistics('oct_missing',[],['a','b'])
    g,loader,bases=setup
    from torch.utils.data import Subset
    class Reversed(Subset):
        split='train';augment=False
    altered=(g,DataLoader(Reversed(loader.dataset,list(reversed(range(12)))),batch_size=4),bases)
    with pytest.raises(ValueError,match='order changed'):
        fitter(altered,tmp_path).statistics('oct_missing',[],['a','b'])


def test_statistics_writer_exclusion(setup,tmp_path):
    f=fitter(setup,tmp_path)
    state=sl.fingerprint(dict(reference=f.identity,pattern='oct_missing',upstream=[],sites=['a','b']))
    with (tmp_path/'moments'/(state+'.lock')).open('a') as lock:
        sl.fcntl.flock(lock,sl.fcntl.LOCK_EX|sl.fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError): f.statistics('oct_missing',[],['a','b'])


def test_complete_best_forward_path_exact(setup,tmp_path):
    from look.methods.independent_greedy import fit_trajectory
    g,loader,bases=setup
    def execute(shared,root):
        fitter_=fitter(setup,root/'cache'); calls=[]
        def fit(n,upstream,where):
            calls.append((n,tuple(a.node_name for a in upstream)))
            stats=fitter_.statistics('oct_missing',upstream,['a','b']) if shared else None
            artifacts=op.fit_look_node(g,loader,n,'oct_missing',1,[1,2],2,torch.device('cpu'),bases[n],
                upstream_artifacts=upstream,latent_statistics=stats[n] if shared else None)
            return [(str(k),a) for k,a in artifacts.items()]
        def evaluate(bank):
            # Deterministic complete graph values, same fixed observations for every candidate.
            score=0.
            for batch in loader:
                x=op.forward_with_look(g,batch['oct'],torch.zeros_like(batch['cfp']),artifacts=bank,stop_node='b')
                target=(batch['oct']+2*batch['cfp'])*3+.2
                score-=float((x-target).square().sum())
            return dict(role='development',score=score)
        bank,r=fit_trajectory(identity='fixed',sites=['a','b'],mode='best_forward',output=root/'path',
            fit_candidates=fit,evaluate=evaluate,save_artifact=lambda a,p:a.save(p),load_artifact=op.LOOKArtifact.load)
        return bank,r,calls
    old,oldr,oldc=execute(False,tmp_path/'old');new,newr,newc=execute(True,tmp_path/'new')
    assert oldc==newc
    assert [sl.fingerprint(asdict(x)) for x in old]==[sl.fingerprint(asdict(x)) for x in new]
    assert oldr['final']==newr['final']
    assert [(d['enabled'],d['winner']['index'],d['winner']['key']) for d in oldr['decisions']]==[(d['enabled'],d['winner']['index'],d['winner']['key']) for d in newr['decisions']]
