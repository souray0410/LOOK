from dataclasses import asdict
from types import SimpleNamespace
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from look.methods import family_statistics as fs, operator as op
from look.methods.linear_vector import ResidualMoments
from look.methods.linear_operator import fingerprint
from look.methods.affine_family import FamilyArtifact, fit_map


class Data(Dataset):
    split='train'; augment=False
    def __len__(self): return 12
    def __getitem__(self,i):
        return dict(oct=torch.tensor([i*.1,1.]),cfp=torch.tensor([.3,i*.2]),
                    label=i%2,participant_id=str(i))


class Graph(torch.nn.Module):
    def __init__(self):
        super().__init__();self.values={};self.calls=0
    def forward(self,levels):
        self.calls+=1
        if levels[0]==0:self.values['a']=self.values['oct']+2*self.values['cfp']
        if levels[0]==1:self.values['b']=self.values['a']*3+.2


@pytest.fixture
def setup(monkeypatch):
    graph=Graph().eval()
    monkeypatch.setattr(op,'_reset_inputs',lambda g,o,c,counts=None:g.values.update(oct=o,cfp=c))
    for module in (fs,op):
        monkeypatch.setattr(module,'site_level',lambda g,n:{'a':0,'b':1}[n])
        monkeypatch.setattr(module,'read_site',lambda g,n:g.values[n])
        monkeypatch.setattr(module,'write_site',lambda g,n,v:g.values.__setitem__(n,v))
    bases={n:op.FullFeaturePCA(n,1,(2,),(2,),torch.tensor([.2,.3]),torch.tensor([1.1,.9]),
           torch.tensor([.1,-.1]),torch.eye(2),torch.tensor([.6,.4]),12,0.,0,'frozen') for n in ('a','b')}
    return graph,DataLoader(Data(),batch_size=4),bases


def fitter(setup,path,**kw):
    graph,loader,bases=setup
    return fs.FamilyStatistics(graph,loader,bases,'cpu',path,{'source':'frozen','ordered_data':'locked'},10**7,2,**kw)


def reference(setup,upstream=(),sites=('a','b')):
    graph,loader,bases=setup;result={}
    for name in sites:
        b=bases[name];s=ResidualMoments.empty(2)
        for full,missing,_,_ in op.iter_feature_pairs(graph,loader,name,'oct_missing',1,'cpu',upstream):
            s.update((missing-b.mean)/b.std,(full-missing)/b.std)
        result[name]=s
    return result


def same(left,right):
    assert fingerprint({n:asdict(v) for n,v in left.items()})==fingerprint({n:asdict(v) for n,v in right.items()})


def test_full_dimensional_multi_site_matches_legacy_and_completed_cache_does_no_forward(setup,tmp_path):
    expected=reference(setup);f=fitter(setup,tmp_path)
    actual=f.statistics('oct_missing',[],['a','b']);same(actual,expected)
    assert f.metrics['full_forwards']==f.metrics['missing_forwards']==3
    assert not list(tmp_path.glob('**/reference*'))
    setup[0].calls=0
    same(f.statistics('oct_missing',[],['a','b']),expected)
    assert setup[0].calls==0
    assert f.metrics['completed_hits']==1


def test_partial_resume_skips_network_and_reproduces_moments(setup,tmp_path):
    tick=[0]
    def pause():tick[0]+=1;return tick[0]==3
    with pytest.raises(fs.SelectionPaused):
        fitter(setup,tmp_path,should_pause=pause).statistics('oct_missing',[],['a','b'])
    f=fitter(setup,tmp_path);actual=f.statistics('oct_missing',[],['a','b'])
    assert f.metrics['full_forwards']==f.metrics['missing_forwards']==1
    assert f.metrics['skipped_batches']==2
    same(actual,reference(setup))


def test_upstream_identity_changes_and_corrected_downstream_is_refitted(setup,tmp_path):
    graph,loader,bases=setup;b=bases['a']
    original=reference(setup)
    base=op.LOOKArtifact('a','oct_missing','normalized_mean',1,1,(2,),(2,),b.mean,b.std,b.pca_mean,
                        b.components[:1],torch.zeros(1,1),torch.zeros(1),.1,0.,0.)
    a=FamilyArtifact(base,fit_map(original['a'],1,.1,arm='residual_rrr',basis=b.components[:1]))
    f=fitter(setup,tmp_path)
    f.statistics('oct_missing',[],['b'])
    corrected=f.statistics('oct_missing',[a],['b'])
    same(corrected,reference(setup,[a],['b']))
    assert f.metrics['full_forwards']==6
    assert not torch.equal(corrected['b'].mean_x,original['b'].mean_x)


def test_paused_batch_order_changed_rejected_before_forward(setup,tmp_path):
    tick=[0]
    def pause():tick[0]+=1;return tick[0]==2
    with pytest.raises(fs.SelectionPaused):
        fitter(setup,tmp_path,should_pause=pause).statistics('oct_missing',[],['a','b'])
    g,loader,bases=setup
    class Reverse(Subset):split='train';augment=False
    changed=(g,DataLoader(Reverse(loader.dataset,list(reversed(range(12)))),batch_size=4),bases)
    g.calls=0
    with pytest.raises(ValueError,match='order changed'):
        fitter(changed,tmp_path).statistics('oct_missing',[],['a','b'])
    assert g.calls==0


def test_corruption_and_combined_memory_guard(setup,tmp_path):
    f=fitter(setup,tmp_path);f.statistics('oct_missing',[],['a','b'])
    p=next(tmp_path.glob('*.pt'));r=torch.load(p,weights_only=False)
    r['payload']['complete']=False;torch.save(r,p)
    with pytest.raises(ValueError,match='corrupt'):f.statistics('oct_missing',[],['a','b'])
    f.budget=1
    with pytest.raises(MemoryError,match='workspace'):f.statistics('oct_missing',[],['a','b'])


def test_projected_target_scatter_matches_direct_centered_residual(setup,tmp_path):
    f=fitter(setup,tmp_path,projected_ranks=(1,2));full=f.statistics('oct_missing',[],['a','b'])
    graph,loader,bases=setup
    for name,b in bases.items():
        values=[]
        for y,x,_,_ in op.iter_feature_pairs(graph,loader,name,'oct_missing',1,'cpu'):
            values.append(((y-x)/b.std).double())
        residual=torch.cat(values)
        for q in (1,2):
            projected=residual@b.components[:q].double().T
            expected=(projected-projected.mean(0)).square().sum()
            torch.testing.assert_close(f.projected[name][q].syy,expected,rtol=1e-13,atol=1e-13)
    restored=fitter(setup,tmp_path,projected_ranks=(1,2));same(restored.statistics('oct_missing',[],['a','b']),full)
    torch.testing.assert_close(restored.projected['a'][1].syy,f.projected['a'][1].syy,rtol=0,atol=0)


def test_checkpoint_save_preserves_payload_without_deepcopy(setup,tmp_path,monkeypatch):
    original=fs.atomic_save
    observed=[]
    def save(path,record):
        payload=record['payload']
        assert record['sha256']==fingerprint(payload)
        original(path,record)
        observed.append(torch.load(path,weights_only=False))
    monkeypatch.setattr(fs,'atomic_save',save)
    # Identity construction still uses dataclass snapshots; collection must not.
    f=fitter(setup,tmp_path,projected_ranks=(1,))
    def no_copy(*args,**kwargs):raise AssertionError('Dense checkpoint deep copy')
    monkeypatch.setattr(torch.Tensor,'__deepcopy__',no_copy)
    result=f.statistics('oct_missing',[],['a','b'])
    assert observed and observed[-1]['payload']['complete']
    for name,s in result.items():
        saved=observed[-1]['payload']['statistics'][name]
        for key,value in vars(s).items():
            if isinstance(value,torch.Tensor):assert torch.equal(value,saved[key])
            else:assert value==saved[key]


@pytest.mark.parametrize('x',[torch.tensor(2.),torch.empty(0),torch.arange(12).reshape(3,4).T,
                            torch.tensor([True,False]),torch.tensor([1+2j])])
def test_buffer_fingerprint_matches_previous_bytes(x):
    import hashlib
    from look.runtime.state import stable_hash
    y=x.detach().cpu().contiguous()
    expected=stable_hash(dict(shape=list(y.shape),dtype=str(y.dtype),
        sha256=hashlib.sha256(y.numpy().tobytes()).hexdigest()))
    assert fingerprint(x)==expected
