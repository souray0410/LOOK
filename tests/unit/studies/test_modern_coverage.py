import json
import numpy as np
import pytest
from look.studies import modern_coverage as m


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(m,'SITES',['first','second'])
    values=dict(participant_ids=np.array(['a','b','c','d']),labels=np.array([0,1,0,1]),
                logits=np.array([[1.,0.],[0.,1.],[1.,0.],[0.,1.]]))
    pred=tmp_path/'dev.npz';np.savez(pred,**values)
    baselines={p:dict(path=str(pred),sha256=m.file_sha256(pred)) for p in m.PATTERNS}
    rows=[]
    for a in m.ARMS:
        for p in m.PATTERNS:
            for s,sites in [('single_site',['first']),('single_site',['second']),
                            ('best_forward',m.SITES),('positive_forward_tree',m.SITES)]:
                output=tmp_path/str(len(rows));output.mkdir()
                selection={'decisions':[{'enabled':True,'winner':{'node':'first'}}]} if s=='best_forward' else {'selected_path':['first']}
                (output/'selection.json').write_text(json.dumps(selection))
                rows.append(dict(arm=a,pattern=p,search=s,sites=sites,output=str(output),
                  selection_sha256=m.file_sha256(output/'selection.json'),
                  final=dict(prediction=str(pred),sha256=m.file_sha256(pred),metrics={'macro_f1':1.,'macro_auroc_ovr':1.})))
    return rows,baselines


def test_full_report_checks_coverage_pairing_and_preserves_search_limitations(tmp_path,monkeypatch):
    rows,base=fixture(tmp_path,monkeypatch)
    def contrast(labels,logits,definitions,iterations,seed):
        assert len(definitions)==32 and len(logits)==34 and iterations==10000
        return {'macro_f1':{'rows':definitions}}
    monkeypatch.setattr(m,'_simultaneous_contrasts',contrast)
    (tmp_path/'suite').mkdir();(tmp_path/'suite/coverage.json').write_text('{}')
    m.report(tmp_path,rows,base,{'cohort':{'development':4},'bootstrap':{'iterations':10000,'seed':3416}},{'fixed':'source'})
    result=json.loads((tmp_path/'results.json').read_text())
    assert len(result['comparisons'])==32 and result['scientific_acceptance'] is False
    assert 'selection bias' in result['intervals']['macro_f1']['limitation']
    assert all(r['selected_path']==['first'] for r in result['comparisons'])
    assert len((tmp_path/'results.csv').read_text().splitlines())==33
    m.verify_coverage(rows,m.SITES)
    with pytest.raises(ValueError,match='Missing or repeated'):m.verify_coverage(rows[:-1],m.SITES)
    with pytest.raises(ValueError,match='Missing or repeated'):m.verify_coverage(rows[:-1]+[rows[0]],m.SITES)
    (tmp_path/'0/selection.json').write_text('changed')
    with pytest.raises(ValueError,match='Selection'):m.verify_coverage(rows,m.SITES)


def test_prediction_order_is_not_silently_repaired(tmp_path,monkeypatch):
    _,base=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='participants'):
        m.prediction(base['oct_missing']['path'],np.array(['b','a','c','d']),np.array([0,1,0,1]))
