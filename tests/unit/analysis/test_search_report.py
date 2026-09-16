import json
import numpy as np
from look.runtime.state import file_sha256
from look.studies.search_protocol import representative_starts,sites
from look.analysis.search_report import report


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def test_complete_matched_report_and_corrupt_replay(tmp_path,monkeypatch):
    import look.studies.search_case
    monkeypatch.setattr(look.studies.search_case,'verify_case',lambda *a:None)
    h=dict(disease='cataract',architecture='resnet50',position='deep',seed=3416)
    starts=representative_starts('resnet50','deep');runs=[]
    y=np.array([0,0,1,1]*8);logits=np.column_stack([1-y,y])*2.
    for n,(mode,start) in enumerate([('best_forward',1)]+[('greedy',i) for i in starts]):
        root=tmp_path/str(n);runs.append(str(root));ordered=sites('resnet50','deep')
        s=dict(host=h,mode=mode,source={'same':1},pca={'same':1},eligible_sites=ordered[start-1:])
        if start!=1:s['start_ordinal']=start
        write(root/'spec.json',s);write(root/'accepted.json',{'state':'accepted'})
        records=[]
        for pattern in ('cfp_missing','oct_missing'):
            write(root/'corrections'/pattern/'factor_selection.json',{'factor':1})
            for method in ('host','search'):
                path=root/(method+'_'+pattern+'.npz')
                np.savez(path,labels=y,participant_ids=np.arange(len(y)),logits=logits)
                records.append(dict(method=method,scenario=pattern,path=str(path),sha256=file_sha256(path),metrics={'macro_f1':1.0}))
        write(root/'development/suite.json',dict(records=records));write(root/'costs.json',dict(known_total_seconds=1))
    manifest=dict(host=h,runs=runs,output=str(tmp_path/'report'),test_access=False)
    r=report(manifest)
    assert r['complete_routes'] and (tmp_path/'report/comparison.svg').exists()
    stats=json.loads((tmp_path/'report/paired_statistics.json').read_text())
    assert len(stats['contrasts'])==10
    assert all(c['difference']==0 and c['zero_variance'] for c in stats['contrasts'])
    assert report(manifest)==r
