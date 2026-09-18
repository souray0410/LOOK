import json
import numpy as np
from look.studies.cohort_delivery import report, ARMS, PATTERNS
from look.runtime.state import file_sha256, stable_hash


def test_report_uses_canonical_metric_and_pairs_all_rows(tmp_path):
    spec={'test':'synthetic_report'}
    for arm in ARMS:
        folder=tmp_path/arm;folder.mkdir();records=[]
        for pattern in PATTERNS:
            for method in (arm,'host'):
                p=folder/f'{method}_{pattern}.npz'
                np.savez(p,labels=np.array([0,0,1,1]),participant_ids=np.array(['a','b','c','d']),
                    logits=np.array([[2.,0.],[1.,0.],[0.,1.],[0.,2.]]))
                records.append(dict(method=method,scenario=pattern,path=str(p),sha256=file_sha256(p),
                    metrics=dict(macro_f1=1.,macro_auroc_ovr=1.)))
        (folder/'accepted.json').write_text(json.dumps(dict(state='accepted',identity=stable_hash(spec),records=records,host_best_sha256='same_host')))
    report(spec,tmp_path)
    result=json.loads((tmp_path/'delivery/results.json').read_text())
    assert len(result['results'])==6
    assert len(result['comparisons'])==6
    assert result['statistics']['participants']==4
    assert '100.000' in (tmp_path/'delivery/README.zh-CN.md').read_text()
    assert json.loads((tmp_path/'delivery/accepted.json').read_text())['report_source_sha256']
