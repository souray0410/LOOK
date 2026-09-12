"""Frozen-host matched predictions, simple controls and shared participant masks."""
from pathlib import Path
import numpy as np
from look.evaluation.evaluator import evaluate_missing, save_prediction_bundle
from look.evaluation.stability import logit_metrics, probabilities_from_logits
from look.data.missingness import missingness_plan
from look.runtime.state import atomic_write_json, file_sha256, stable_hash

PATTERNS=('complete','oct_missing','cfp_missing')


def check_matched(a,b):
    if not np.array_equal(a['participant_ids'].astype(str),b['participant_ids'].astype(str)) or not np.array_equal(a['labels'],b['labels']):
        raise ValueError('Predictions are not participant-paired')


def fit_logit_controls(full,missing):
    """Least-squares map to the same host's complete logits; train data only."""
    check_matched(full,missing)
    x=np.asarray(missing['logits'],dtype=np.float64);y=np.asarray(full['logits'],dtype=np.float64)
    design=np.column_stack([x,np.ones(len(x))])
    # Fixed pseudoinverse tolerance, no test/development tuning.
    weight=np.linalg.lstsq(design,y,rcond=1e-10)[0]
    return dict(schema='look_train_only_logit_controls_v1',affine=weight.tolist(),bias=(y-x).mean(0).tolist(),
                count=len(x),rcond=1e-10,train_identity=stable_hash(full['participant_ids'].astype(str).tolist()))


def transformed(result,logits):
    return dict(result,logits=logits,probabilities=probabilities_from_logits(logits),scores=logits[:,1]-logits[:,0],metrics=logit_metrics(result['labels'],logits))


def apply_control(result,control,method):
    x=result['logits']
    z=x+np.asarray(control['bias']) if method=='bias' else np.column_stack([x,np.ones(len(x))])@np.asarray(control['affine'])
    return transformed(result,z)


def assemble_mixed(complete,missing,ratio,seed):
    for r in missing.values():check_matched(complete,r)
    ids=complete['participant_ids'].astype(str);plan,metadata=missingness_plan(ids,ratio,seed)
    z=complete['logits'].copy();patterns=np.asarray([plan[x] for x in ids])
    for pattern,result in missing.items():
        mask=patterns==pattern;z[mask]=result['logits'][mask]
    return dict(transformed(complete,z),patterns=patterns,missingness=metadata)


def evaluate_suite(graph,loader,device,banks,controls,output,ratios,mask_seed):
    """Compute three deterministic inputs once; mix participants without rerunning."""
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    predictions={};records=[]
    def publish(method,scenario,result):
        path=out/(method+'__'+scenario+'.npz');save_prediction_bundle(result,path)
        records.append(dict(method=method,scenario=scenario,metrics=result['metrics'],path=str(path),sha256=file_sha256(path)))
    baseline={p:evaluate_missing(graph,loader,device,fixed_pattern=p) for p in PATTERNS}
    for p,result in baseline.items():publish('host',p,result)
    predictions['host']=baseline
    for method,by_pattern in banks.items():
        values={'complete':baseline['complete']}
        for p in PATTERNS[1:]:
            values[p]=evaluate_missing(graph,loader,device,fixed_pattern=p,artifact_banks=by_pattern)
            check_matched(baseline[p],values[p]);publish(method,p,values[p])
        publish(method,'complete',values['complete']);predictions[method]=values
    for method in ('bias','affine'):
        values={'complete':baseline['complete']}
        for p in PATTERNS[1:]:
            values[p]=apply_control(baseline[p],controls[p],method);publish(method,p,values[p])
        publish(method,'complete',values['complete']);predictions[method]=values
    for method,values in predictions.items():
        for ratio in ratios:
            result=assemble_mixed(values['complete'],{p:values[p] for p in PATTERNS[1:]},ratio,mask_seed)
            publish(method,f'mixed_{ratio:.1f}',result)
    atomic_write_json(dict(schema='look_matched_suite_v1',split=loader.dataset.split,
                          test_access=loader.dataset.split=='test',records=records),out/'suite.json')
    return records
