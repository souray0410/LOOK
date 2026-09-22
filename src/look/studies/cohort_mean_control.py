"""Fixed empty-prefix mean control: reuse slopes/lambda, evaluate both means."""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from look.studies.cohort_delivery import load_selected, validate, PATTERNS, ARMS
from look.data.array_pair import ArrayPair
from look.data.observed_pair import collate_observed
from look.methods.affine_family import FamilyArtifact
from look.runtime.state import atomic_write_json, file_sha256, stable_hash
from look.runtime.host_checkpoint import atomic_save, cpu_tree
from look.training.mechanism_training import state_equal
from look.evaluation.evaluator import evaluate_missing, save_prediction_bundle
from look.analysis.cohort_publication import PATTERNS as LABELS, SITES


def constrain_mean(artifact):
    result=deepcopy(artifact);q=result.base.components.double();m=result.mapping
    if not torch.allclose(q@q.T,torch.eye(len(q),dtype=q.dtype),atol=2e-5,rtol=2e-5):
        raise ValueError('Invalid PCA basis')
    m.output_mean=(m.output_mean@q.T)@q
    m.method=artifact.mapping.method+'_fixed_slope_projected_mean'
    m.diagnostics=dict(control='only_mean_projected_onto_same_complete_PCA_basis',
        refitting=False,rank=len(q),ridge_lambda=m.ridge_lambda)
    for field in ('input_mean','left','right'):
        if not torch.equal(getattr(m,field),getattr(artifact.mapping,field)):raise ValueError('Slope changed')
    return result


def publish(root,rows,total,state):
    out=root/'publication';out.mkdir(exist_ok=True)
    doc=dict(schema='look_cohort_fixed_mean_v1',state=state,test_used=False,seed=3416,
        complete=len(rows)==total,expected=total,results=rows,new_training=0,new_fits=0,
        scope='fixed_empty_prefix_same_slope_lambda_basis_and_host; forced_single_site_not_tree',
        limitations=['exploratory_same_dev','single_seed','no_multiple_comparison_inference','not_independent_reoptimized_constrained_method'])
    atomic_write_json(doc,out/'current.json')
    lines=['# 固定斜率后，自由均值是否有帮助？','',
        '同一青光眼小队列1264 train / 296 dev，种子3416，ResNet18普通宿主及MMTM适配宿主。',
        '逐项复用两种自由均值拟合的空前缀候选，固定宿主、位置、斜率、λ和PCA基，仅将残差均值投影回原PCA子空间。两缺失状态、全部9位置均展示，不按正负挑行。',
        '不重新训练或拟合；这是强制单点的机制对照，不是另一棵完整树，也不能替代约束均值方法独立搜索。原自由预测须现场重放一致。',
        f'状态：{state}；完成{len(rows)}/{total}项。','',
        '|宿主|拟合|缺失|位置|自由均值F1|约束均值F1|自由−约束(pp)|自由NLL|约束NLL|',
        '|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        a,b=r['free_metrics'],r['constrained_metrics']
        lines.append(f"|{r['host']}|{r['family']}|{LABELS[r['pattern']]}|{SITES[r['site']]}|{100*a['macro_f1']:.2f}|{100*b['macro_f1']:.2f}|{100*(a['macro_f1']-b['macro_f1']):+.2f}|{a['negative_log_likelihood']:.3f}|{b['negative_log_likelihood']:.3f}|")
    lines+=['','F1越大越好；NLL为负对数似然，越小越好。本表描述已参与设计的dev，未作多重比较推断，不能从某行最大差值宣称泛化优势。',
        'PCA：主成分子空间内的残差拟合；自由低秩残差：在完整降维特征空间拟合秩32残差。两者内部均比较相同斜率，两个家族之间斜率并不相同。',
        '[完整正收益树累计结果](../small_cohort/README.md)；[研究决策](../research_decisions/README.md)。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')


def run(plan):
    if plan['schema']!='look_cohort_fixed_mean_plan_v1' or plan['test_used'] is not False:raise ValueError('Unregistered plan')
    root=Path(plan['root']);root.mkdir(exist_ok=True,parents=True)
    with (root/'manager.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rows=[];total=36*len(plan['specs'])
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        for ref in plan['specs']:
            if file_sha256(ref['path'])!=ref['sha256']:raise ValueError('Spec changed')
            s=json.loads(Path(ref['path']).read_text());validate(s);original=Path(s['output'])
            torch.cuda.set_per_process_memory_fraction(s['gpu_budget_bytes']/torch.cuda.get_device_properties(0).total_memory)
            g,host=load_selected(s,original);frozen=cpu_tree(g.state_dict())
            dev=ArrayPair(s['data_root'],'development');assert len(dev)==296
            loader=DataLoader(dev,batch_size=s['training']['microbatch'],shuffle=False,num_workers=0,collate_fn=collate_observed)
            for family in ARMS:
                a=json.loads((original/family/'accepted.json').read_text())
                if a['state']!='accepted' or a['identity']!=stable_hash(s) or a['host_best_sha256']!=host['files']['best.pt']:raise ValueError('Unmatched accepted family')
                for pattern in PATTERNS:
                    base=original/family/'corrections'/pattern
                    selection=json.loads((base/'selection.json').read_text())
                    decisions=[d for d in selection['decisions'] if d['path']==[]]
                    if len(decisions)!=1 or len(decisions[0]['candidates'])!=9:raise ValueError('Incomplete root candidates')
                    for c in decisions[0]['candidates']:
                        import psutil
                        if psutil.virtual_memory().available<.15*psutil.virtual_memory().total:raise MemoryError('Resource reserve')
                        path=base/c['artifact'];e=c['evidence']
                        if file_sha256(path)!=c['sha256'] or file_sha256(e['prediction'])!=e['sha256']:raise ValueError('Candidate evidence changed')
                        key=stable_hash(dict(spec=ref['sha256'],artifact=c['sha256'],prediction=e['sha256'],control='project_mean_v1'))
                        folder=root/'cases'/key;folder.mkdir(parents=True,exist_ok=True)
                        if (folder/'accepted.json').exists():
                            r=json.loads((folder/'accepted.json').read_text())
                            if r['identity']!=key:raise ValueError('Control identity changed')
                            for name,digest in r['files'].items():
                                if file_sha256(folder/name)!=digest:raise ValueError('Cached control changed')
                            row=r['row']
                        else:
                            artifact=FamilyArtifact.from_record(torch.load(path,map_location='cpu',weights_only=False))
                            free=evaluate_missing(g,loader,'cuda:0',fixed_pattern=pattern,artifact_banks={pattern:[artifact]})
                            with np.load(e['prediction'],allow_pickle=False) as old:
                                for k in ('participant_ids','labels','logits'):
                                    if not np.array_equal(free[k],old[k]):raise ValueError('Free candidate replay changed: '+k)
                            constrained=constrain_mean(artifact);atomic_save(folder/'artifact.pt',constrained.record())
                            restored=FamilyArtifact.from_record(torch.load(folder/'artifact.pt',map_location='cpu',weights_only=False))
                            result=evaluate_missing(g,loader,'cuda:0',fixed_pattern=pattern,artifact_banks={pattern:[restored]})
                            for k in ('participant_ids','labels'):
                                if not np.array_equal(free[k],result[k]):raise ValueError('Participant mismatch')
                            save_prediction_bundle(result,folder/'predictions.npz')
                            row=dict(host='ResNet18＋MMTM' if 'mmtm' in s else 'ResNet18',run_id=s['run_id'],
                                family='PCA' if family=='pca_free_mean' else '自由低秩残差',pattern=pattern,site=c['node'],
                                source_artifact_sha256=c['sha256'],source_prediction_sha256=e['sha256'],
                                constrained_prediction_sha256=file_sha256(folder/'predictions.npz'),
                                lambda_fixed=artifact.mapping.ridge_lambda,free_metrics=free['metrics'],constrained_metrics=result['metrics'])
                            atomic_write_json(dict(identity=key,row=row,files={n:file_sha256(folder/n) for n in ('artifact.pt','predictions.npz')}),folder/'accepted.json')
                        rows.append(row);publish(root,rows,total,'running')
                        atomic_write_json(dict(state='running',completed=len(rows),total=total,time=time.time(),pid=os.getpid()),root/'status.json')
            state_equal(g,frozen);del g;torch.cuda.empty_cache()
        publish(root,rows,total,'accepted')
        atomic_write_json(dict(state='accepted',completed=len(rows),total=total,time=time.time(),test_used=False),root/'status.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args();plan=json.loads(Path(a.plan).read_text())
    uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader','-i',os.environ['CUDA_VISIBLE_DEVICES']],text=True).strip()
    with (Path(plan['lock_root'])/(uuid+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(plan)
        except Exception as e:
            atomic_write_json(dict(state='needs_review',error=repr(e),time=time.time()),Path(plan['root'])/'status.json');raise
