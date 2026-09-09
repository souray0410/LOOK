"""Bounded, validation-only diagnostics; no model or test configuration changes."""
from pathlib import Path
import hashlib
import csv
import numpy as np
from look.runtime.scheduler import read, record, verify, atomic, digest
from look.evaluation.stability import logit_metrics

PROTOCOL = 'look_stability_calibration_cost_v1'
SEEDS = (3407, 3408, 3409)
PATTERNS = ('oct_missing', 'cfp_missing')
POLICIES = ('joint', 'self_input_missing_only', 'ssf', 'logit_affine')


def validate_scope(plan):
    if (plan['protocol'] != PROTOCOL or plan['test_access'] is not False
            or plan['refit_correction'] is not False or plan['change_test_roster'] is not False):
        raise ValueError('Diagnostics cannot refit corrections or change test')
    jobs = plan['models']
    if len(jobs) != 12 or {(j['seed'], j['policy']) for j in jobs} != {
            (s, p) for s in SEEDS for p in POLICIES}:
        raise ValueError('Exactly three seeds and four fixed policies are required')
    if any(j['fusion'] != 'layer3' or j['filling'] != 'normalized_mean' for j in jobs):
        raise ValueError('Unexpected diagnostic setting')
    for seed in SEEDS:
        bases = [j['base'] for j in jobs if j['seed'] == seed]
        if any(b != bases[0] for b in bases):
            raise ValueError('Methods must share an identical frozen backbone and data')
    if plan['gpu_jobs'] != [dict(id=f'seed_{s}', seed=s) for s in SEEDS]:
        raise ValueError('Unexpected supplemental GPU jobs')


def freeze(request_path, output, source_hash):
    from look.evaluation.held_out import result_job, verify_jobs
    from look.studies.starts import source_result
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    req = read(request_path); destination = out/'diagnostic_manifest.json'
    if destination.exists():
        plan = read(destination); validate_scope(plan)
        if plan['request'] != record(request_path) or plan['source_manifest_sha256'] != source_hash:
            raise ValueError('Diagnostic identity changed')
        verify_jobs(plan['models'])
        return plan
    # Declaration must precede test access. Subsequent execution never reads test predictions.
    if read(req['test_queue']).get('test_access') is not False:
        raise ValueError('Pre-test diagnostic declaration is too late')
    parent_path = verify(req['parent']); parent = read(parent_path)
    if parent['status'] != 'complete': raise ValueError('Incomplete parent')
    models = []
    for seed in SEEDS:
        stage = f'normalized_mean_layer3_{seed}'
        original = result_job(source_result(parent, parent_path.parents[2], stage), stage, 'original')
        models.append(original)
        for field, policies in [('methods', ('ssf', 'logit_affine')), ('self_input', ('self_input_missing_only',))]:
            summary = read(verify(req[field]))
            if summary['status'] != 'complete' or summary['test_access'] is not False:
                raise ValueError('Incomplete validation control')
            for policy in policies:
                matches = []
                for name, rec in summary['completed_cases'].items():
                    result = read(verify(rec))
                    if result['case']['seed'] == seed and result['case']['method'] == policy:
                        matches.append(result_job(rec['path'], name, 'method', original['base']))
                if len(matches) != 1: raise ValueError('Missing/duplicate matched control')
                models.extend(matches)
    artifacts = verify_jobs(models)
    original_costs={}
    for seed in SEEDS:
        p=Path(req['methods']['path']).parent/'reference_analyses'/f'original_LOOK_{seed}'/'complete.json'
        if p.exists():
            original_costs[str(seed)]=record(p);artifacts.append(record(p))
    plan = dict(protocol=PROTOCOL, test_access=False, refit_correction=False, change_test_roster=False,
        request=record(request_path), source_manifest_sha256=source_hash, models=models, artifacts=artifacts,original_costs=original_costs,
        gpu_jobs=[dict(id=f'seed_{s}', seed=s) for s in SEEDS],
        scope=dict(fusion='layer3', filling='normalized_mean', seeds=list(SEEDS), patterns=list(PATTERNS),
                   splits=['train', 'validation'], train_participants=1264, validation_participants=296),
        calibration=dict(folds=5, fold_seed=3407, log_temperature_bounds=[-6., 12.],
            interpretation='validation development diagnostic; base models were selected using this validation set; not independent validation',
            test_application=False),
        stability=dict(policies=['joint', 'self_input_missing_only'], prefixes='all accepted prefixes including all OFF',
            complete_target='unmodified complete-input features of the same frozen backbone',
            interpretation='conditional sequential effects, not independent causal node contributions'),
        cost=dict(repeats=20, warmup=3, scope='warm inference including filling and correction; excludes data loading',
            historical_timing='preserve original scope and hardware; unavailable fields are null, never zero'),
        scheduling=dict(after='existing test queue completes', test_predictions_read=False,
                        gpu_budget_mib=14336, microbatches=[8,4,2,1]))
    validate_scope(plan); atomic(plan, destination); return plan


def fold_assignment(ids, labels, k=5):
    ids = np.asarray(ids).astype(str); labels = np.asarray(labels)
    if len(set(ids)) != len(ids) or len(ids) != len(labels): raise ValueError('Repeated participant')
    folds = np.empty(len(ids), dtype=int)
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        if len(indices) < k: raise ValueError('Insufficient class support for folds')
        ordered = sorted(indices, key=lambda i: hashlib.sha256(('3407:'+ids[i]).encode()).hexdigest())
        for n, i in enumerate(ordered): folds[i] = n % k
    if set(np.unique(labels)) != {0, 1}: raise ValueError('Binary labels required')
    return folds


def fit_temperature(labels, logits):
    from scipy.optimize import minimize_scalar
    labels, logits = np.asarray(labels), np.asarray(logits, dtype=np.float64)
    if logits.shape != (len(labels), 2) or not np.isfinite(logits).all(): raise ValueError('Invalid logits')
    signed = (2*labels-1)*(logits[:,1]-logits[:,0])
    def objective(t): return float(np.logaddexp(0, -signed*np.exp(-t)).mean())
    fit = minimize_scalar(objective, bounds=(-6.,12.), method='bounded', options={'xatol':1e-10})
    if not fit.success: raise ValueError('Temperature optimization failed')
    # Include identity and endpoints; no unbounded retries or silent search expansion.
    candidates = [-6., 0., 12., float(fit.x)]
    chosen = min(candidates, key=lambda t:(objective(t), abs(t)))
    return dict(temperature=float(np.exp(chosen)), log_temperature=chosen,
                at_boundary=chosen in (-6.,12.), fit_nll=objective(chosen))


def calibration(ids, labels, logits):
    logits = np.asarray(logits, dtype=np.float64); labels = np.asarray(labels)
    folds = fold_assignment(ids, labels); scaled = np.empty_like(logits); records=[]
    for f in range(5):
        train = folds != f; held = ~train
        fit = fit_temperature(labels[train], logits[train])
        scaled[held] = logits[held]/fit['temperature']
        records.append(dict(fold=f, fit_n=int(train.sum()), heldout_n=int(held.sum()), **fit))
    if not np.array_equal(logits.argmax(1),scaled.argmax(1)): raise ValueError('Temperature changed decisions')
    return dict(raw=logit_metrics(labels,logits), crossfit=logit_metrics(labels,scaled), folds=records,
                full_validation_fit=fit_temperature(labels,logits),
                note='Out-of-fold only for temperature, not for upstream model selection. Pooled ranking may change across fold-specific temperatures.'), folds, scaled


def write_csv(rows, path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    keys=sorted({k for row in rows for k in row})
    tmp=path.with_suffix('.partial')
    with tmp.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    tmp.replace(path)


def save_npz(path, **arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.partial')
    with tmp.open('wb') as f: np.savez_compressed(f,**arrays)
    tmp.replace(path)


def cpu_analysis(output):
    from look.evaluation.held_out import verify_jobs
    import pandas as pd
    out=Path(output);plan=read(out/'diagnostic_manifest.json');validate_scope(plan);verify_jobs(plan['models'])
    identity=record(out/'diagnostic_manifest.json');complete=out/'cpu_complete.json'
    if complete.exists():
        old=read(complete)
        if old['identity']!=identity: raise ValueError('Cross-identity CPU resume')
        for rec in old['outputs']: verify(rec)
        return old
    reference=plan['models'][0]['base']['labels'];frame=pd.read_csv(verify(reference),dtype={'participant_id':str})
    valid=frame.loc[frame['split']=='validation']; expected=set(valid.participant_id)
    if len(expected)!=296: raise ValueError('Full validation required')
    rows=[]; outputs=[]; sources=[]; costs=[]; reference_ids=None; reference_labels=None
    for model in plan['models']:
        result=read(verify(model['validation_result']))
        measured_costs=result.get('costs')
        if model['policy']=='joint' and str(model['seed']) in plan['original_costs']:
            cost_record=plan['original_costs'][str(model['seed'])]
            measured_costs=read(verify(cost_record))['costs'];sources.append(cost_record)
        # Existing timings are preserved verbatim, not conflated with a controlled benchmark.
        costs.append(dict(seed=model['seed'],policy=model['policy'],source=model['validation_result'],
                          costs=measured_costs,decisions_timing={p:{k:v for k,v in result.get('decisions',{}).get(p,{}).items()
                            if k.endswith('seconds')} for p in PATTERNS}))
        for pattern in PATTERNS:
            names=[('joint','look_after_fill_'+pattern),('filling','fill_'+pattern)] if model['policy']=='joint' else [(model['policy'],pattern)]
            for policy,scenario in names:
                rec=model['validation_predictions'][scenario]; path=verify(rec);sources.append(rec)
                with np.load(path,allow_pickle=False) as b:
                    ids=b['participant_ids'].astype(str);order=np.argsort(ids)
                    ids,labels,logits=ids[order],b['labels'][order],b['logits'][order]
                if set(ids)!=expected or len(ids)!=296:raise ValueError('Prediction is not full validation')
                if reference_ids is None:reference_ids,reference_labels=ids,labels
                if not np.array_equal(ids,reference_ids) or not np.array_equal(labels,reference_labels):
                    raise ValueError('Participant or label disagreement between methods')
                value,folds,scaled=calibration(ids,labels,logits)
                for metric, score in value['raw'].items():
                    if isinstance(score,(int,float)) and metric in result['validation'][scenario]:
                        if not np.isclose(score,result['validation'][scenario][metric],atol=1e-10,rtol=0):
                            raise ValueError('Saved metric does not reproduce')
                key=f'{policy}_{model["seed"]}_{pattern}'
                npz=out/'calibration'/f'{key}.npz';save_npz(npz,participant_ids=ids,labels=labels,
                    raw_logits=logits,crossfit_logits=scaled,folds=folds)
                jp=npz.with_suffix('.json');atomic(dict(seed=model['seed'],policy=policy,pattern=pattern,
                    source=rec,test_access=False,**value),jp);outputs.extend([record(npz),record(jp)])
                row=dict(seed=model['seed'],policy=policy,pattern=pattern)
                for kind in ('raw','crossfit'):
                    for m in ('macro_f1','macro_auroc_ovr','negative_log_likelihood','multiclass_brier','ece_15'):
                        row[kind+'_'+m]=value[kind][m]
                rows.append(row)
    write_csv(rows,out/'calibration.csv');atomic(costs,out/'historical_costs.json')
    outputs.extend([record(out/'calibration.csv'),record(out/'historical_costs.json')])
    value=dict(status='complete',identity=identity,test_access=False,new_backbone_fits=0,
               correction_refits=0,prediction_cases=len(rows),outputs=outputs,sources=sources)
    atomic(value,complete);report(out);return value


def report(output):
    out=Path(output);plan=read(out/'diagnostic_manifest.json');validate_scope(plan)
    lines=['# LOOK 稳定性、校准与成本补充报告','',
        '固定 layer3、normalized_mean、三个模型种子和两个缺失方向。此报告仅含 train／validation 开发证据。',
        '范围在 test 开启前冻结；不读取 test 预测，不修改其方法名单或任何已有 W/b。','',
        '## 输出校准诊断','',
        '五折以参与者划分，所有方法和种子共享划分。温度只在另外四折拟合；模型此前使用过整个 validation 进行选择，故此结果不是独立验证。',
        '不同折温度不同，合并后的 AUROC 可变；不据此声称温度缩放改善了原模型排序。','',
        '| 方法 | 缺失方向 | 原始 NLL，均值 ± SD | 温度交叉拟合 NLL，均值 ± SD |',
        '|---|---|---:|---:|']
    if (out/'calibration.csv').exists():
        with (out/'calibration.csv').open() as f:rows=list(csv.DictReader(f))
        for policy in ('filling',*POLICIES):
            for pattern in PATTERNS:
                r=[x for x in rows if x['policy']==policy and x['pattern']==pattern]
                if len(r)!=3:continue
                def fmt(key):
                    a=[float(x[key]) for x in r];return f'{np.mean(a):.4f} ± {np.std(a,ddof=1):.4f}'
                lines.append(f'| {policy} | {pattern} | {fmt("raw_negative_log_likelihood")} | {fmt("crossfit_negative_log_likelihood")} |')
        plot_calibration(rows,out/'calibration.png')
        lines.extend(['','![校准开发诊断](calibration.png)',
                      '每个点为一个模型种子；连线配对同一种子，短横线为三种子均值。NLL 使用对数坐标。'])
    else:lines.append('| 待完成 | — | — | — |')
    completed=[]
    for job in plan['gpu_jobs']:
        p=out/'gpu'/job['id']/'complete.json'
        if p.exists():
            v=read(p)
            if v.get('identity',{}).get('manifest')!=record(out/'diagnostic_manifest.json'):raise ValueError('Report identity mismatch')
            for rec in v['outputs']:verify(rec)
            completed.append(v)
    lines.extend(['','## 逐层稳定性与统一成本实测','',f'完整种子任务：{len(completed)}/3。未完成项不填零、不进入三种子汇总。',
        '稳定性比较每个已接受前缀，完整输入特征为参照；前后差异是顺序条件下的变化，不是节点独立因果贡献。',
        '历史成本见 historical_costs.json；缺失项保留 null。共享 PCA 时间不能跨方向重复相加，也不能据历史异构执行时间宣称加速比。'])
    gpurows=[];timings=[]
    for v in completed:
        for rec in v['outputs']:
            p=Path(rec['path'])
            if p.name.endswith('_stability.json'):gpurows.extend(read(p)['rows'])
            if p.name=='benchmark.json':timings.extend(read(p)['rows'])
    if gpurows:write_csv(gpurows,out/'stability.csv')
    if timings:write_csv(timings,out/'benchmark.csv')
    if len(completed)==3:lines.append('全部三种子补充回放完成；数值和逐步记录见 stability.csv、benchmark.csv 及各 GPU 任务的原始文件。')
    target=out/'REPORT.md';temp=target.with_suffix('.partial');temp.write_text('\n'.join(lines)+'\n');temp.replace(target)


def plot_calibration(rows, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=('filling','joint','self_input_missing_only','ssf','logit_affine')
    labels=('Filling','Original LOOK','Self-input LOOK','SSF','Logit affine')
    colors=('#C66B40','#247B91')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    for ax,pattern,title in zip(axes,PATTERNS,('OCT missing','CFP missing')):
        for i,policy in enumerate(names):
            matched=sorted([r for r in rows if r['policy']==policy and r['pattern']==pattern],key=lambda r:int(r['seed']))
            if len(matched)!=3:continue
            raw=np.array([float(r['raw_negative_log_likelihood']) for r in matched])
            calibrated=np.array([float(r['crossfit_negative_log_likelihood']) for r in matched])
            for j in range(3):
                x=i+(j-1)*.05
                ax.plot([x-.14,x+.14],[raw[j],calibrated[j]],color='#BDC6CC',linewidth=.8,zorder=1)
            for delta,values,color,label in zip((-.14,.14),(raw,calibrated),colors,('Raw','Temperature cross-fit')):
                ax.scatter(i+delta+np.array([-.05,0,.05]),values,s=28,color=color,zorder=3,label=label if i==0 else None)
                ax.plot([i+delta-.07,i+delta+.07],[values.mean()]*2,color=color,linewidth=2.5)
        ax.set_yscale('log');ax.set_xticks(range(5),labels,rotation=22,ha='right');ax.set_title(title)
        ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Negative log-likelihood (log scale; lower is better)')
    axes[1].legend(frameon=False,fontsize=9)
    fig.suptitle('Calibration diagnostic | layer3 + normalized_mean | 3 seeds',fontsize=13)
    fig.text(.5,.015,'Development evidence: upstream model selection used the full validation set. No test calibration.',ha='center',fontsize=9,color='#55616B')
    fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(destination,dpi=220,bbox_inches='tight');plt.close(fig)
