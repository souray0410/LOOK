"""Fixed-backbone three-start evidence and frozen accepted-prefix diagnostics.

No fitting, threshold adjustment, start selection, or test inference occurs here.
"""
from pathlib import Path
import json
import numpy as np
from look.runtime.state import atomic_write_json, atomic_write_text, stable_hash, utc_now
from look.studies.starts import read_json, file_record, verify_file
from look.evaluation.stability import logit_metrics

STARTS = (1, 4, 7)
SEEDS = (3407, 3408, 3409)
PATTERN = 'oct_missing'
METRICS = ('macro_f1', 'macro_auroc_ovr', 'macro_auprc_ovr', 'ece_15',
           'multiclass_brier', 'negative_log_likelihood')


def chosen_candidates(decisions):
    """Only accepted candidates in their actual sequential order."""
    result = []
    for d in decisions:
        if d['state'] != 'on':
            continue
        found = [c for c in d['candidates'] if c['latent_dim'] == d['chosen_dimension']]
        if len(found) != 1 or not d['selected_score'] > d['baseline_score']:
            raise ValueError('Invalid accepted decision')
        result.append(dict(node=d['node'], expected_f1=d['selected_score'],
                           baseline_f1=d['baseline_score'], **found[0]))
    return result


def summarize_logits(labels, logits):
    labels, logits = np.asarray(labels), np.asarray(logits, dtype=np.float64)
    if logits.shape != (len(labels), 2) or not np.isfinite(logits).all():
        raise ValueError('Expected finite binary logits')
    metrics = logit_metrics(labels, logits)
    margin = logits[:, 1] - logits[:, 0]
    signed = np.where(labels == 1, margin, -margin)
    nll = np.logaddexp(0, -signed)
    if not np.isclose(nll.mean(), metrics['negative_log_likelihood'], rtol=0, atol=1e-10):
        raise ValueError('Stable NLL calculation differs')
    return dict(metrics=metrics, max_abs_margin=float(abs(margin).max()),
                median_abs_margin=float(np.median(abs(margin))), max_nll=float(nll.max()),
                n=int(len(labels)), wrong=int((logits.argmax(1) != labels).sum()))


def read_prediction(path, expected_sha=None):
    rec = file_record(Path(path))
    if expected_sha and rec['sha256'] != expected_sha:
        raise ValueError('Prediction hash mismatch')
    with np.load(path, allow_pickle=False) as b:
        result = dict(labels=b['labels'].copy(), logits=b['logits'].copy())
        if 'participant_ids' in b:
            result['participant_ids'] = b['participant_ids'].astype(str)
    return result, rec


def load_scope(path):
    from look.analysis.start_evidence import load_evidence
    evidence = load_evidence(path, deep=False)
    records = [r for r in evidence['completed_cases'].values() if r['start_ordinal'] in STARTS]
    if len(records) != 9 or {(r['seed'], r['start_ordinal']) for r in records} != {
            (s, i) for s in SEEDS for i in STARTS}:
        raise ValueError('Exactly nine fixed-layer3 representative cases required')
    return evidence, sorted(records, key=lambda r: (r['seed'], r['start_ordinal']))


def cpu_evidence(manifest_path, output):
    from look.analysis.start_evidence import verify_records
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    evidence, records = load_scope(manifest_path)
    code = file_record(Path(__file__))
    identity = dict(protocol='compact_starts_frozen_prefix_v1', starts=list(STARTS), seeds=list(SEEDS),
                    fusion='layer3', filling='normalized_mean', test_access=False,
                    manifest=file_record(Path(manifest_path)), code=code)
    ip = out/'identity.json'
    if ip.exists() and read_json(ip) != identity:
        raise ValueError('Output identity differs; use a new directory')
    atomic_write_json(identity, ip)
    if (out/'cpu_complete.json').exists():
        old = read_json(out/'cpu_complete.json')
        for rec in old['provenance']: verify_file(rec)
        return evidence, records
    rows, provenance = [], []
    for r in records:
        verify_records(r)
        result = read_json(Path(r['result']['path']))
        provenance.append(r['result'])
        for pattern in ('oct_missing', 'cfp_missing'):
            scenario = 'look_after_fill_' + pattern
            p = Path(r['result']['path']).parent/'predictions'/f'validation__{scenario}.npz'
            pred, rec = read_prediction(p)
            stats = summarize_logits(pred['labels'], pred['logits'])
            for m in METRICS:
                if not np.isclose(stats['metrics'][m], result['validation'][scenario][m], rtol=0, atol=1e-12):
                    raise ValueError('Original metrics fail reproduction')
            provenance.append(rec)
            rows.append(dict(seed=r['seed'], start_ordinal=r['start_ordinal'], allowed_start=r['allowed_start'],
                             pattern=pattern, first_enabled=r['banks'][pattern]['first_enabled'],
                             enabled_count=r['banks'][pattern]['enabled_count'], **stats))
    original = next(r for r in records if r['seed'] == 3407 and r['start_ordinal'] == 1)
    base = Path(original['result']['path']).parent/'predictions'/f'validation__fill_{PATTERN}.npz'
    pred, rec = read_prediction(base); provenance.append(rec)
    prefix = [dict(prefix=0, node='all_off', prediction=rec, **summarize_logits(pred['labels'],pred['logits']))]
    labels = pred['labels']
    for k, chosen in enumerate(chosen_candidates(original['banks'][PATTERN]['decisions']), 1):
        pred, rec = read_prediction(chosen['prediction_path'], chosen['prediction_sha256'])
        if not np.array_equal(labels, pred['labels']): raise ValueError('Candidate label order differs')
        stats = summarize_logits(pred['labels'], pred['logits'])
        if stats['metrics']['macro_f1'] != chosen['expected_f1']:
            raise ValueError('Accepted candidate F1 differs')
        if prefix[-1]['metrics']['macro_f1'] != chosen['baseline_f1']:
            raise ValueError('Accepted prefix chain differs')
        prefix.append(dict(prefix=k, node=chosen['node'], prediction=rec, **stats)); provenance.append(rec)
    final_path = Path(original['result']['path']).parent/'predictions'/f'validation__look_after_fill_{PATTERN}.npz'
    final, rec = read_prediction(final_path); provenance.append(rec)
    if not np.array_equal(final['logits'], pred['logits']):
        raise ValueError('Last accepted prefix differs from original final logits')
    atomic_write_json(rows, out/'three_start_results.json')
    atomic_write_json(prefix, out/'saved_prefix_results.json')
    atomic_write_json(dict(status='complete',test_access=False,reused_cases=9,new_fits=0,
                          accepted_prefixes=len(prefix)-1,provenance=provenance),out/'cpu_complete.json')
    write_report(out)
    return evidence, records


def run_gpu(manifest_path, output):
    """Replay original accepted matrices; source inference batch/precision retained."""
    import torch
    from unittest.mock import patch
    from look.runtime.bounded import BudgetRunner, install_allocator_limits
    from look.studies.methods import make_resource_case
    from look.studies.starts import verify_source
    from look.methods.operator import load_selected_bank
    from look.methods.joint import read_site
    from look.methods import kernels as kernels
    from look.methods.imputation import NormalizedMeanFiller
    from look.runtime.provenance import seed_everything
    from look.training.distributed import module_state_sha256
    out=Path(output)
    evidence, records=cpu_evidence(manifest_path,out)
    if (out/'gpu_complete.json').exists():
        for rec in read_json(out/'gpu_complete.json')['provenance']:verify_file(rec)
        return
    install_allocator_limits(); seed_everything(3407)
    source=evidence['sources']['normalized_mean_layer3_3407'];verify_source(source)
    runner=BudgetRunner(source,make_resource_case(3407),out/'resources',out/'cache',torch.device('cuda:0'),(0,1))
    loaders,_=runner._build_loaders()
    graph,_=runner._load_frozen_graph(runner._train_or_resume());graph.eval()
    before=module_state_sha256(graph)
    original=next(r for r in records if r['seed']==3407 and r['start_ordinal']==1)
    bank=load_selected_bank(Path(original['banks'][PATTERN]['bank_root']))
    saved=read_json(out/'saved_prefix_results.json')
    if [a.node_name for a in bank] != [x['node'] for x in saved[1:]]:
        raise ValueError('Active matrices differ from accepted prefix order')
    filler=NormalizedMeanFiller();results=[];provenance=[]
    for k in range(len(bank)+1):
        result_path=out/f'prefix_{k:02d}.json'
        if result_path.exists():
            old=read_json(result_path)
            if old['identity_sha256']!=stable_hash(read_json(out/'identity.json')):raise ValueError('Cross-identity prefix resume')
            verify_file(old['prediction']);results.append(old);provenance.append(file_record(result_path));continue
        values={};feature_norms=[];ys=[];zs=[];ids=[]
        actual_apply=kernels.apply_artifact
        def traced(feature,artifact):
            corrected=actual_apply(feature,artifact)
            # Per feature row: these can be eye rows, not necessarily participants.
            a=feature.flatten(1).norm(dim=1);b=corrected.flatten(1).norm(dim=1)
            delta=(corrected-feature).flatten(1).norm(dim=1)
            item=values.setdefault(artifact.node_name,[])
            item.extend(torch.stack((a,b,delta),1).detach().cpu().double().numpy().tolist())
            return corrected
        with torch.no_grad(),patch.object(kernels,'apply_artifact',traced):
            for batch in loaders['validation']:
                o,c=filler.fill(batch['oct'].to('cuda:0'),batch['cfp'].to('cuda:0'),PATTERN)
                logits=kernels.forward_control(graph,o,c,bank[:k],policy='joint',pattern=PATTERN)
                feature_norms.extend(read_site(graph,'fusion_participant_feature').flatten(1).norm(dim=1).cpu().tolist())
                ys.append(batch['label'].numpy());zs.append(logits.detach().cpu().numpy());ids.extend(map(str,batch['participant_id']))
        labels,logits=np.concatenate(ys),np.concatenate(zs)
        expected,_=read_prediction(saved[k]['prediction']['path'],saved[k]['prediction']['sha256'])
        if not np.array_equal(labels,expected['labels']):raise ValueError('Replay label order changed')
        if 'participant_ids' in expected and not np.array_equal(np.array(ids),expected['participant_ids']):raise ValueError('Replay participant order changed')
        error=float(np.max(np.abs(logits.astype(np.float64)-expected['logits'])))
        if not np.allclose(logits,expected['logits'],rtol=0,atol=1e-4):
            raise ValueError(f'Frozen prefix {k} does not reproduce historical logits: {error}')
        pp=out/f'prefix_{k:02d}.npz'
        with pp.with_suffix('.partial').open('wb') as f:np.savez_compressed(f,labels=labels,logits=logits,participant_ids=np.array(ids))
        pp.with_suffix('.partial').replace(pp)
        norms={}
        for node,array in values.items():
            a=np.asarray(array)
            norms[node]=dict(unit='feature_rows',n=len(a),mean_before_l2=float(a[:,0].mean()),
                mean_after_l2=float(a[:,1].mean()),max_after_l2=float(a[:,1].max()),
                mean_delta_l2=float(a[:,2].mean()),max_delta_l2=float(a[:,2].max()))
        record=dict(prefix=k,node=saved[k]['node'],identity_sha256=stable_hash(read_json(out/'identity.json')),
            prediction=file_record(pp),max_logit_reproduction_error=error,feature_norms=norms,
            final_feature_mean_l2=float(np.mean(feature_norms)),final_feature_max_l2=float(np.max(feature_norms)),
            **summarize_logits(labels,logits))
        atomic_write_json(record,result_path);results.append(record);provenance.append(file_record(result_path))
        print(f"prefix={k} node={record['node']} F1={record['metrics']['macro_f1']:.6f} NLL={record['metrics']['negative_log_likelihood']:.6f} max_error={error}",flush=True)
    if module_state_sha256(graph)!=before:raise ValueError('Frozen backbone changed')
    for result in results:provenance.append(result['prediction'])
    atomic_write_json(dict(status='complete',test_access=False,new_fits=0,validation_n=results[0]['n'],
        prefixes=results,backbone_unchanged=True,source_inference_batch=runner.config.micro_batch_size,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        provenance=provenance),out/'gpu_complete.json')
    write_report(out)


def write_report(out):
    rows=read_json(out/'three_start_results.json');prefix=read_json(out/'saved_prefix_results.json')
    text='# LOOK 补充实验：固定融合位置的三起点与已接受路径诊断\n\n'
    text+='融合位置始终为 layer3；normalized_mean；seeds 3407/3408/3409。起点固定为① joint_input、④ joint_layer2、⑦ fusion_layer4。起点前OFF，之后逐层搜索，不强制起点ON。9个组合全部由已有结果核验复用；完整九起点仍保留。该子集是开发结果之后确定的结构性归纳，不是事前预设或新的最佳起点选择。\n\n'
    text+='| 校正起点 | OCT缺失 F1均值±SD | CFP缺失 F1均值±SD |\n|---|---:|---:|\n'
    for i in STARTS:
        rr=[r for r in rows if r['start_ordinal']==i];vs=[]
        for p in ('oct_missing','cfp_missing'):
            v=[r['metrics']['macro_f1'] for r in rr if r['pattern']==p]
            vs.append(f'{np.mean(v):.4f} ± {np.std(v,ddof=1):.4f}')
        text+=f"| {i}: {rr[0]['allowed_start']} | {vs[0]} | {vs[1]} |\n"
    text+='\n## 固定3407/OCT缺失的已接受路径\n\n下列前缀使用最终选定空间因子的实际ON节点；W/b不重拟合。每行包含本步以前接受的校正，不能解释为节点独立贡献。\n\n| 前缀 | 最后加入的节点 | Macro-F1 | NLL | 最大绝对logit差 |\n|---|---|---:|---:|---:|\n'
    for p in prefix:text+=f"| {p['prefix']} | {p['node']} | {p['metrics']['macro_f1']:.6f} | {p['metrics']['negative_log_likelihood']:.6f} | {p['max_abs_margin']:.3f} |\n"
    done=(out/'gpu_complete.json').exists()
    text+=f'\nGPU冻结前缀复现：{"完成" if done else "待执行"}。\n'
    text+='\n该诊断只能定位当前已接受路径何时放大预测，不能证明所有早期校正有害，也不能据此事后调整选择指标、阈值、alpha或PCA。输入开始的贪心路径不保证优于从中间开始的路径。测试集未访问；未启动延期起点、其他模型、填充或新训练。\n'
    atomic_write_text(text,out/'补充实验报告.md')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,pat in zip(axes,('oct_missing','cfp_missing')):
        for seed in SEEDS:
            rr=sorted([r for r in rows if r['seed']==seed and r['pattern']==pat],key=lambda r:r['start_ordinal'])
            ax.plot(STARTS,[r['metrics']['macro_f1'] for r in rr],'o-',label=f'Seed {seed}',alpha=.7)
        ax.set(xticks=STARTS,xticklabels=['1: joint_input','4: joint_layer2','7: fusion_layer4'],ylim=(0,1),ylabel='Validation Macro-F1',title=pat.replace('_',' '));ax.legend()
    fig.suptitle('Fixed layer3 fusion | 3 correction starts | existing results; no new selection')
    fig.savefig(out/'three_starts.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].plot([p['prefix'] for p in prefix],[p['metrics']['macro_f1'] for p in prefix],'o-');axes[0].set(ylim=(0,1),ylabel='Validation Macro-F1')
    axes[1].plot([p['prefix'] for p in prefix],[p['metrics']['negative_log_likelihood'] for p in prefix],'o-',color='#b54a45');axes[1].set(yscale='log',ylabel='NLL (log scale)')
    for ax in axes:ax.set(xlabel='Number of accepted corrections in frozen prefix',xticks=range(len(prefix)))
    fig.suptitle('Seed 3407 / OCT missing | conditional prefix effects, not isolated node effects')
    fig.savefig(out/'accepted_prefix.png',dpi=180);plt.close(fig)
