"""Finite progress, preregistered contrast families and overlap-aware bootstrap."""
import csv
from pathlib import Path
import json
import numpy as np
from look.runtime.state import atomic_write_json, stable_hash, file_sha256
from look.analysis.observed_report import f1_from_confusion


def comparison_registry():
    """Contrast types are fixed before outcomes; concrete view weights are bound later."""
    families={
        'original': ['look_minus_'+x for x in ('host','bias','affine','single_final','all_on','available_parent')],
        'regime': ['look_minus_host_original','look_minus_host_continue_complete',
            'look_minus_host_continue_missing','look_gain_missing_minus_original',
            'look_gain_missing_minus_complete_continuation'],
        'practical': ['look_minus_distill','look_minus_ce_student','look_minus_available_parent',
            'distill_minus_ce_student'],
        'mechanism': ['look_minus_'+x for x in ('independent','shuffle_mean','missing_readout',
            'available_readout','missing_refit','available_refit','mlp','affine_equivalence')],
        'sample_efficiency': ['subset_minus_full_fixed','subset_minus_full_reselect'],
    }
    return dict(schema='look_mechanism_comparisons_v1',families=families,
        architecture_aggregation=['all_equal','resnet50','densenet121','swin_b'],
        disease_aggregation=['all_equal','glaucoma','cataract','macular_degeneration'],
        directions=['oct_missing','cfp_missing'],positions='equal_mean_three',seeds='equal_mean_three',
        shuffle_repeats='equal_mean_three',subset_repeats='equal_mean_three',
        subset_fractions=[.01,.05,.1,.25,.5],iterations=10000,
        global_scope='all_primary_comparisons_with_shared_union_participant_resampling',
        missing_views='not_estimable_never_change_weights',reference_margin=.01)


def progress(feed,output):
    from look.studies.mechanism_case import verify_case
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    definitions=comparison_registry();path=root/'comparisons.json'
    if path.exists() and json.loads(path.read_text())!=definitions:raise ValueError('Comparison registry changed')
    atomic_write_json(definitions,path)
    rows=[];counts={};metrics=[]
    for t in feed['tasks']:
        run=Path(t['run_dir']);spec=json.loads(Path(t['spec']).read_text());task=spec['task']
        status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {'state':'pending'}
        state=status['state']
        if (run/'accepted.json').exists():
            from look.runtime.mechanism_receipts import monitored_acceptance
            receipt=monitored_acceptance(run,spec,verify_case,root/'verification_cache');state=receipt['state']
            if state=='accepted':
                for r in json.loads((run/'records.json').read_text()):
                    metrics.append(dict(task_id=t['id'],**task['host'],kind=task['kind'],arm=task['arm'],
                        method=r['method'],scenario=r['scenario'],**r['metrics']))
        counts[state]=counts.get(state,0)+1
        rows.append(dict(id=t['id'],kind=task['kind'],arm=task['arm'],state=state,run_id=run.name))
    result=dict(schema='look_mechanism_progress_v1',expected=feed['expected'],registered=len(rows),
        waiting_dependencies=feed['waiting_dependencies'],counts=counts,tasks=rows,
        complete=len(rows)==feed['expected'] and all(r['state'] in ('accepted','infeasible') for r in rows),
        scientific_completion='accepted_only; infeasible_is_a_technical_exclusion',test_access=False)
    atomic_write_json(result,root/'progress.json')
    if metrics:
        fields=sorted(set().union(*(r.keys() for r in metrics)))
        with (root/'development_results.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(metrics)
    (root/'README.zh-CN.md').write_text('# LOOK补充研究\n\n'
        +f"已登记 {len(rows)}/{feed['expected']}；等待依赖 {feed['waiting_dependencies']}；状态 {counts}。\n\n"
        +'378项神经网络训练与修正拟合、评价视图分别计数。未齐组不排名；技术不可行不计为性能失败。\n'
        +'macro-F1越大越好；差值为LOOK减对照。当前为dev研究，含选优影响。\n')
    return result


def aggregate(feed,base_feed,output,iterations=10000):
    """Bind fixed contrast types to actual accepted prediction files.

    Missing or infeasible rows retain their planned definition and are reported
    non-estimable. No renormalization over whichever seeds happen to finish.
    """
    from look.studies.mechanism_case import verify_case
    from look.studies.project_case import verify_case as verify_base
    from look.studies.mechanism_protocol import DISEASES,ARCHITECTURES,POSITIONS,SEEDS
    root=Path(output);root.mkdir(parents=True,exist_ok=True);views=[];by_key={};file_index={}
    def add(key,record):
        if file_sha256(record['path'])!=record['sha256']:raise ValueError('Prediction source changed')
        digest=record['sha256']
        if digest not in file_index:
            with np.load(record['path'],allow_pickle=False) as r:
                file_index[digest]=len(views)
                views.append(dict(participant_ids=r['participant_ids'].copy(),labels=r['labels'].copy(),
                                  predictions=r['logits'].argmax(1),sha256=digest))
        if key in by_key and by_key[key]!=file_index[digest]:raise ValueError('Conflicting evaluation identity')
        by_key[key]=file_index[digest]
    for t in base_feed['tasks']:
        run=Path(t['run_dir'])
        if not (run/'accepted.json').exists():continue
        s=json.loads(Path(t['spec']).read_text());verify_base(run,s)
        h=(s['disease'],s['model']['name'],s['position'],s['seed'])
        for r in json.loads((run/'report/source_records.json').read_text()):
            add((*h,'original',r['method'],r['scenario'],None,None),r)
    for t in feed['tasks']:
        run=Path(t['run_dir'])
        if not (run/'accepted.json').exists():continue
        s=json.loads(Path(t['spec']).read_text());receipt=verify_case(run,s)
        if receipt['state']!='accepted':continue
        task=s['task'];h=task['host'];positions=[h['position']] if 'position' in h else POSITIONS
        for r in json.loads((run/'records.json').read_text()):
            for position in positions:
                group=(h['disease'],h['architecture'],position,h['seed'])
                regime=task['arm'] if task['kind']=='host_training' else task['kind']
                repeat=task.get('repeat');fraction=task.get('fraction')
                add((*group,regime,r['method'],r['scenario'],fraction,repeat),r)
    # Paired inference requires identical people and labels within each disease.
    by_disease={}
    for key,view_id in by_key.items():
        v=views[view_id];identity=dict(zip(map(str,v['participant_ids']),map(int,v['labels'])))
        if key[0] in by_disease and by_disease[key[0]]!=identity:raise ValueError('Unmatched disease evaluation cohort')
        by_disease[key[0]]=identity
    definitions=[];weights=[];families=[];aliases={};missing=[]
    registry=comparison_registry()
    for disease in ('all_equal',*DISEASES):
        for architecture in ('all_equal',*ARCHITECTURES):
            arches=ARCHITECTURES if architecture=='all_equal' else [architecture]
            hosts=[(d,a,p,s) for d in (DISEASES if disease=='all_equal' else [disease]) for a in arches for p in POSITIONS for s in SEEDS]
            for pattern in ('oct_missing','cfp_missing'):
                def term(regime,method,coefficient=1.,fraction=None,repeats=(None,)):
                    return [( (*h,regime,method,pattern,fraction,r),coefficient/len(hosts)/len(repeats)) for h in hosts for r in repeats]
                def gain(regime,coef=1):return term(regime,'look',coef)+term(regime,'host',-coef)
                definitions_here=[]
                for ref in ('host','bias','affine','single_final','all_on','available_parent'):
                    definitions_here.append(('original','look_minus_'+ref,None,term('original','look')+term('original',ref,-1)))
                definitions_here.extend([
                    ('regime','look_minus_host_original',None,gain('original')),
                    ('regime','look_minus_host_continue_complete',None,gain('continue_complete')),
                    ('regime','look_minus_host_continue_missing',None,gain('continue_missing')),
                    ('regime','look_gain_missing_minus_original',None,gain('continue_missing')+gain('original',-1)),
                    ('regime','look_gain_missing_minus_complete_continuation',None,gain('continue_missing')+gain('continue_complete',-1)),
                    ('practical','look_minus_distill',None,term('original','look')+term('student_training','distill',-1)),
                    ('practical','look_minus_ce_student',None,term('original','look')+term('student_training','ce',-1)),
                    ('practical','look_minus_available_parent',None,term('original','look')+term('original','available_parent',-1)),
                    ('practical','distill_minus_ce_student',None,term('student_training','distill')+term('student_training','ce',-1))])
                for arm in ('independent','shuffle','missing_readout','available_readout','missing_refit','available_refit','mlp','affine_equivalence'):
                    definitions_here.append(('mechanism','look_minus_'+('shuffle_mean' if arm=='shuffle' else arm),None,
                        term('original','look')+term('correction',arm,-1,repeats=(0,1,2) if arm=='shuffle' else (None,))))
                for selection in ('fixed','reselect'):
                    for fraction in (.01,.05,.1,.25,.5):
                        definitions_here.append(('sample_efficiency','subset_minus_full_'+selection,fraction,
                            term('sample_curve',selection,fraction=fraction,repeats=(0,1,2))+
                            term('sample_curve',selection,-1,fraction=1.,repeats=(0,))))
                for family,name,fraction,terms in definitions_here:
                    definition=dict(disease=disease,architecture=architecture,pattern=pattern,family=family,name=name,fraction=fraction)
                    absent=[key for key,_ in terms if key not in by_key]
                    if absent:missing.append(dict(**definition,reason='required_view_missing_or_infeasible',missing=len(absent)));continue
                    w=np.zeros(len(views))
                    for key,c in terms:w[by_key[key]]+=c
                    # Same statistical comparison may answer several research questions.
                    signature=stable_hash([(int(i),round(float(w[i]),14)) for i in np.flatnonzero(abs(w)>1e-14)])
                    if signature in aliases:
                        definitions[aliases[signature]].setdefault('additional_questions',[]).append(definition);continue
                    aliases[signature]=len(definitions);definitions.append(definition);weights.append(w);families.append(family+'/'+disease)
    lock=dict(schema='look_bound_comparisons_v1',definitions=definitions,not_estimable=missing,
        view_sha256=[v['sha256'] for v in views],weights=[w.tolist() for w in weights],registry=registry)
    atomic_write_json(lock,root/'bound_comparisons.json')
    all_resolved=len(feed['tasks'])==feed['expected'] and all((Path(t['run_dir'])/'accepted.json').exists() for t in feed['tasks'])
    if not all_resolved:
        return dict(state='waiting_complete_comparison_scope',estimable=len(definitions),not_estimable=len(missing))
    signature=stable_hash(lock)
    if (root/'statistics.json').exists() and json.loads((root/'statistics.json').read_text()).get('signature')==signature:
        return dict(state='accepted_reused',signature=signature)
    family_memberships=[sorted(set([d['family']+'/'+d['disease']]+[q['family']+'/'+q['disease'] for q in d.get('additional_questions',[])])) for d in definitions]
    stats=overlap_bootstrap(views,weights,family_memberships,iterations)
    stats.update(signature=signature,definitions=definitions,not_estimable=missing,test_access=False)
    atomic_write_json(stats,root/'statistics.json')
    claims=[]
    for definition,stat in zip(definitions,stats['contrasts']):
        claims.append(dict(**definition,**stat,evidence='development_selected_models',
            limitation='no_independent_test_or_cross_hospital_claim'))
    atomic_write_json(claims,root/'claims.json')
    with (root/'claims.csv').open('w',newline='') as f:
        fields=['disease','architecture','pattern','family','name','fraction','difference','ordinary_95','global_simultaneous_95','classification','limitation']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(claims)
    return dict(state='accepted',signature=signature,comparisons=len(definitions))


def classify(interval, margin=.01):
    if interval is None:return 'undefined'
    lo,hi=interval
    if lo>margin:return 'substantial_improvement'
    if hi<-margin:return 'substantial_decline'
    if lo>=-margin and hi<=margin:return 'practically_close'
    return 'unresolved'


def overlap_bootstrap(views,weights,families,iterations=10000,seed=7341618):
    """Union-participant bootstrap retains overlapping people across diseases.

    Each view retains its own eligibility mask and label. A person's resampling
    multiplicity is shared across tasks/models; labels need not agree across
    diseases. Never resample the model/seed rows as independent participants.
    """
    ids=sorted(set().union(*(set(map(str,v['participant_ids'])) for v in views)))
    index={p:i for i,p in enumerate(ids)};w=np.asarray(weights,dtype=float)
    if w.ndim!=2 or w.shape[1]!=len(views) or len(families)!=len(w):raise ValueError('Contrast dimensions mismatch')
    cached=[]
    for v in views:
        own=list(map(str,v['participant_ids']));y=np.asarray(v['labels']);pred=np.asarray(v['predictions'])
        if len(set(own))!=len(own) or len(y)!=len(own) or len(pred)!=len(own):raise ValueError('Unpaired view')
        if not np.isin(y,[0,1]).all() or not np.isin(pred,[0,1]).all():raise ValueError('Binary classification required')
        code=2*y+pred;ix=np.array([index[p] for p in own]);cached.append([ix[code==c] for c in range(4)])
    point=w@np.array([float(f1_from_confusion(np.array([len(c) for c in groups]))) for groups in cached])
    rng=np.random.default_rng(seed);draws=[]
    for start in range(0,iterations,16):
        n=min(16,iterations-start)
        counts=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=n)
        f1=[]
        for groups in cached:
            confusion=np.column_stack([counts[:,g].sum(1) for g in groups])
            if np.any(confusion.sum(1)==0):raise ValueError('Non-estimable bootstrap view with zero sampled participants')
            f1.append(f1_from_confusion(confusion))
        draws.append(np.stack(f1,axis=1)@w.T)
    draws=np.concatenate(draws);sd=draws.std(0,ddof=1);valid=sd>0
    standardized=np.zeros_like(draws);standardized[:,valid]=(draws[:,valid]-draws[:,valid].mean(0))/sd[valid]
    critical=float(np.quantile(np.abs(standardized).max(1),.95))
    memberships=[f if isinstance(f,list) else [f] for f in families]
    family_q={f:float(np.quantile(np.abs(standardized[:,[f in group for group in memberships]]).max(1),.95)) for f in set(sum(memberships,[]))}
    result=[]
    for i,p in enumerate(point):
        interval=[float(p-critical*sd[i]),float(p+critical*sd[i])] if valid[i] else None
        q=family_q[memberships[i][0]]
        result.append(dict(difference=float(p),ordinary_95=np.quantile(draws[:,i],[.025,.975]).tolist(),
            family_simultaneous_95=[float(p-q*sd[i]),float(p+q*sd[i])] if valid[i] else None,
            all_family_simultaneous_95={f:[float(p-family_q[f]*sd[i]),float(p+family_q[f]*sd[i])] if valid[i] else None for f in memberships[i]},
            global_simultaneous_95=interval,zero_variance=not bool(valid[i]),classification=classify(interval)))
    return dict(iterations=iterations,union_participants=len(ids),seed=seed,contrasts=result,
        scope='conditional_on_locked_models; union_participant_bootstrap; no_selection_bias_removal')
