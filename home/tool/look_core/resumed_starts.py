"""Resume only the explicitly authorized remainder of the original start grid."""
from pathlib import Path
from collections import Counter
from .dual_queue import read, record, verify, atomic, digest, check_source, supervise, validate_receipt


def split_scope(cases, completed):
    ids = [c['case_id'] for c in cases]
    if len(ids) != 243 or len(set(ids)) != 243 or not set(completed).issubset(ids):
        raise ValueError('Full start scope must contain 243 unique original cases')
    groups = Counter(c['context'] for c in cases)
    if len(groups) != 27 or set(groups.values()) != {9}:
        raise ValueError('Expected 27 contexts with nine starts each')
    for context in groups:
        if sorted(c['start_ordinal'] for c in cases if c['context']==context) != list(range(1,10)):
            raise ValueError('Every context must contain starts 1 through 9')
    return [c for c in cases if c['case_id'] not in completed]


def verify_previous(case, row):
    from .start_study import audit_case, verify_file
    for rec in [row['result'],row['manifest'],*row['predictions']]: verify_file(rec)
    for bank in row['banks'].values():
        for rec in [bank['selected_manifest'],bank['factor_selection'],*bank['artifacts'],*bank['decision_files']]: verify_file(rec)
    return audit_case(case,Path(row['result']['path']))


def prepare(root, config_path, output, acceptance):
    from .start_study import verify_source, audit_case, choose_start, PATTERNS
    root, output = Path(root), Path(output)
    cfg=read(config_path)
    if cfg.get('test_access') is not False or cfg['expected_remaining']!=182:
        raise ValueError('Explicit sealed 182-case authorization required')
    historical=read(cfg['historical_summary']);drain=read(cfg['drain_complete'])
    cases=historical['identity']['cases'];by_id={c['case_id']:c for c in cases}
    frozen=dict(config=record(config_path),history=record(cfg['historical_summary']),
                drain=record(cfg['drain_complete']),inventory=record(cfg['source_inventory']),
                source_manifest_sha256=check_source(root))
    output.mkdir(parents=True,exist_ok=True)
    pin=output/'predecessor_identity.json'
    if pin.exists() and read(pin)!=frozen:raise ValueError('Predecessor or source changed')
    atomic(frozen,pin)
    inventory=read(cfg['source_inventory'])
    for rec in inventory.get('protected_sources',[]):
        from .start_study import verify_file
        verify_file(rec)
        current=root/Path(rec['path']).relative_to(Path(cfg['parent_source']))
        if record(current)['sha256']!=rec['sha256']:raise ValueError('Scientific kernel changed')
    completed={}
    for name,row in historical['completed_cases'].items():
        completed[name]=verify_previous(by_id[name],row)
    if drain['status']!='complete' or drain['test_access'] is not False:
        raise ValueError('Drain is incomplete')
    name=drain['case_id']
    if name in completed:raise ValueError('Drain would be counted twice')
    completed[name]=verify_previous(by_id[name],drain['record'])
    pending=split_scope(cases,completed)
    if len(completed)!=61 or len(pending)!=cfg['expected_remaining']:
        raise ValueError('Observed completion differs from 61 reused + 182 pending')
    sources={}
    for context,source in inventory['sources'].items():
        verify_source(source)
        p=output/'sources'/f'{context}.json'
        if p.exists() and read(p)!=source:raise ValueError('Frozen source changed')
        atomic(source,p);sources[context]=record(p)
    reused_selected={}
    selected_provenance=[]
    for context,rec in historical['selected_contexts'].items():
        from .start_study import verify_file
        rp=verify_file(rec);manifest=rp.with_name('frozen_selection.json');f=read(manifest)
        rows=[r for r in completed.values() if r['context']==context]
        if len(rows)!=9:raise ValueError('Historical selection lacks nine completed starts')
        for pattern in PATTERNS:
            best=choose_start([dict(start_ordinal=r['start_ordinal'],macro_f1=r['banks'][pattern]['macro_f1'],record=r) for r in rows])['record']
            if f['directions'][pattern]['result'] != best['result']:
                raise ValueError('Historical selected start differs from fixed rule')
        if read(rp).get('frozen_selection_sha256')!=record(manifest)['sha256']:
            raise ValueError('Selected-context freeze mismatch')
        reused_selected[context]=rec
        selected_provenance.extend(record(p) for p in rp.parent.rglob('*') if p.is_file() and p.suffix in ('.json','.npz'))
    # Interleave seeds at each fusion/filling/start. Ranking and fitting rules are unchanged.
    fusions=list(dict.fromkeys(c['fusion_position'] for c in cases))
    fillings=list(dict.fromkeys(c['filling'] for c in cases))
    pending.sort(key=lambda c:(fillings.index(c['filling']),fusions.index(c['fusion_position']),c['start_ordinal'],c['seed']))
    jobs=[dict(id=c['case_id'],kind='suffix',seed=c['seed'],case=c,source=sources[c['context']],after=[]) for c in pending]
    for context in sources:
        if context in reused_selected:continue
        group=sorted([c for c in cases if c['context']==context],key=lambda c:c['start_ordinal'])
        entries=[];after=[]
        for c in group:
            name=c['case_id']
            if name in completed:entries.append(dict(case=c,result=completed[name]['result']))
            else:entries.append(dict(case=c,job_id=name));after.append(name)
        jobs.append(dict(id='select__'+context,kind='selected',seed=group[0]['seed'],context=context,
                         source=sources[context],records=entries,after=after))
    plan=dict(protocol='independent_single_gpu_cases_v1',test_access=False,min_free_disk_gib=100,acceptance_record=record(acceptance),
              amendment='User explicitly resumed the previously deferred full start grid on 2026-09-07.',jobs=jobs)
    pp=output/'plan.json'
    if pp.exists() and read(pp)!=plan:raise ValueError('Execution plan changed')
    atomic(plan,pp)
    evidence=dict(completed_cases=completed,selected_contexts=reused_selected,selected_provenance=selected_provenance,
                  sources=sources,identity=historical['identity'],frozen=frozen)
    atomic(evidence,output/'reused_evidence.json')
    summary=dict(identity=dict(protocol='resumed_full_suffix_v1',parent=historical['identity'],plan_sha256=digest(plan)),
                 status='ready',test_access=False,output=str(output),expected_cases=243,expected_new_cases=182,
                 expected_reused_cases=61,completed_cases=completed,selected_contexts=reused_selected,
                 expected_new_selected_evaluations=27-len(reused_selected),new_case_execution=str(output/'queue/queue_status.json'))
    if not (output/'summary.json').exists():atomic(summary,output/'summary.json')
    print(f'prepared: reused={len(completed)} pending={len(pending)} reused_contexts={len(reused_selected)} new_selected_evaluations={27-len(reused_selected)}',flush=True)
    return pp


def collect(output):
    from .start_study import audit_case
    from .start_report import write_report
    output=Path(output);summary=read(output/'summary.json');plan=read(output/'plan.json')
    state=read(output/'queue/queue_status.json')
    for job in plan['jobs']:
        if job['id'] not in state['completed']:continue
        rp=output/'queue/cases'/job['id']/'queue_complete.json'
        receipt=validate_receipt(rp,state['identity'],job);path=verify(receipt['result'])
        if job['kind']=='suffix':summary['completed_cases'][job['id']]=audit_case(job['case'],path)
        else:summary['selected_contexts'][job['context']]=record(path)
    complete=len(summary['completed_cases'])==243 and len(summary['selected_contexts'])==27
    summary['status']='complete' if complete else state['status']
    summary['active_cases']=state['active'];atomic(summary,output/'summary.json')
    write_report(summary,output/'report')
    return summary


def run(root, config, output, acceptance, lock_path, *, prepare_only=False):
    pp=prepare(root,config,output,acceptance)
    if prepare_only:return
    try:
        supervise(root,pp,Path(output)/'queue',lock_path)
    finally:
        if (Path(output)/'queue/queue_status.json').exists():collect(output)
