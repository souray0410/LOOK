"""Freeze a test-only test roster after every declared validation dependency.

This module plans and verifies artifacts on CPU. It never opens a test loader.
The historical validation scheduler and scientific releases remain immutable.
"""
from pathlib import Path
from collections import Counter
import math
from look.runtime.scheduler import read, record, verify, atomic, digest

PROTOCOL = 'look_test_follow_on_v1'
PATTERNS = ('oct_missing', 'cfp_missing')
FILLINGS = ('normalized_mean', 'raw_zero', 'paired_cgan')
SEEDS = (3407, 3408, 3409)


def sealed_result(value):
    if (value.get('status') != 'complete' or value.get('phase', 'validation') != 'validation'
            or value.get('test_access', False) is not False or value.get('test')):
        raise ValueError('A complete validation-only result is required')


def validate_request(req):
    if (req.get('protocol') != PROTOCOL or req.get('cohort') != 'test'
            or req.get('natural_test') != 'withdrawn' or req.get('test_participants') != 290
            or req.get('expected_jobs') != 87 or req.get('missingness_seed') != 3407
            or req.get('random_ratios') != [.2, .4, .6, .8, 1.]
            or req.get('allow_refit') is not False or req.get('select_on_test') is not False):
        raise ValueError('Unexpected test-only evaluation scope')
    if req.get('bootstrap_iterations') != 2000 or req.get('replay_participants') != 8:
        raise ValueError('Unexpected reporting or replay protocol')


def dependency_status(req):
    validate_request(req)
    for k in ('parent', 'methods', 'self_input', 'reused', 'validation_plan'):
        verify(req[k])
    q = read(req['validation_queue'])
    if q['identity'] != req['validation_identity'] or q.get('test_access') is not False:
        raise ValueError('Validation queue identity or sealed boundary changed')
    jobs = read(req['validation_plan']['path'])['jobs']
    expected = {j['id'] for j in jobs}
    if not set(q.get('completed', {})).issubset(expected):
        raise ValueError('Unknown validation completion')
    if q['status'] == 'complete':
        if set(q['completed']) != expected or q.get('active'):
            raise ValueError('Premature validation completion')
        return 'ready'
    return 'waiting_validation' if q['status'] in ('starting','running','waiting_resources','paused') else 'blocked_dependency'


def bank_record(root):
    root = Path(root)
    m = record(root/'selected_manifest.json')
    artifacts = []
    for item in read(m['path'])['artifacts']:
        p = root/item['path']
        if p.parent != root/'selected' or p.is_symlink():
            raise ValueError('Unsafe correction bank path')
        r = record(p)
        if r['sha256'] != item['sha256']:
            raise ValueError('Selected correction artifact changed')
        artifacts.append(r)
    return dict(root=str(root), manifest=m, artifacts=artifacts)


def result_job(path, job_id, family, base=None, banks=None):
    """Bind inference to existing validation weights; no source fitting entry point."""
    path = Path(path); result = read(path); sealed_result(result)
    if base is None:
        manifest = record(path.with_name('validation_manifest.json'))
        m = read(manifest['path']); config = m['config']; selection = result['selection']
        checkpoint = record(result['checkpoint']['path'])
        if checkpoint['sha256'] != result['checkpoint']['sha256']:
            raise ValueError('Backbone checksum mismatch')
        generators = {}
        if selection['filling_strategy'] == 'paired_cgan':
            for direction, d in result['filling']['directions'].items():
                generators[direction] = record(d['checkpoint'])
                if generators[direction]['sha256'] != d['checkpoint_sha256']:
                    raise ValueError('Generator checksum mismatch')
        base = dict(config=config, selection=selection, checkpoint=checkpoint,
                    labels=record(config['labels_csv']), generators=generators,
                    validation_manifest=manifest)
    else:
        base = dict(base)
        if result.get('checkpoint') and result['checkpoint']['sha256'] != base['checkpoint']['sha256']:
            raise ValueError('Method checkpoint differs from its parent')
    policy = result.get('case', {}).get('method', 'joint')
    parameters = {}
    if family == 'single_modality':
        banks = {}
    elif policy == 'logit_affine':
        banks = {}; parameters = {p:result['decisions'][p]['selected'] for p in PATTERNS}
    elif policy == 'ssf':
        banks = {}
        for p in PATTERNS:
            d = result['decisions'][p]['selected']; r = record(d['checkpoint'])
            if r['sha256'] != d['checkpoint_sha256']: raise ValueError('SSF checksum mismatch')
            parameters[p] = r
    elif banks is None:
        banks = {p:bank_record(path.parent/('look' if policy == 'joint' else '')/p) for p in PATTERNS}
    if policy not in ('joint','logit_affine','ssf','independent_fit','missing_only','bias_only','self_input_missing_only'):
        raise ValueError('Unknown inference policy')
    refs = {}
    for scenario in result['validation']:
        # Only final selected configurations; factor candidates never enter test.
        if scenario.startswith('look_x'): continue
        p = path.parent/'predictions'/f'validation__{scenario}.npz'
        if p.exists(): refs[scenario] = record(p)
    return dict(id=job_id, family=family, base=base, banks=banks, policy=policy,
                parameters=parameters, validation_result=record(path), validation_predictions=refs,
                selection_manifest=record(path.with_name('frozen_selection.json')) if path.with_name('frozen_selection.json').exists() else None,
                seed=base['selection']['seed'], filling=base['selection']['filling_strategy'],
                fusion=base['selection']['fusion_position'])


def job_artifacts(job):
    b = job['base']
    out = [b['checkpoint'], b['labels'], b['validation_manifest'], job['validation_result'],
           *b['generators'].values(), *job['validation_predictions'].values()]
    for bank in job['banks'].values(): out.extend([bank['manifest'], *bank['artifacts']])
    if job.get('selection_manifest'):out.append(job['selection_manifest'])
    if job['policy'] == 'ssf': out.extend(job['parameters'].values())
    return out


def verify_jobs(jobs):
    unique = {}
    for job in jobs:
        for rec in job_artifacts(job):
            if rec['path'] in unique and unique[rec['path']] != rec:
                raise ValueError('Conflicting dependency identity')
            unique[rec['path']] = rec
    for rec in unique.values(): verify(rec)
    return list(unique.values())


def collect_starts(req, require_complete=True):
    """Read authoritative atomic receipts, not the live display summary."""
    from look.runtime.scheduler import validate_receipt
    reused = read(req['reused']['path'])
    rows = dict(reused['completed_cases']); chosen = dict(reused['selected_contexts'])
    plan = read(req['validation_plan']['path']); q = read(req['validation_queue'])
    for job in plan['jobs']:
        if job['id'] not in q['completed']: continue
        rp = Path(req['validation_queue']).parent/'cases'/job['id']/'queue_complete.json'
        if record(rp) != q['completed'][job['id']]: raise ValueError('Validation receipt changed')
        receipt = validate_receipt(rp, req['validation_identity'], job)
        if job['kind'] == 'suffix':
            rows[job['id']] = dict(job['case'], result=receipt['result'])
        elif job['kind'] == 'selected': chosen[job['context']] = receipt['result']
        else: raise ValueError('Unexpected resumed-start job')
    if require_complete and (len(rows) != 243 or len(chosen) != 27):
        raise ValueError('All 243 starts and 27 selected contexts must finish before test')
    return rows, chosen


def verify_selection(context, rows, selected_path):
    """Reproduce deterministic selection from validation predictions only."""
    import numpy as np
    from look.evaluation.stability import logit_metrics
    group = sorted([r for r in rows.values() if r['context'] == context], key=lambda r:r['start_ordinal'])
    if [r['start_ordinal'] for r in group] != list(range(1, 10)):
        raise ValueError('Selected context lacks nine ordered starts')
    path = Path(selected_path); frozen = read(path.with_name('frozen_selection.json'))
    if read(path)['frozen_selection_sha256'] != record(path.with_name('frozen_selection.json'))['sha256']:
        raise ValueError('Selected-start freeze changed')
    banks = {}
    for pattern in PATTERNS:
        scores = []
        for row in group:
            rp = verify(row['result']); result = read(rp); sealed_result(result)
            score = result['validation']['look_after_fill_'+pattern]['macro_f1']
            with np.load(rp.parent/'predictions'/f'validation__look_after_fill_{pattern}.npz', allow_pickle=False) as p:
                if logit_metrics(p['labels'],p['logits'])['macro_f1'] != score:
                    raise ValueError('Validation selection score is not reproduced by predictions')
            if not math.isfinite(score): raise ValueError('Nonfinite validation selection')
            scores.append((score,row))
        best = min(scores, key=lambda x:(-x[0],x[1]['start_ordinal']))[1]
        d = frozen['directions'][pattern]
        if d['result'] != best['result'] or d['start_ordinal'] != best['start_ordinal']:
            raise ValueError('Start differs from the fixed validation selection rule')
        banks[pattern] = bank_record(Path(best['result']['path']).parent/'look'/pattern)
    return banks


def build_jobs(req, *, require_complete=True):
    from look.studies.starts import source_result
    parent_path = verify(req['parent']); parent = read(parent_path)
    if parent['status'] != 'complete' or len(parent['completed_stages']) != 52:
        raise ValueError('Incomplete original 52 stages')
    jobs = []; originals = {}
    for stage in sorted(parent['completed_stages']):
        if stage.startswith(('screen_','replicate_')): continue
        family = ('single_modality' if stage.startswith('reference_') else
                  'ablation' if stage.startswith(('input_only_','fusion_only_')) else 'original')
        job = result_job(source_result(parent, parent_path.parents[2], stage), stage, family)
        jobs.append(job)
        if family == 'original': originals[stage] = job
    for field, expected in [('methods',15),('self_input',3)]:
        summary = read(verify(req[field]))
        if summary['status'] != 'complete' or summary['test_access'] is not False or len(summary['completed_cases']) != expected:
            raise ValueError('Incomplete method/self-input evidence')
        for name, rec in sorted(summary['completed_cases'].items()):
            path = verify(rec); result = read(path)
            source = originals[f'normalized_mean_layer3_{result["case"]["seed"]}']
            jobs.append(result_job(path,name,'method',source['base']))
    if require_complete:
        rows, selected = collect_starts(req)
    else:
        reused = read(verify(req['reused'])); rows,selected = reused['completed_cases'],reused['selected_contexts']
    for context, rec in sorted(selected.items()):
        path = verify(rec); banks = verify_selection(context,rows,path)
        jobs.append(result_job(path,'selected__'+context,'selected',originals[context]['base'],banks))
    for seed in SEEDS:
        context=f'normalized_mean_layer3_{seed}'
        row=next(r for r in rows.values() if r['context']==context and r['start_ordinal']==9)
        jobs.append(result_job(verify(row['result']),f'terminal_only_{seed}','terminal',originals[context]['base']))
    counts = Counter(j['family'] for j in jobs)
    if require_complete and counts != dict(original=27,ablation=6,single_modality=6,method=18,selected=27,terminal=3):
        raise ValueError(f'Unexpected test roster: {counts}')
    if len({j['id'] for j in jobs}) != len(jobs): raise ValueError('Duplicate test job')
    return jobs


def freeze(req_path, destination, source_hash):
    req = read(req_path)
    if dependency_status(req) != 'ready': raise ValueError('Validation is not complete')
    dest = Path(destination)
    if dest.exists():
        prior = read(dest)
        if prior['request'] != record(req_path) or prior['source_manifest_sha256'] != source_hash:
            raise ValueError('Frozen test identity changed')
        verify_jobs(prior['jobs']); return prior
    verify_numerics(req,Path(__file__).resolve().parents[3])
    sources=verify_source_inventory(req)
    jobs = build_jobs(req)
    media = verify_test_media(req,jobs)
    # Pins include original PCA provenance through source inventory and the actual
    # selected artifacts, which embed the frozen train PCA used at inference.
    artifacts = verify_jobs(jobs)
    value = dict(protocol=PROTOCOL, status='frozen', cohort='test', test_access=False,
                 request=record(req_path), source_manifest_sha256=source_hash,
                 validation_queue=record(req['validation_queue']), jobs=jobs,test_media=media,
                 scientific_kernels=req['scientific_kernels'],source_artifacts=sources,
                 artifacts=artifacts, report_rules=report_rules())
    atomic(value,dest);return value


def report_rules():
    return dict(primary_metric='macro_f1', prediction='argmax_raw_logits',
        secondary=['macro_auroc_ovr','macro_auprc_ovr','nll','brier','ece'],
        scope='balanced internal test; no natural-distribution or external-cohort claim',
        seed_summary='three independent model seeds; same participants, never pooled as 870 independent observations',
        comparisons=['original versus its filling baseline','selected versus original',
                     'methods and terminal-only versus original and filling','input-only/fusion-only versus original',
                     'complete multimodal versus unimodal descriptive references'],
        bootstrap='2000 paired participant resamples, same draw across all model seeds; mean seed Macro-F1 difference',
        multiple_testing='Holm within original-LOOK and selected-start directional comparison families',
        negative_results='retain all completed cases; missing/failed cases never zero-filled or included in 3-seed aggregates',
        test_feedback='no reselection, refitting, threshold change, or adaptive experiment addition from test scores')


def verify_numerics(req, root):
    records=req['scientific_kernels']
    if not records:raise ValueError('Numerical kernel provenance is required')
    for relative,rec in records.items():
        verify(rec)
        current=record(Path(root)/relative)
        if current['sha256']!=rec['sha256'] or current['bytes']!=rec['bytes']:
            raise ValueError('A numerical kernel differs from validation: '+relative)


def verify_test_media(req, jobs):
    import json
    import pandas as pd
    ledger=verify(req['test_media_manifest'])
    entries=[json.loads(line) for line in ledger.read_text().splitlines()]
    if len(entries)!=1160:raise ValueError('Expected 1160 paired bilateral test images')
    if any(j['base']['config']['image_root']!=req['image_root'] for j in jobs):
        raise ValueError('Test image root differs from verified media root')
    labels={j['base']['labels']['sha256'] for j in jobs}
    if len(labels)!=1:raise ValueError('Evaluation jobs differ in participant split')
    image_columns=['left_fundus_path','right_fundus_path','left_oct_path','right_oct_path']
    frame=pd.read_csv(jobs[0]['base']['labels']['path'],dtype={'participant_id':str},
                      usecols=['participant_id','split','instance',*image_columns])
    expected=set(frame.loc[frame['split']=='test',image_columns].to_numpy().ravel())
    if expected!={r['path'] for r in entries}:raise ValueError('Test image ledger differs from actual loader paths')
    sets={name:set(frame.loc[frame['split']==name,'participant_id']) for name in ('train','validation','test')}
    if [len(sets[n]) for n in ('train','validation','test')]!=[1264,296,290]:raise ValueError('Frozen cohort counts differ')
    if any(sets[a]&sets[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('Participant leakage between splits')
    for r in entries:
        current=record(Path(req['image_root'])/r['path'])
        if current['sha256']!=r['sha256'] or current['bytes']!=r['bytes']:raise ValueError('Test image hash differs')
    return dict(manifest=req['test_media_manifest'],verified_image_files=1160,participants=290,participant_overlap=False)


def verify_source_inventory(req):
    sources=read(verify(req['reused']))['sources'];pinned={}
    for source_record in sources.values():
        source=read(verify(source_record));pinned[source_record['path']]=source_record
        records=[source[k] for k in ('result','manifest','checkpoint','pca_manifest','labels','natural_labels')]
        records += source['pcas']+list(source['generators'].values())
        for r in records:
            rec={k:r[k] for k in ('path','bytes','sha256')}
            if rec['path'] in pinned and pinned[rec['path']]!=rec:raise ValueError('Conflicting source provenance')
            pinned[rec['path']]=rec
    for rec in pinned.values():verify(rec)
    return list(pinned.values())
