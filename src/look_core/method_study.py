"""Bounded method-evidence cases, queued AFTER the full suffix supplement."""
from __future__ import annotations
import gc
import json
import time
from pathlib import Path
import torch

from .state import atomic_write_json, stable_hash, PipelineState, utc_now, file_sha256
from .start_study import (read_json, file_record, verify_file, verify_source, pin_source,
    source_result, verify_parent, SuffixRunner, sites_for_fusion, PATTERNS)
from .reproducibility import implementation_sha256, seed_everything
from .look import load_selected_bank
from .filling import NormalizedMeanFiller
from .data import make_loader
from .evaluate import save_prediction_bundle
from .method_kernels import fit_control, evaluate_control, forward_control
from .method_ssf import SSFAdapter, train_ssf
from .method_analysis import representation_evidence, measure_inference


def make_method_cases(spec):
    scopes={'frozen_look_method_evidence_v1':['ssf','independent_fit','missing_only'],
            'frozen_look_method_evidence_v2':['logit_affine','bias_only','ssf','independent_fit','missing_only']}
    if (spec['protocol'] not in scopes or spec['fusion_position'] != 'layer3'
        or spec['filling'] != 'normalized_mean' or spec['seeds'] != [3407,3408,3409]
        or spec['methods'] != scopes[spec['protocol']]
        or spec['patterns'] != PATTERNS or spec['test_access'] is not False):
        raise ValueError('Unexpected method-evidence scope')
    return [dict(case_id=f'{method}_layer3_normalized_mean_{seed}',method=method,seed=seed,
                 fusion_position='layer3',filling='normalized_mean')
            for method in spec['methods'] for seed in spec['seeds']]


def verify_predecessors(parent, suffix, parent_spec, parent_spec_path):
    verify_parent(parent,parent_spec,parent_spec_path)
    if (suffix.get('status') != 'complete' or len(suffix.get('completed_cases',{})) != 243
        or len(suffix.get('selected_contexts',{})) != 27 or suffix.get('test_access') is not False):
        raise RuntimeError('All 243 suffix cases and 27 selected contexts must finish before method evidence')


def make_resource_case(seed):
    sites=sites_for_fusion('layer3')
    return dict(case_id=f'resource_layer3_{seed}',start_ordinal=1,eligible_sites=sites,
                candidate_sites=sites,allowed_start=sites[0])


def run_method_case(source, case, spec, output, device, devices):
    if case['method']=='logit_affine':
        from .method_logit import run_logit_case
        return run_logit_case(source,case,spec,output,device,devices)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    identity=dict(case=case,spec=spec,source_sha256=stable_hash(source),
                  implementation_sha256=implementation_sha256(Path(__file__).resolve().parents[2]))
    identity_path=output/'identity.json'
    if identity_path.exists() and read_json(identity_path)!=identity:
        raise RuntimeError('Method case identity changed; cross-method resume forbidden')
    atomic_write_json(identity,identity_path)
    result_path=output/'validation_result.json'
    verify_source(source)
    if result_path.exists():
        old=read_json(result_path)
        if old['identity']!=identity or old['status']!='complete': raise RuntimeError('Invalid completed case')
        for artifact in old['provenance']: verify_file(artifact)
        return old
    verify_source(source)
    seed_everything(case['seed'])
    runner=SuffixRunner(source,make_resource_case(case['seed']),output/'resources',output/'cache',device,devices)
    loaders,datasets=runner._build_loaders()
    graph,_=runner._load_frozen_graph(runner._train_or_resume())
    filler=NormalizedMeanFiller()
    pcas=runner._prepare_shared_pca(graph,loaders['look_train'],Path(source['checkpoint']['path']))
    banks,decisions,adapters={}, {}, {}
    started=time.perf_counter()
    try:
        for pattern in PATTERNS:
            if case['method']=='ssf':
                train=make_loader(datasets['train'],spec['ssf']['micro_batch_size'],spec['ssf']['num_workers'],
                    True,case['seed'],runner.config.sampling_strategy)
                decisions[pattern]=train_ssf(graph,train,loaders['validation'],device,pattern,
                    spec['ssf'],output/pattern,stable_hash(source),case['seed'])
                del train;gc.collect()
            else:
                banks[pattern],decisions[pattern]=fit_control(graph,loaders['look_train'],loaders['validation'],
                    device,pcas,runner.config,pattern,case['method'],output/pattern,stable_hash(source))
        if case['method']=='ssf':
            for pattern in PATTERNS:
                adapter=SSFAdapter(graph,pattern)
                rec=decisions[pattern]['selected']
                if file_sha256(Path(rec['checkpoint']))!=rec['checkpoint_sha256']:raise RuntimeError('SSF selected checkpoint changed')
                adapter.load_state_dict(torch.load(rec['checkpoint'],map_location=device,weights_only=True))
                adapters[pattern]=adapter
        def predict(o,c,pattern):
            for key,adapter in adapters.items():adapter.enabled=key==pattern
            return forward_control(graph,o,c)
        validation={}; prediction_records=[]
        scenarios=[('complete',dict(fixed_pattern='complete'))]
        scenarios += [(p,dict(fixed_pattern=p)) for p in PATTERNS]
        scenarios += [(f'random_{r:.1f}',dict(random_ratio=r)) for r in spec['random_ratios']]
        for name,kwargs in scenarios:
            result=evaluate_control(graph,loaders['validation'],device,banks,
                policy='joint' if case['method']=='ssf' else case['method'],filler=filler,
                mask_seed=spec['missingness_seed'],predict=predict if adapters else None,**kwargs)
            p=output/'predictions'/f'validation__{name}.npz';save_prediction_bundle(result,p)
            prediction_records.append(file_record(p));validation[name]=result['metrics']
            if name in PATTERNS:
                expected=decisions[name]['selected']['macro_f1'] if adapters else decisions[name]['macro_f1']
                if validation[name]['macro_f1']!=expected:raise RuntimeError('Frozen direction configuration failed reproduction')
        for adapter in adapters.values():adapter.enabled=False
        diagnostic, costs={},{}
        first=next(iter(loaders['validation']))
        for pattern in PATTERNS:
            for adapter in adapters.values():adapter.enabled=False
            policy='joint' if adapters else case['method']
            diagnostic[pattern]=representation_evidence(graph,loaders['validation'],device,pattern,filler,
                banks.get(pattern,()),policy,output/'analysis'/pattern,adapters.get(pattern))
            costs[pattern]=measure_inference(graph,first,device,pattern,filler,banks.get(pattern,()),policy,adapters.get(pattern))
        costs['filling_only']=measure_inference(graph,first,device,'oct_missing',filler,(),'joint')
        costs['complete']=measure_inference(graph,first,device,'complete',filler,(),'joint')
        # Cost comparison shares source PCA. Do not sum the same basis across directions.
        unique_pca={p.source_id:p for p in pcas.values()}
        costs['shared_pca_fit_seconds']=sum(p.fit_seconds for p in unique_pca.values())
        costs['shared_pca_reused']=True
        costs['case_elapsed_seconds_this_invocation']=time.perf_counter()-started
        atomic_write_json(costs,output/'costs.json')
        provenance=prediction_records+[file_record(identity_path),file_record(output/'costs.json')]
        for pattern in PATTERNS:
            provenance.extend(file_record(p) for p in sorted((output/pattern).rglob('*'))
                if p.is_file() and p.suffix in ('.json','.pt','.npz') and p.name!='last.pt')
            provenance.extend(file_record(p) for p in sorted((output/'analysis'/pattern).glob('*')) if p.is_file())
        record=dict(status='complete',phase='validation',test_access=False,identity=identity,
            case=case,completed_at_utc=utc_now(),source=source['result'],checkpoint=source['checkpoint'],
            validation=validation,decisions=decisions,diagnostics=diagnostic,costs=costs,provenance=provenance)
        atomic_write_json(record,result_path)
        return record
    finally:
        for adapter in adapters.values():adapter.close()
        del graph,runner,loaders,datasets,pcas,banks
        gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()


def analyze_reference(source, label, bank_roots, output, device, devices):
    output=Path(output)
    verify_source(source)
    identity=dict(source_sha256=stable_hash(source),label=label,bank_roots=bank_roots)
    if (output/'complete.json').exists():
        record=read_json(output/'complete.json')
        if record['identity']!=identity:raise RuntimeError('Reference analysis identity changed')
        for p in record['provenance']:verify_file(p)
        return record
    runner=SuffixRunner(source,make_resource_case(source['selection']['seed']),output/'resources',output/'cache',device,devices)
    loaders,datasets=runner._build_loaders();graph,_=runner._load_frozen_graph(runner._train_or_resume())
    banks={p:load_selected_bank(Path(bank_roots[p])) for p in PATTERNS}
    filler=NormalizedMeanFiller();diagnostic,costs={},{}
    first=next(iter(loaders['validation']))
    for pattern in PATTERNS:
        diagnostic[pattern]=representation_evidence(graph,loaders['validation'],device,pattern,filler,
            banks[pattern],'joint',output/pattern)
        costs[pattern]=measure_inference(graph,first,device,pattern,filler,banks[pattern],'joint')
    atomic_write_json(costs,output/'costs.json')
    bank_records=[]
    for root in bank_roots.values():
        root=Path(root);manifest=root/'selected_manifest.json'
        bank_records.append(file_record(manifest))
        bank_records.extend(file_record(root/r['path']) for r in read_json(manifest)['artifacts'])
    record=dict(identity=identity,label=label,seed=source['selection']['seed'],diagnostics=diagnostic,costs=costs,
        provenance=bank_records+[file_record(p) for p in sorted(output.rglob('*')) if p.is_file() and p.suffix in ('.csv','.json')])
    atomic_write_json(record,output/'complete.json')
    del graph,runner,loaders,datasets,banks;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    return record


def run_method_study(paths,parent_path,parent_source,suffix_path,devices,*,execute=False,wait=False,render=True,start_evidence_manifest=None):
    from .start_evidence import load_evidence
    if (suffix_path is None) == (start_evidence_manifest is None):
        raise ValueError('Choose exactly one suffix summary or representative evidence manifest')
    representative = start_evidence_manifest is not None
    parent_path,parent_source=map(lambda p:Path(p).resolve(),(parent_path,parent_source))
    suffix_path=Path(start_evidence_manifest if representative else suffix_path).resolve()
    spec_path=paths.project_root/'configs/method_evidence.json';spec=read_json(spec_path)
    cases=make_method_cases(spec);parent=read_json(parent_path);suffix=load_evidence(suffix_path) if representative else read_json(suffix_path)
    identity=dict(protocol=spec['protocol'],spec_sha256=file_sha256(spec_path),parent_path=str(parent_path),
        parent_implementation=parent['implementation_sha256'],suffix_path=str(suffix_path),
        suffix_identity=stable_hash(suffix['identity']),implementation_sha256=implementation_sha256(paths.project_root),devices=list(devices))
    if representative:identity['start_scope']='representative'
    output=paths.runs_root/'method_study'/stable_hash(identity)[:12]
    if any(output.resolve().is_relative_to(p.parents[2]) for p in ((parent_path,) if representative else (parent_path,suffix_path))):
        raise ValueError('Method output must be in its own release run root')
    summary=dict(identity=identity,spec=spec,cases=cases,output=str(output),status='planned',completed_cases={},
        reference_analyses={},expected_new_cases=len(cases),expected_direction_fits=2*len(cases),combined_development_stages=265+len(cases),
        test_access=False,parent_summary=str(parent_path),suffix_summary=str(suffix_path))
    if representative:
        summary.update(start_evidence_manifest=str(suffix_path),start_scope='representative',combined_development_stages=None)
    if not execute:return summary
    output.mkdir(parents=True,exist_ok=True)
    path=output/'summary.json'
    with PipelineState(output/'state','method_evidence',identity,[spec_path]) as state:
        if path.exists():
            summary=read_json(path)
            if summary['identity']!=identity:raise RuntimeError('Method study identity changed')
        def report(status,**fields):
            summary.update(dict(status=status,updated_at_utc=utc_now(),error=None) | fields)
            atomic_write_json(summary,path)
            print(f'{utc_now()} method_status={status} completed={len(summary["completed_cases"])}/{len(cases)} {fields}',flush=True)
        try:
            report('waiting_predecessors')
            if render:
                from .method_report import write_method_report
                write_method_report(summary,output/'report')
            while parent['status']!='complete' or (not representative and suffix['status']!='complete'):
                if parent['status'] in ('failed','cancelled') or suffix['status'] in ('failed','cancelled'):
                    raise RuntimeError('A predecessor failed; method queue will not start or restart it')
                if not wait:raise RuntimeError('Predecessors incomplete; --wait-predecessors is required')
                time.sleep(30);parent=read_json(parent_path);suffix=read_json(suffix_path)
            parent_spec=parent_source/'configs/unified_study.json'
            if representative:
                verify_parent(parent,read_json(parent_spec),parent_spec)
                suffix=load_evidence(suffix_path)
            else:
                verify_predecessors(parent,suffix,read_json(parent_spec),parent_spec)
            if stable_hash(suffix['identity'])!=identity['suffix_identity']:raise RuntimeError('Suffix identity changed')
            if parent['implementation_sha256']!=identity['parent_implementation']:raise RuntimeError('Parent identity changed')
            inventory_path=output/'source_inventory.json'
            if inventory_path.exists():
                inventory=read_json(inventory_path)
                for rec in inventory['upstream_records']:verify_file(rec)
                for rec in inventory['source_files']:verify_file(rec)
                for source in inventory['sources'].values():verify_source(source)
            else:
                # The controls are separate modules: all original numerical kernels stay byte-identical.
                for folder in ('src/look_core','third_party/MHD_Project/V4'):
                    for p in (parent_source/folder).glob('*.py'):
                        if p.name=='pipeline.py':continue
                        if file_sha256(p)!=file_sha256(paths.project_root/p.relative_to(parent_source)):
                            raise RuntimeError(f'Parent numerical kernel changed: {p.name}')
                sources={str(seed):pin_source(source_result(parent,parent_path.parents[2],f'normalized_mean_layer3_{seed}'),
                    parent,parent_path.parents[2],paths.project_root) for seed in spec['seeds']}
                inventory=dict(sources=sources,upstream_records=[file_record(parent_path),file_record(suffix_path)],
                    scope=spec,source_files=[file_record(p) for p in (paths.project_root/'src/look_core').glob('method_*.py')])
                atomic_write_json(inventory,inventory_path)
            report('running')
            device=torch.device('cuda:0')
            for case in cases:
                report('running',active_case=case['case_id'])
                source=inventory['sources'][str(case['seed'])]
                result=run_method_case(source,case,spec,output/'cases'/case['case_id'],device,devices)
                summary['completed_cases'][case['case_id']]=file_record(output/'cases'/case['case_id']/'validation_result.json')
                report('running',active_case=case['case_id'])
                if render:
                    from .method_report import write_method_report
                    write_method_report(summary,output/'report')
                state.checkpoint(last_case=case['case_id'])
            # Reuse the existing final-node suffix; do not fit a duplicate terminal-only control.
            for seed in spec['seeds']:
                source=inventory['sources'][str(seed)];context=f'normalized_mean_layer3_{seed}'
                original=Path(source['result']['path']).parent/'look'
                rows=[r for r in suffix['completed_cases'].values() if r['context']==context]
                terminal=next(r for r in rows if r['start_ordinal']==9)
                frozen=read_json(Path(suffix['selected_contexts'][context]['path']).with_name('frozen_selection.json'))
                roots={'original_LOOK':{p:str(original/p) for p in PATTERNS},
                       'terminal_only':{p:terminal['banks'][p]['bank_root'] for p in PATTERNS},
                       'selected_start_LOOK':{p:str(Path(frozen['directions'][p]['result']['path']).parent/'look'/p) for p in PATTERNS}}
                for label,bank_roots in roots.items():
                    key=f'{label}_{seed}'
                    summary['reference_analyses'][key]=analyze_reference(source,label,bank_roots,output/'reference_analyses'/key,device,devices)
                    report('running',active_case=key)
            freeze=dict(protocol=spec['protocol'],spec=spec,test_access=False,
                parent=file_record(parent_path),suffix=file_record(suffix_path),
                method_results=summary['completed_cases'],source_inventory=file_record(inventory_path),
                interpretation='Methods and comparisons frozen for later sealed-test evaluation; no test run is authorized here')
            atomic_write_json(freeze,output/'comparison_freeze.json')
            report('complete',active_case=None)
            if render:
                from .method_report import write_method_report
                write_method_report(summary,output/'report')
            state.complete([path,inventory_path,output/'comparison_freeze.json'])
        except BaseException as error:
            report('failed',error=repr(error))
            raise
    return summary
