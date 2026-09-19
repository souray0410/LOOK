"""Finite fresh-host cohort package using the same LOOK fitting/search kernels.

This is a declared study, not a fallback reader for Ibex parent experiments.
The public initialization and small-cohort identity are never called trained parents.
"""
import argparse
import fcntl
import json
import os
import random
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
from look.data.array_pair import ArrayPair
from look.data.observed_pair import collate_observed
from look.runtime.state import atomic_write_json, file_sha256, stable_hash
from look.runtime.host_checkpoint import cpu_tree
from look.models.native_host import build_native_host
from look.training.observed_host import train_host, validate_config
from look.methods.operator import prepare_complete_pca_bank
from look.methods.joint import correction_sites
from look.methods.family_greedy import fit_family_trajectory
from look.studies.family_search_case import load_bank
from look.studies.project_case import CheckedLoader
from look.evaluation.evaluator import evaluate_missing, save_prediction_bundle
from look.evaluation.observed_suite import check_matched
from look.training.mechanism_training import state_equal
from look.analysis.observed_report import simultaneous_bootstrap

ARMS=('residual_rrr','pca_free_mean')
PATTERNS=('oct_missing','cfp_missing')


def read(p):return json.loads(Path(p).read_text())



def validate(s):
    if s['schema']!='look_fresh_cohort_delivery_v1' or s['test_access'] is not False:
        raise ValueError('Undeclared study')
    if s['seed']!=3416 or s['arms']!=list(ARMS) or s['search']!='positive_forward_tree':raise ValueError('Unregistered package')
    if s.get('study_kind')=='fusion_stage_v1':
        from look.studies.cohort_fusion_stage import validate_member
        validate_member(s)
    elif (s['factor'],s['rank'],s['position'])!=(16,32,'deep') or s['architecture'] not in ('resnet18','resnet34','resnet50','densenet121'):
        raise ValueError('Unregistered scope')
    if s['initialization']['kind']!='public_imagenet_fresh_host':raise ValueError('Fresh host required')
    if 'mmtm' in s and (s['architecture']!='resnet18' or s['mmtm']!={'stage':'stage3','ratio':4,'gate_scale':1.0}):
        raise ValueError('Unregistered MMTM host variant')
    validate_config(s['training'])
    if file_sha256(Path(s['data_root'])/'accepted.json')!=s['data_audit_sha256']:raise ValueError('Data identity changed')
    for row in s['source_pins']:
        if file_sha256(row['path'])!=row['sha256']:raise ValueError('Source changed')
    if file_sha256(s['initialization']['path'])!=s['initialization']['sha256']:raise ValueError('Initialization changed')


def make_graph(s):
    from mhd_framework.models import create_model
    # Explicit weight version; TORCH_HOME binding points to the verified public asset.
    torch.hub.set_dir(str(Path(s['initialization']['path']).parent.parent))
    cfg=dict(name=s['architecture'],spatial_dims=2,in_channels=3,num_classes=2,views=1,granularity='block')
    first=create_model(cfg,weights='IMAGENET1K_V1'); second=create_model(cfg,weights='IMAGENET1K_V1')
    return build_native_host(SimpleNamespace(graph=first),SimpleNamespace(graph=second),s['position'],'cuda:0',mmtm=s.get('mmtm'))


def load_selected(s,root):
    r=read(root/'host/accepted.json')
    if r['identity']!=stable_hash(s) or r['state']!='accepted':raise ValueError('Host unaccepted')
    for name,sha in r['files'].items():
        if file_sha256(root/'host'/name)!=sha:raise ValueError('Host evidence changed')
    g=make_graph(s);state=torch.load(root/'host/best.pt',map_location='cpu',weights_only=False)
    if state['node_ids']!=[(n.id,n.name) for n in sorted(g.nodes,key=lambda n:n.id)]:raise ValueError('Node identities differ')
    g.load_state_dict(state['model'],strict=True);g.eval()
    for p in g.parameters():p.requires_grad_(False)
    return g,r


def work(s,root,stage):
    import psutil
    validate(s);root=Path(root);identity=stable_hash(s); stop=False;resource_peak={'rss':0}
    if s.get('study_kind')=='fusion_stage_v1' and (root/'repair_resume_0207.json').exists():
        from look.studies.cohort_fusion_stage import verify_management_overlay
        verify_management_overlay(root/'management_overlay.json')
    def request(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,request);signal.signal(signal.SIGUSR1,request)
    props=torch.cuda.get_device_properties(0)
    if torch.cuda.mem_get_info()[0]<s['gpu_reserve_bytes']+s['gpu_budget_bytes']:raise MemoryError('Exclusive budget unavailable')
    torch.cuda.set_per_process_memory_fraction(s['gpu_budget_bytes']/props.total_memory)
    torch.set_num_threads(2);torch.manual_seed(s['seed']);np.random.seed(s['seed']);random.seed(s['seed'])
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    def check():
        if s.get('disk_reserve_bytes',0) and shutil.disk_usage(root).free<s['disk_reserve_bytes']:
            raise OSError('Artifact volume reserve breached; preserve existing outputs')
        rss=psutil.Process().memory_info().rss;resource_peak['rss']=max(resource_peak['rss'],rss)
        if rss>.85*s['ram_budget_bytes']:raise MemoryError('Host reserve breached')
        if torch.cuda.mem_get_info()[0]<s['gpu_reserve_bytes']:raise MemoryError('Device reserve breached')
        return stop
    train=ArrayPair(s['data_root'],'train',augment=True,seed=s['seed'])
    fit=ArrayPair(s['data_root'],'train');dev=ArrayPair(s['data_root'],'development')
    if set(fit.participant_ids)&set(dev.participant_ids):raise ValueError('Train/dev overlap')
    def loader(ds):return CheckedLoader(DataLoader(ds,batch_size=s['training']['microbatch'],shuffle=False,
        num_workers=0,collate_fn=collate_observed,generator=torch.Generator().manual_seed(s['seed'])),check)
    atomic_write_json(dict(stage=stage,state='running',time=time.time(),pid=os.getpid(),
        gpu=props.name,torch=torch.__version__,cuda=torch.version.cuda,identity=identity),root/(stage+'_status.json'))
    if stage in ('host','profile'):
        g=make_graph(s)
        target=root/('host' if stage=='host' else 'profile/host')
        if stage=='host':
            profile=read(root/'profile/accepted.json')
            if profile['identity']!=identity or profile['gpu']!=props.name:raise ValueError('Profile mismatch')
        r=train_host(g,train,dev,s['training'],s['seed'],target,identity,torch.device('cuda:0'),check,
            preflight_updates=1 if stage=='profile' else None)
        if stage=='profile':
            # Resume the exact same run state for the next actual update.
            del g;torch.cuda.empty_cache();g=make_graph(s)
            second=train_host(g,train,dev,s['training'],s['seed'],target,identity,torch.device('cuda:0'),check,preflight_updates=1)
            if read(target/'status.json')['updates']!=2:raise ValueError('Recovery did not advance')
            peak=torch.cuda.max_memory_reserved()
            if peak>s['gpu_budget_bytes'] or check():raise MemoryError('Profile reserve')
            atomic_write_json(dict(identity=identity,state='accepted',gpu=props.name,peak_reserved=peak,
                train=len(train),dev=len(dev),resume_updates=2,coverage='full_dev_max_observed_eye_batch_two_training_updates_resume_not_convergence'),root/'profile/accepted.json')
        elif r.get('state')!='accepted':raise RuntimeError('Host not yet accepted: '+str(r.get('state')))
        elif s.get('study_kind')=='fusion_stage_v1':
            receipt=read(target/'accepted.json')
            from look.studies.cohort_fusion_stage import runtime_host_structure
            structure=runtime_host_structure(g,s['position'])
            if structure!=s['fusion_stage_structure']:
                raise ValueError('Fusion-stage runtime structure differs from prespecified contract')
            if receipt.get('fusion_stage_structure') not in (None,structure):
                raise ValueError('Fusion-stage host structure changed')
            if receipt.get('fusion_stage_structure') is None:
                receipt['fusion_stage_structure']=structure
                atomic_write_json(receipt,target/'accepted.json')
    else:
        g,h=load_selected(s,root);frozen=cpu_tree(g.state_dict())
        bank=prepare_complete_pca_bank(g,loader(fit),correction_sites(g),[16],32,torch.device('cuda:0'),root/'pca',
            dict(case=identity,host_best_sha256=h['files']['best.pt']),root/'quarantine',load_only=stage!='pca',strict_rank=True)
        if stage=='fit_profile':
            if len(dev)!=296:raise ValueError('Fusion fitting profile requires full 296-person development set')
            from look.studies.fusion_cache_revalidation import revalidate_or_fit
            profile_root=root/'profile';raw_root=profile_root/'fitting';revalidated_root=profile_root/'revalidated'
            revalidated_root.mkdir(parents=True,exist_ok=True)
            torch.cuda.reset_peak_memory_stats();disk_before=shutil.disk_usage(root).free;records=[]
            def role_loader(role):
                return loader(fit if role=='train' else dev)
            def fresh_graph():
                graph,_=load_selected(s,root)
                return graph
            for arm in ARMS:
                for pattern in PATTERNS:
                    source=raw_root/arm/pattern
                    target=revalidated_root/arm/pattern
                    arts,result,receipt=revalidate_or_fit(source=source,target=target,graph=g,loader=role_loader,
                        device=torch.device('cuda:0'),arm=arm,pattern=pattern,sites=correction_sites(g),factor=16,
                        pca_bank=bank,identity=identity,workspace_bytes=s['workspace_bytes'],should_pause=check,
                        load_fresh_graph=fresh_graph)
                    receipt_path=target.parent/'audit'/target.name/'accepted.json'
                    records.append(dict(arm=arm,pattern=pattern,source_tree_present=source.exists(),
                        revalidated_root=str(target.relative_to(root)),
                        revalidation_receipt=str(receipt_path.relative_to(root)),
                        revalidation_receipt_sha256=file_sha256(receipt_path),
                        selection_sha256=file_sha256(target/'selection.json'),
                        bank_sha256=file_sha256(target/'bank.pt'),
                        final=result['final']))
            state_equal(g,frozen);check();peak=torch.cuda.max_memory_reserved();disk_after=shutil.disk_usage(root).free
            if peak>s['gpu_budget_bytes']:raise MemoryError('Fusion fitting profile GPU budget breached')
            profile_bytes=sum(x.stat().st_size for x in profile_root.rglob('*') if x.is_file())
            atomic_write_json(dict(schema='look_fusion_fitting_profile_v2',state='accepted',identity=identity,test_access=False,
                methods=list(ARMS),patterns=list(PATTERNS),development=len(dev),fresh_host_revalidation=True,
                no_duplicate_refit=True,records=records,gpu_peak_reserved_bytes=peak,rss_peak_bytes=resource_peak['rss'],
                profile_bytes=profile_bytes,disk_free_before=disk_before,disk_free_after=disk_after,
                disk_reserve_bytes=s.get('disk_reserve_bytes',0)),raw_root/'accepted.json')
        elif stage in ARMS:
            out=root/stage;records=[];profile_receipt=None;migration_patterns={}
            if s.get('study_kind')=='fusion_stage_v1':
                profile_receipt=read(root/'profile/fitting/accepted.json')
                if (profile_receipt.get('schema')!='look_fusion_fitting_profile_v2'
                        or profile_receipt.get('state')!='accepted' or profile_receipt.get('identity')!=identity
                        or profile_receipt.get('test_access') is not False
                        or profile_receipt.get('fresh_host_revalidation') is not True
                        or profile_receipt.get('no_duplicate_refit') is not True):
                    raise ValueError('Fusion fitting profile is not accepted for logical promotion')
            for pattern in PATTERNS:
                atomic_write_json(dict(state='running',pattern=pattern,time=time.time()),out/'status.json')
                if profile_receipt is not None:
                    matches=[row for row in profile_receipt['records'] if row['arm']==stage and row['pattern']==pattern]
                    if len(matches)!=1:raise ValueError('Fusion revalidated profile record mismatch')
                    row=matches[0];correction=root/row['revalidated_root']
                    receipt_path=root/row['revalidation_receipt']
                    if (file_sha256(receipt_path)!=row['revalidation_receipt_sha256']
                            or file_sha256(correction/'selection.json')!=row['selection_sha256']
                            or file_sha256(correction/'bank.pt')!=row['bank_sha256']):
                        raise ValueError('Fusion revalidated profile evidence changed')
                    revalidation=read(receipt_path)
                    if (revalidation.get('schema')!='look_fusion_fresh_revalidation_v2'
                            or revalidation.get('state')!='accepted' or revalidation.get('identity')!=identity
                            or revalidation.get('test_access') is not False
                            or revalidation.get('source_raw_manifest_unchanged') is not True
                            or revalidation.get('graph_state_exact_before_after_and_fresh') is not True
                            or revalidation.get('references_within_revalidated_tree') is not True
                            or revalidation.get('relocation_no_scientific_value_change') is not True
                            or revalidation.get('complete_source_science_exact') is False):
                        raise ValueError('Fusion revalidation receipt changed')
                    arts=load_bank(correction);selection=read(correction/'selection.json')
                else:
                    correction=out/'corrections'/pattern
                    arts,_=fit_family_trajectory(g,loader(fit),loader(dev),arm=stage,pattern=pattern,sites=correction_sites(g),
                        factor=16,candidates=[dict(rank=32,ridge_lambda=None)],pca_bank=bank,identity=identity,
                        output=correction,device=torch.device('cuda:0'),workspace_bytes=s['workspace_bytes'],
                        mode='positive_forward_tree',should_pause=check,penalty_policy='prefix_train_pca_gcv')
                    selection=None;receipt_path=None
                a=evaluate_missing(g,loader(dev),torch.device('cuda:0'),fixed_pattern=pattern,artifact_banks={pattern:arts})
                b=evaluate_missing(g,loader(dev),torch.device('cuda:0'),fixed_pattern=pattern,
                    artifact_banks={pattern:load_bank(correction)})
                check_matched(a,b)
                if not np.array_equal(a['logits'],b['logits']):raise ValueError('Correction replay mismatch')
                if selection is not None:
                    from look.studies.fusion_cache_revalidation import prediction_values_sha
                    if (selection['final']['values_sha256']!=prediction_values_sha(a)
                            or selection['final']['metrics']!=a['metrics']):
                        raise ValueError('Fresh revalidated fitting profile changed on formal replay')
                    migration_patterns[pattern]=dict(
                        revalidated_root=str(correction.relative_to(root)),
                        revalidation_receipt=str(receipt_path.relative_to(root)),
                        revalidation_receipt_sha256=file_sha256(receipt_path),
                        selection_sha256=file_sha256(correction/'selection.json'),
                        bank_sha256=file_sha256(correction/'bank.pt'),
                        selected_path=selection['selected_path'],
                        final_values_sha256=selection['final']['values_sha256'],
                        revalidated_science_sha256=revalidation['revalidated_science_sha256'],
                        source_science_sha256=revalidation['source_science_sha256'],
                        runtime_sha256=stable_hash(revalidation['runtime']),
                        graph_state_sha256=revalidation['graph_state_sha256'],
                        source_raw_manifest_sha256=revalidation['source_raw_manifest_sha256'],
                        source_raw_manifest_final_sha256=revalidation['source_raw_manifest_final_sha256'],
                        relocation_receipt_sha256=revalidation['relocation_receipt_sha256'],
                        relocation_mapping_count=revalidation['relocation_mapping_count'])
                baseline=evaluate_missing(g,loader(dev),torch.device('cuda:0'),fixed_pattern=pattern);check_matched(a,baseline)
                for method,result in ((stage,a),('host',baseline)):
                    p=out/'development'/f'{method}_{pattern}.npz';save_prediction_bundle(result,p)
                    records.append(dict(method=method,scenario=pattern,path=str(p),sha256=file_sha256(p),metrics=result['metrics']))
            state_equal(g,frozen)
            accepted=dict(state='accepted',identity=identity,test_access=False,records=records,
                host_best_sha256=h['files']['best.pt'],replay_exact=True)
            if profile_receipt is not None:
                accepted['profile_migration']=dict(schema='look_fusion_profile_migration_v2',no_refit=True,
                    no_physical_relocation=True,profile_receipt_sha256=file_sha256(root/'profile/fitting/accepted.json'),
                    source_role='fresh_revalidated_same_run_same_identity',patterns=migration_patterns)
            atomic_write_json(accepted,out/'accepted.json')
    atomic_write_json(dict(stage=stage,state='completed',time=time.time(),identity=identity,
        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved()),root/(stage+'_status.json'))


def verify_fusion_formal_arm(s,root,arm):
    root=Path(root);receipt=read(root/arm/'accepted.json');identity=stable_hash(s)
    if receipt.get('state')!='accepted' or receipt.get('identity')!=identity or receipt.get('test_access') is not False:
        raise ValueError('Fusion formal arm acceptance changed')
    host=read(root/'host/accepted.json')
    if receipt.get('host_best_sha256')!=host['files']['best.pt'] or receipt.get('replay_exact') is not True:
        raise ValueError('Fusion formal arm host/replay identity changed')
    migration=receipt.get('profile_migration',{})
    if (migration.get('schema')!='look_fusion_profile_migration_v2' or migration.get('no_refit') is not True
            or migration.get('no_physical_relocation') is not True
            or migration.get('source_role')!='fresh_revalidated_same_run_same_identity'):
        raise ValueError('Fusion formal arm lacks accepted logical profile migration')
    if file_sha256(root/'profile/fitting/accepted.json')!=migration.get('profile_receipt_sha256'):
        raise ValueError('Fusion fitting profile receipt changed')
    if set(migration.get('patterns',{}))!=set(PATTERNS):
        raise ValueError('Fusion logical migration pattern coverage changed')
    for pattern,row in migration['patterns'].items():
        correction=root/row['revalidated_root'];receipt_path=root/row['revalidation_receipt']
        if (file_sha256(receipt_path)!=row['revalidation_receipt_sha256']
                or file_sha256(correction/'selection.json')!=row['selection_sha256']
                or file_sha256(correction/'bank.pt')!=row['bank_sha256']):
            raise ValueError('Fusion logical migration evidence changed')
        revalidation=read(receipt_path)
        if (revalidation.get('schema')!='look_fusion_fresh_revalidation_v2'
                or revalidation.get('source_raw_manifest_unchanged') is not True
                or revalidation.get('graph_state_exact_before_after_and_fresh') is not True
                or revalidation.get('references_within_revalidated_tree') is not True
                or revalidation.get('revalidated_science_sha256')!=row['revalidated_science_sha256']
                or revalidation.get('source_science_sha256')!=row['source_science_sha256']
                or stable_hash(revalidation.get('runtime'))!=row['runtime_sha256']
                or revalidation.get('graph_state_sha256')!=row['graph_state_sha256']
                or revalidation.get('source_raw_manifest_sha256')!=row['source_raw_manifest_sha256']
                or revalidation.get('source_raw_manifest_final_sha256')!=row['source_raw_manifest_final_sha256']
                or revalidation.get('relocation_receipt_sha256')!=row['relocation_receipt_sha256']
                or revalidation.get('relocation_mapping_count')!=row['relocation_mapping_count']
                or revalidation.get('relocation_no_scientific_value_change') is not True
                or revalidation.get('complete_source_science_exact') is False):
            raise ValueError('Fusion logical migration revalidation changed')
        selection=read(correction/'selection.json')
        if selection['selected_path']!=row['selected_path'] or selection['final']['values_sha256']!=row['final_values_sha256']:
            raise ValueError('Fusion logical migration scientific selection changed')
    for row in receipt['records']:
        if file_sha256(row['path'])!=row['sha256']:raise ValueError('Fusion formal prediction changed')
    return receipt


def report(s,root):
    rows=[];arrays={};host_sha=set()
    for arm in ARMS:
        r=read(root/arm/'accepted.json')
        if r['state']!='accepted' or r['identity']!=stable_hash(s):raise ValueError('Unaccepted arm')
        host_sha.add(r['host_best_sha256'])
        for x in r['records']:
            if file_sha256(x['path'])!=x['sha256']:raise ValueError('Prediction changed')
            a=dict(np.load(x['path'],allow_pickle=False));key=(x['method'],x['scenario'])
            if key in arrays:
                if not all(np.array_equal(arrays[key][k],a[k]) for k in a):raise ValueError('Baseline differs')
            else:arrays[key]=a;rows.append(dict(method=key[0],scenario=key[1],metrics=x['metrics']))
    if len(host_sha)!=1:raise ValueError('Unmatched hosts')
    keys=list(arrays);ref=arrays[keys[0]];weights=[];definitions=[]
    for a in arrays.values():
        for k in ('participant_ids','labels'):
            if not np.array_equal(ref[k],a[k]):raise ValueError('Report identity mismatch')
    for p in PATTERNS:
        for a,b in ((ARMS[0],ARMS[1]),(ARMS[0],'host'),(ARMS[1],'host')):
            w=np.zeros(len(keys));w[keys.index((a,p))]=1;w[keys.index((b,p))]=-1
            weights.append(w);definitions.append(dict(pattern=p,method=a,reference=b))
    stats=simultaneous_bootstrap(ref['labels'],np.stack([arrays[k]['logits'].argmax(1) for k in keys]),weights,10000,7341618)
    out=root/'delivery';out.mkdir(exist_ok=True)
    atomic_write_json(dict(results=rows,statistics=stats,comparisons=definitions,
        conclusion_scope='small_cohort_single_seed_dev_selection_not_independent_test',test_access=False),out/'results.json')
    lines=['# LOOK 小队列拟合方法匹配结果','',f'1264 train / 296 dev；首种子3416；同一个重新训练的{s.get("architecture","synthetic")}深层融合宿主。',
        '两方法均自由均值、秩32、二维空间x16、正收益树；公开ImageNet初始化，不复用历史医学宿主。',
        '开发集选择有偏，配对区间不能消除选择偏差；不与Ibex不同队列混合排名。','',
        '|方法|缺失状态|Macro-F1 (%)|AUROC (%)|','|---|---|---:|---:|']
    for r in rows:lines.append(f"|{r['method']}|{r['scenario']}|{100*r['metrics']['macro_f1']:.3f}|{100*r['metrics']['macro_auroc_ovr']:.3f}|")
    (out/'README.zh-CN.md').write_text('\n'.join(lines)+'\n')
    atomic_write_json(dict(state='accepted',identity=stable_hash(s),test_access=False,
        report_source_sha256=file_sha256(__file__),files={n:file_sha256(out/n) for n in ('results.json','README.zh-CN.md')}),out/'accepted.json')


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--stage',required=True,
        choices=['pipeline','profile','host','pca','fit_profile',*ARMS,'report']);a=p.parse_args();s=read(a.spec);root=Path(s['output']);root.mkdir(parents=True,exist_ok=True)
    if a.stage=='report':report(s,root);return
    if a.stage=='pipeline':
        validate(s)
        with (root/'pipeline.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            def launch(stage,device):
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(device))
                log=(root/(stage+'.log')).open('a')
                child=subprocess.Popen([sys.executable,'-m','look.studies.cohort_delivery','--spec',a.spec,'--stage',stage],env=env,stdout=log,stderr=subprocess.STDOUT)
                log.close();return child
            def wait(stage,child):
                code=child.wait()
                if code:raise RuntimeError(f'Stage {stage} failed ({code}); preserve outputs for repair')
            try:
                previous=root/'pipeline_status.json'
                if previous.exists() and read(previous).get('state')=='needs_review':
                    atomic_write_json(read(previous),root/'incidents'/f'pipeline_before_resume_{time.time_ns()}.json')
                atomic_write_json(dict(state='running',identity=stable_hash(s),time=time.time(),pid=os.getpid()),previous)
                for stage in ('profile','host','pca','fit_profile'):
                    # Host checks its accepted receipt; all other stages verify/resume their own outputs.
                    if stage=='profile' and (root/'profile/accepted.json').exists():
                        if read(root/'profile/accepted.json')['identity']!=stable_hash(s):raise ValueError('Profile identity changed')
                        continue
                    if stage=='host' and (root/'host/accepted.json').exists():
                        receipt=read(root/'host/accepted.json')
                        if receipt['identity']!=stable_hash(s):raise ValueError('Host identity changed')
                        for name,sha in receipt['files'].items():
                            if file_sha256(root/'host'/name)!=sha:raise ValueError('Host evidence changed')
                        if s.get('study_kind')=='fusion_stage_v1' and receipt.get('fusion_stage_structure')!=s['fusion_stage_structure']:
                            raise ValueError('Fusion-stage accepted host structure changed')
                        continue
                    if stage=='fit_profile' and (root/'profile/fitting/accepted.json').exists():
                        receipt=read(root/'profile/fitting/accepted.json')
                        if receipt['identity']!=stable_hash(s) or receipt.get('state')!='accepted':raise ValueError('Fitting profile identity changed')
                        continue
                    wait(stage,launch(stage,s['devices'][0]))
                # Two finite independent arms; share only the immutable accepted host/PCA.
                errors=[]
                if len(s['devices']) == 1:
                    # One project lane: finish each arm before launching the next.
                    for arm in ARMS:
                        if s.get('study_kind')=='fusion_stage_v1' and (root/arm/'accepted.json').exists():
                            verify_fusion_formal_arm(s,root,arm);continue
                        code=launch(arm,s['devices'][0]).wait()
                        if code:errors.append((arm,code))
                else:
                    jobs=[(arm,launch(arm,s['devices'][i])) for i,arm in enumerate(ARMS)]
                    for arm,child in jobs:
                        code=child.wait()
                        if code:errors.append((arm,code))
                if errors:raise RuntimeError('Family failure: '+str(errors))
                report(s,root)
                atomic_write_json(dict(state='accepted',identity=stable_hash(s),time=time.time()),root/'pipeline_status.json')
            except Exception as e:
                atomic_write_json(dict(state='needs_review',error=repr(e),time=time.time()),root/'pipeline_status.json');raise
        return
    uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader','-i',os.environ['CUDA_VISIBLE_DEVICES']],text=True).strip()
    locks=Path(s['lock_root']);locks.mkdir(parents=True,exist_ok=True)
    with (locks/(uuid+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:work(s,root,a.stage)
        except Exception as e:
            import traceback
            atomic_write_json(dict(state='needs_review',error=repr(e),traceback=traceback.format_exc(),time=time.time()),root/(a.stage+'_status.json'));raise

if __name__=='__main__':main()
