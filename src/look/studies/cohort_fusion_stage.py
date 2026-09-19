"""Finite WS02 fusion-stage mechanism package.

This is not a scheduler. It prepares and validates one existing cohort_sequence
plan containing one strict accepted deep reference and exactly two new ResNet18
executions: middle and features.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess

from look.runtime.state import atomic_write_json, file_sha256, stable_hash
from look.models.native_host import build_native_host
from look.methods.joint import correction_sites


TASK_ID='look-ws02-fusion-stage-20260919-v1'
STUDY_KIND='fusion_stage_v1'
REPAIR_0038_SHA256='a58ece479f0b0cb8471c1fd62170ab480972ceac83eab9558f26b30de2e8712f'
REPAIR_0130_SHA256='b7c382d8c97fb5b4bac34f631873e5564a6ac610f9c8adbb26a2d8e979ab8cbd'
REPAIR_0207_SHA256='b03c96b816ebf535e15c94b68686ac9d7bacddb3e5b141a2099eceb1cf767eda'
SCIENTIFIC_SOURCE_COMMIT='7876dc85388336acb1b9032ff50b9e0b1db66c28'
NUMERICAL_MODULES=(
    'src/look/models/native_host.py','src/look/training/observed_host.py','src/look/methods/family_greedy.py',
    'src/look/methods/positive_forward_tree.py','src/look/methods/family_statistics.py','src/look/methods/operator.py',
    'src/look/evaluation/evaluator.py','src/look/data/observed_pair.py')
DEEP_RUN_ID='2026_09_18_11_18_28_650020'
DEEP_SPEC_SHA256='f09a452f8584861ff5291cb262894e16f4ba4c7c962f43f6ba4cfbccb18929e8'
DEEP_DELIVERY_SHA256='105f0d94c9020e9b008f46c4f24543878c63b7461fb1c350bbbd3c2d3706eb52'
POSITIONS=('deep','middle','features')
METHODS=('residual_rrr','pca_free_mean')
PATTERNS=('oct_missing','cfp_missing')
CROSS_HOST_COMPARISONS=[
    dict(position=position,reference_position='deep',method=method,pattern=pattern,
         definition='(method-host gain at position) - (method-host gain at deep)')
    for position in ('middle','features') for method in METHODS for pattern in PATTERNS
]
FIXED_KEYS=('schema','architecture','seed','factor','rank','arms','search','data_root','data_audit_sha256',
    'training','framework_commit','initialization','test_access','devices','lock_root',
    'gpu_budget_bytes','gpu_reserve_bytes','ram_budget_bytes','workspace_bytes')


def read(path):
    return json.loads(Path(path).read_text())


def _accepted_reference(root):
    root=Path(root)
    spec=read(root/'spec.json')
    if spec.get('run_id')!=DEEP_RUN_ID or file_sha256(root/'spec.json')!=DEEP_SPEC_SHA256:
        raise ValueError('Deep accepted reference spec identity changed')
    if (spec.get('architecture'),spec.get('position'),spec.get('seed'),spec.get('factor'),spec.get('rank'))!=('resnet18','deep',3416,16,32):
        raise ValueError('Deep accepted reference science changed')
    if spec.get('arms')!=list(METHODS) or spec.get('search')!='positive_forward_tree' or spec.get('test_access') is not False or 'mmtm' in spec:
        raise ValueError('Deep accepted reference scope changed')
    delivery=root/'delivery/accepted.json'
    if not delivery.exists() or file_sha256(delivery)!=DEEP_DELIVERY_SHA256:
        raise ValueError('Deep accepted reference delivery changed')
    receipt=read(delivery)
    if receipt.get('state')!='accepted' or receipt.get('identity')!=stable_hash(spec) or receipt.get('test_access') is not False:
        raise ValueError('Deep accepted reference is not accepted')
    for name,digest in receipt['files'].items():
        if file_sha256(root/'delivery'/name)!=digest:
            raise ValueError('Deep accepted reference report bytes changed')
    return spec,receipt


def _fixed_science(spec):
    return {key:spec[key] for key in FIXED_KEYS}


def runtime_host_structure(graph,position):
    return dict(position=position,fusion_position=graph.fusion_position,
        fusion_endpoint=graph.native_host_provenance['fusion_endpoint'],
        architecture_id=graph.architecture_id,correction_sites=correction_sites(graph),
        parameters=sum(p.numel() for p in graph.parameters()),
        trainable_parameters=sum(p.numel() for p in graph.parameters() if p.requires_grad))


def host_structure(position):
    from types import SimpleNamespace
    from mhd_framework.models import create_model
    cfg=dict(name='resnet18',spatial_dims=2,in_channels=3,num_classes=2,views=1,granularity='block')
    first=create_model(cfg,weights=None)
    second=create_model(cfg,weights=None)
    graph=build_native_host(SimpleNamespace(graph=first),SimpleNamespace(graph=second),position,'cpu')
    return runtime_host_structure(graph,position)


def validate_member(spec):
    if spec.get('study_kind')!=STUDY_KIND or spec.get('fusion_stage_task')!=TASK_ID:
        raise ValueError('Unregistered fusion-stage member')
    if (spec.get('architecture'),spec.get('position'),spec.get('seed'),spec.get('factor'),spec.get('rank')) not in (
            ('resnet18','middle',3416,16,32),('resnet18','features',3416,16,32)):
        raise ValueError('Unregistered fusion-stage member scope')
    if spec.get('arms')!=list(METHODS) or spec.get('search')!='positive_forward_tree' or spec.get('test_access') is not False or 'mmtm' in spec:
        raise ValueError('Fusion-stage member changed')
    if spec.get('disk_reserve_bytes')!=50*1024**3:
        raise ValueError('Fusion-stage disk reserve changed')
    reference=spec.get('fusion_stage_reference',{})
    if reference!={'run_id':DEEP_RUN_ID,'spec_sha256':DEEP_SPEC_SHA256,'delivery_sha256':DEEP_DELIVERY_SHA256}:
        raise ValueError('Fusion-stage reference changed')


def _git_blob_sha(root,commit,rel):
    data=subprocess.check_output(['git','-C',str(root),'show',f'{commit}:{rel}'])
    import hashlib
    return hashlib.sha256(data).hexdigest()


def build_management_overlay_receipt(overlay_root,management_commit):
    overlay_root=Path(overlay_root).resolve()
    head=subprocess.check_output(['git','-C',str(overlay_root),'rev-parse','HEAD'],text=True).strip()
    if head!=management_commit or subprocess.check_output(['git','-C',str(overlay_root),'status','--porcelain'],text=True).strip():
        raise ValueError('Management overlay Git identity changed')
    files=subprocess.check_output(['git','-C',str(overlay_root),'ls-files','src/look'],text=True).splitlines()
    overlay_files={rel:file_sha256(overlay_root/rel) for rel in files}
    numerical={}
    for rel in NUMERICAL_MODULES:
        old=_git_blob_sha(overlay_root,SCIENTIFIC_SOURCE_COMMIT,rel);new=overlay_files[rel]
        if old!=new:raise ValueError('Management overlay changed numerical module: '+rel)
        numerical[rel]=dict(scientific_sha256=old,overlay_sha256=new,equal=True)
    return dict(schema='look_fusion_management_overlay_v2',task_id=TASK_ID,repair_packet_sha256=REPAIR_0207_SHA256,
        scientific_source_commit=SCIENTIFIC_SOURCE_COMMIT,management_commit=management_commit,
        overlay_root=str(overlay_root),overlay_files=overlay_files,numerical_modules=numerical,test_access=False)


def verify_management_overlay(path):
    path=Path(path);value=read(path)
    if (value.get('schema')!='look_fusion_management_overlay_v2' or value.get('task_id')!=TASK_ID
            or value.get('repair_packet_sha256')!=REPAIR_0207_SHA256
            or value.get('scientific_source_commit')!=SCIENTIFIC_SOURCE_COMMIT or value.get('test_access') is not False):
        raise ValueError('Fusion management overlay receipt changed')
    root=Path(value['overlay_root']).resolve()
    if subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()!=value['management_commit']:
        raise ValueError('Fusion management overlay commit changed')
    for rel,digest in value['overlay_files'].items():
        if file_sha256(root/rel)!=digest:raise ValueError('Fusion management overlay file changed: '+rel)
    if set(value.get('numerical_modules',{}))!=set(NUMERICAL_MODULES):raise ValueError('Fusion numerical overlay coverage changed')
    for rel,row in value['numerical_modules'].items():
        if row.get('equal') is not True or row.get('scientific_sha256')!=row.get('overlay_sha256'):
            raise ValueError('Fusion numerical overlay equivalence changed')
        if _git_blob_sha(root,SCIENTIFIC_SOURCE_COMMIT,rel)!=row['scientific_sha256']:
            raise ValueError('Fusion scientific numerical source changed')
    return value


def validate_repair_resume(plan,row,spec,runroot):
    marker=Path(runroot)/'repair_resume_0207.json'
    if not marker.exists():return None
    value=read(marker)
    pipeline=Path(runroot)/'pipeline_status.json'
    if not pipeline.exists() or read(pipeline).get('state')!='needs_review':
        raise ValueError('Fusion repair marker requires a needs_review pipeline')
    expected=dict(schema='look_fusion_stage_repair_resume_v2',task_id=TASK_ID,run_id=spec['run_id'],
        spec_sha256=file_sha256(row['spec']),prior_pipeline_status_sha256=file_sha256(pipeline),
        repair_packet_sha256=REPAIR_0207_SHA256,test_access=False)
    if {k:value.get(k) for k in expected}!=expected:
        raise ValueError('Fusion repair resume marker changed')
    sequence_overlay=verify_management_overlay(Path(plan['root'])/'management_overlay.json')
    run_overlay=verify_management_overlay(Path(runroot)/'management_overlay.json')
    if file_sha256(Path(plan['root'])/'management_overlay.json')!=file_sha256(Path(runroot)/'management_overlay.json'):
        raise ValueError('Fusion management overlay receipts differ between sequence and run')
    if sequence_overlay!=run_overlay:raise ValueError('Fusion management overlay contents differ')
    return value


def validate_sequence(plan):
    if (plan.get('schema')!='look_cohort_sequence_v1' or plan.get('study_kind')!=STUDY_KIND
            or plan.get('task_id')!=TASK_ID or plan.get('test_access') is not False):
        raise ValueError('Unregistered fusion-stage sequence')
    if plan.get('cross_host_comparisons')!=CROSS_HOST_COMPARISONS:
        raise ValueError('Fusion-stage comparison family changed')
    expected_structures={position:host_structure(position) for position in POSITIONS}
    if plan.get('host_structures')!=expected_structures:
        raise ValueError('Fusion-stage host structure contract changed')
    rows=plan.get('tasks',[])
    if len(rows)!=3 or [(row.get('role'),row.get('position')) for row in rows] != [
            ('accepted_reference','deep'),('execute','middle'),('execute','features')]:
        raise ValueError('Fusion-stage task order changed')
    specs=[read(row['spec']) for row in rows]
    for row,spec in zip(rows,specs):
        if file_sha256(row['spec'])!=row['spec_sha256']:
            raise ValueError('Fusion-stage spec changed')
        if row['position']!=spec['position']:
            raise ValueError('Fusion-stage position mismatch')
    deep_root=Path(specs[0]['output'])
    deep,_=_accepted_reference(deep_root)
    if specs[0]!=deep:
        raise ValueError('Fusion-stage deep reference is not the accepted spec')
    fixed=_fixed_science(deep)
    for spec in specs[1:]:
        validate_member(spec)
        if _fixed_science(spec)!=fixed:
            raise ValueError('Fusion-stage scientific fixed factors changed')
        if spec.get('fusion_stage_structure')!=plan['host_structures'][spec['position']]:
            raise ValueError('Fusion-stage member structure changed')
        if spec.get('source_commit')!=plan.get('source_commit'):
            raise ValueError('Fusion-stage source commit changed')
        for pin in spec.get('source_pins',[]):
            if file_sha256(pin['path'])!=pin['sha256']:
                raise ValueError('Fusion-stage source pin changed')
    if len({s['run_id'] for s in specs})!=3 or len({s['output'] for s in specs})!=3:
        raise ValueError('Fusion-stage duplicate run')
    return specs


def _source_pins(source_root,reference_spec):
    source_root=Path(source_root)
    if subprocess.check_output(['git','-C',str(source_root),'status','--porcelain'],text=True).strip():
        raise ValueError('Fusion-stage source snapshot is dirty')
    files=subprocess.check_output(['git','-C',str(source_root),'ls-files','src/look'],text=True).splitlines()
    pins=[dict(path=str(source_root/rel),sha256=file_sha256(source_root/rel)) for rel in files]
    framework=[copy.deepcopy(row) for row in reference_spec['source_pins'] if '/site-packages/mhd_framework/' in row['path']]
    if len(framework)!=18:
        raise ValueError('Unexpected MHD pin set')
    for row in framework:
        if file_sha256(row['path'])!=row['sha256']:
            raise ValueError('MHD pin changed')
    return framework+pins


def prepare(reference_root,output,run_root,publication,sequence_id,source_root,source_commit,middle_run_id,features_run_id):
    reference_root=Path(reference_root)
    output=Path(output)
    run_root=Path(run_root)
    deep,_=_accepted_reference(reference_root)
    source_root=Path(source_root)
    if subprocess.check_output(['git','-C',str(source_root),'rev-parse','HEAD'],text=True).strip()!=source_commit:
        raise ValueError('Fusion-stage source commit mismatch')
    pins=_source_pins(source_root,deep)
    tasks=[dict(role='accepted_reference',position='deep',spec=str(reference_root/'spec.json'),
        spec_sha256=file_sha256(reference_root/'spec.json'),
        delivery_sha256=file_sha256(reference_root/'delivery/accepted.json'))]
    reference=dict(run_id=DEEP_RUN_ID,spec_sha256=DEEP_SPEC_SHA256,delivery_sha256=DEEP_DELIVERY_SHA256)
    structures={position:host_structure(position) for position in POSITIONS}
    for position,run_id in (('middle',middle_run_id),('features',features_run_id)):
        spec=copy.deepcopy(deep)
        spec.update(run_id=run_id,output=str(run_root/run_id),position=position,source_commit=source_commit,
            source_pins=pins,study_kind=STUDY_KIND,fusion_stage_task=TASK_ID,fusion_stage_reference=reference,
            fusion_stage_structure=structures[position],disk_reserve_bytes=50*1024**3)
        root=Path(spec['output'])
        root.mkdir(parents=True,exist_ok=False)
        atomic_write_json(spec,root/'spec.json')
        tasks.append(dict(role='execute',position=position,spec=str(root/'spec.json'),spec_sha256=file_sha256(root/'spec.json')))
    plan=dict(schema='look_cohort_sequence_v1',study_kind=STUDY_KIND,task_id=TASK_ID,test_access=False,
        sequence_id=sequence_id,source_commit=source_commit,tasks=tasks,root=str(output),publication=str(publication),
        cross_host_comparisons=CROSS_HOST_COMPARISONS,deep_reference=reference,host_structures=structures,
        conclusion_scope='single_seed_same_dev_fusion_stage_mechanism_not_external_A_B_C_coverage')
    output.mkdir(parents=True,exist_ok=False)
    atomic_write_json(plan,output/'plan.json')
    validate_sequence(plan)
    atomic_write_json(dict(schema='look_fusion_stage_preparation_v1',state='accepted',test_access=False,
        plan_sha256=file_sha256(output/'plan.json'),source_commit=source_commit,
        deep_reference=reference,new_runs=[middle_run_id,features_run_id]),output/'preparation_acceptance.json')
    return plan


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare')
    for name in ('reference_root','output','run_root','publication','sequence_id','source_root','source_commit','middle_run_id','features_run_id'):
        a.add_argument('--'+name.replace('_','-'),required=True)
    r=sub.add_parser('repair-receipt');r.add_argument('--overlay-root',required=True);r.add_argument('--management-commit',required=True);r.add_argument('--output',required=True)
    args=p.parse_args()
    if args.command=='repair-receipt':
        value=build_management_overlay_receipt(args.overlay_root,args.management_commit);atomic_write_json(value,args.output);print(json.dumps(value,sort_keys=True));return
    if args.command=='prepare':
        keys=('reference_root','output','run_root','publication','sequence_id','source_root','source_commit','middle_run_id','features_run_id')
        value=prepare(**{key:getattr(args,key) for key in keys})
        print(json.dumps(value,sort_keys=True))


if __name__=='__main__':
    main()
