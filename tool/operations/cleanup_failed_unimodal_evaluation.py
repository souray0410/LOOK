#!/usr/bin/env python3
"""Remove only the failed first unimodal evaluation attempt; retain verified training."""
import argparse
import json
import os
from pathlib import Path
import shutil
from look_core.cli import add_runtime_arguments,resolve_runtime_arguments
from look_core.state import atomic_write_json,file_sha256,utc_now


def main():
    parser=argparse.ArgumentParser(description=__doc__);add_runtime_arguments(parser)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args();paths=resolve_runtime_arguments(args)
    root=paths.data_root
    if root.name!='2026_09_04_19_18_07':raise RuntimeError('Wrong release for this narrowly scoped cleanup')
    study=root/'runs/unified_study/042ba0e49137'
    summary=json.loads((study/'summary.json').read_text())
    assert summary['status']=='failed' and summary['completed_stages']=={} and summary['backbones']=={}
    assert summary['active_fusion']=='oct_only' and summary['active_seed']==3407 and summary['test_access'] is False
    assert summary['implementation_sha256']=='4f6c474784c960f0971ab4e9a8784c9cec32c9455b326d3b6b0a6c6bdfadb860'
    assert 'feature_message' in summary['error']
    sweep=root/'runs/sweeps/validation__2b40273ebcc9'
    plan=json.loads((sweep/'study_plan.json').read_text())
    assert plan['phase']=='validation' and plan['grid']['fusion_positions']==['oct_only'] and plan['grid']['seeds']==[3407]
    assert len(plan['experiment_ids'])==1
    eid=plan['experiment_ids'][0]
    assert eid=='resnet50_oct_only__normalized_mean__seed3407__clf-unified-macro-f1__gan-not_applicable__look-unified_backbone__0d2c14235a1e'
    experiment=root/'runs/experiments'/eid
    assert not (experiment/'validation_result.json').exists()
    state=root/'cache/pipeline_state/unified_study'
    global_state=json.loads((state/'state.json').read_text())
    assert global_state['status']=='failed' and global_state['config_hash'].startswith('042ba0e49137')
    targets=[experiment,sweep,study,state,root/'cache/pipeline_state'/f'experiment__{eid}__validation']
    files=[]
    for target in targets:
        assert target.is_dir() and target.resolve()==target and root in target.parents
        assert not {'backbones','dataset','generators','pca','freezes','test'}.intersection(target.relative_to(root).parts)
        for p in [target,*target.rglob('*')]:
            if p.is_symlink():raise RuntimeError(f'Refusing symlink: {p}')
            if p.is_file():
                assert p.suffix=='.json' and p.name!='validation_result.json',str(p)
                files.append(dict(path=str(p),bytes=p.stat().st_size,sha256=file_sha256(p)))
    for registry in (root/'runs').glob('*registry*'):
        if registry.is_file() and eid in registry.read_text():raise RuntimeError('Unexpected registry record; requires explicit scoped review')
    writers=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            if proc.stat().st_uid!=os.getuid():continue
            for fd in (proc/'fd').iterdir():
                try:path=Path(os.readlink(fd).removesuffix(' (deleted)'))
                except (FileNotFoundError,PermissionError,OSError):continue
                if any(path==target or target in path.parents for target in targets):writers.append(dict(pid=int(proc.name),path=str(path)))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    if writers:raise RuntimeError(f'Open file holders remain: {writers}')
    backbone=root/'runs/backbones'/summary['active_backbone_id']
    completion=json.loads((backbone/'training_complete.json').read_text())
    checkpoint=backbone/'best.pt';before=file_sha256(checkpoint)
    assert completion['status']=='complete' and before==completion['checkpoint_sha256']
    audit=dict(status='planned',reason='Post-training evaluation failed because the shared input reset accessed an absent unimodal node; no valid evaluation result was produced.',created_at_utc=utc_now(),test_access=False,
        old_study=summary,old_plan=plan,targets=list(map(str,targets)),files=files,total_bytes=sum(p['bytes'] for p in files),open_file_holders=writers,registry_records_removed=0,
        preserved_checkpoint=str(checkpoint),preserved_checkpoint_sha256=before,script_sha256=file_sha256(Path(__file__)))
    if args.execute:
        destination=root/'runs/maintenance/failed_unimodal_evaluation_cleanup.json'
        if destination.exists():raise RuntimeError('Cleanup audit already exists; do not overwrite it')
        atomic_write_json(audit,destination)
        for target in targets:shutil.rmtree(target)
        assert file_sha256(checkpoint)==before
        audit.update(status='complete',completed_at_utc=utc_now(),checkpoint_sha256_after=before)
        atomic_write_json(audit,destination)
    print(json.dumps(dict(status=audit['status'],targets=audit['targets'],total_bytes=audit['total_bytes'],preserved_checkpoint_sha256=before),indent=2))

if __name__=='__main__':main()
