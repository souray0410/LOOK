#!/usr/bin/env python3
"""Retire only model/experiment outputs of the user-replaced 2026_09_03_19_35_04 release."""
import argparse,json,os,shutil
from pathlib import Path
from look_core.joint_cleanup import assert_no_writers
from look_core.state import atomic_write_json,file_sha256,utc_now

OLD_RELEASE='2026_09_03_19_35_04'
NEW_RELEASE='2026_09_04_10_49_20'
TARGET_NAMES=('backbones','baseline_selection','experiments','joint_look','logs','overnight','pca','reviewed_look','smoke','sweeps','experiment_registry.json')


def inspect_tree(path,old):
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):raise ValueError(f'Symlink: {path}')
    if old not in path.parents:raise ValueError('Outside old release')
    if {'dataset','preprocessed_pairs','generators','primary_test','natural_test'} & set(path.relative_to(old).parts):raise ValueError('Protected data or test path')
    files=list(path.rglob('*')) if path.is_dir() else [path]
    total=count=0
    for p in files:
        if p.is_symlink():raise ValueError(f'Symlink: {p}')
        if p.name.startswith(('test_result','test__','primary_test','natural_test')):raise ValueError(f'Test output protected: {p}')
        if p.name=='stage.lock':
            pid=int(json.loads(p.read_text())['pid'])
            try:os.kill(pid,0)
            except ProcessLookupError:pass
            else:raise RuntimeError(f'Live lock: {p}')
        if p.is_file():count+=1;total+=p.stat().st_size
    return dict(path=str(path),files=count,bytes=total)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-root',type=Path,required=True);p.add_argument('--new-root',type=Path,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    old,new=a.old_root.absolute(),a.new_root.absolute()
    if old.name!=OLD_RELEASE or new.name!=NEW_RELEASE or old.parent!=new.parent:raise ValueError('Wrong release scope')
    audit=new/'runs/maintenance/retired_auroc_release.json'
    targets=[old/'runs'/n for n in TARGET_NAMES]+[old/'cache/partial',old/'cache/pipeline_state',old/'runs/maintenance/joint_smoke']
    targets=[x for x in targets if x.exists()]
    if not targets and audit.exists():print('Already retired',audit);return
    manifest=[inspect_tree(x,old) for x in targets]
    assert_no_writers(targets)
    protected={str(p):file_sha256(p) for p in (old/'dataset').rglob('*.json')}
    protected.update({str(p):file_sha256(p) for p in (old/'dataset').rglob('*.csv')})
    configs=[]
    for p in (old/'runs/backbones').glob('*/run_config.json'):
        payload=json.loads(p.read_text());configs.append(dict(path=str(p),sha256=file_sha256(p),seed=payload.get('seed'),architecture=payload.get('architecture_id'),primary_metric=payload.get('config',{}).get('primary_metric'),backbone_code=payload.get('backbone_implementation_sha256')))
    report=dict(status='planned',reason='User restarted under new timestamp with Macro-F1 checkpoint and LOOK selection; old model/results are not retained',old_release=OLD_RELEASE,new_release=NEW_RELEASE,paths=manifest,old_configuration_identities=configs,protected_dataset_manifest=protected,total_bytes=sum(x['bytes'] for x in manifest),test_access=False,preserves=['shared dataset/images/preprocessing/environment','small task-selection provenance','existing maintenance audits','Git source history'])
    if a.execute:
        proof=new/'runs/maintenance/macro_f1_training_verification.json'
        if not proof.exists() or json.loads(proof.read_text()).get('status')!='passed':raise RuntimeError('New Macro-F1 training verification required before retirement')
        report.update(status='deleting',started_at_utc=utc_now());atomic_write_json(report,audit)
        for target in targets:
            inspect_tree(target,old)
            if target.is_dir():shutil.rmtree(target)
            else:target.unlink()
        if not all(file_sha256(Path(p))==h for p,h in protected.items()):raise RuntimeError('Protected data hash changed')
        report.update(status='complete',completed_at_utc=utc_now(),all_targets_absent=all(not p.exists() for p in targets));atomic_write_json(report,audit)
    else:atomic_write_json(report,new/'runs/maintenance/retirement_preview.json')
    print(json.dumps({k:report[k] for k in ('status','total_bytes','paths')},indent=2))

if __name__=='__main__':main()
