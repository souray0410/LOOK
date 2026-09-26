"""Build a pinned full-cohort method spec after current-parent acceptance."""
import argparse,json
from pathlib import Path
from look.runtime.state import atomic_write_json,file_sha256
from look.studies.modern_embracenet import SCHEMA,TRAINING,LOOK,RECIPE,validate

def prepare(receipt,source,commit,output,lock_root):
    receipt=Path(receipt).resolve(); source=Path(source).resolve(); output=Path(output).resolve()
    accepted=json.loads(receipt.read_text())
    if accepted.get('state')!='current_assets_and_cpu_consumer_accepted' or accepted.get('test_access') is not False:
        raise ValueError('Current parent consumer acceptance required')
    parents=accepted['assets']
    if len(parents)!=2 or any(p.get('execution_version')!='V5' for p in parents):raise ValueError('Two V5 parents required')
    import mhd_framework,mhd_models
    pins=[]
    for root in (source/'src/look',Path(mhd_framework.__file__).resolve().parent,Path(mhd_models.__file__).resolve().parent):
        pins += [{'path':str(p),'sha256':file_sha256(p)} for p in sorted(root.rglob('*.py'))]
    spec=dict(schema=SCHEMA,task_id='modern-convnext-embracenet-20260926',run_id='modern-convnext-embracenet-3416',
        architecture='convnext_base',seed=3416,embracement_size=256,test_access=False,training=TRAINING,look=LOOK,recipe=RECIPE,
        cohort={'train':58403,'development':12510},parents=[{'path':p['path'],'manifest_sha256':p['manifest_sha256']} for p in parents],
        parent_acceptance={'path':str(receipt),'sha256':file_sha256(receipt)},source_commit=commit,source_root=str(source),source_pins=pins,
        framework_commit='1287681c08846e11364c81653048435482e772a7',
        bootstrap={'iterations':10000,'seed':3416,'metrics':['macro_f1','macro_auroc_ovr','negative_log_likelihood','multiclass_brier']},
        lease_safety_seconds=900,devices=[0],lock_root=str(Path(lock_root).resolve()),gpu_budget_bytes=80*1024**3,gpu_reserve_bytes=0,
        ram_budget_bytes=128*1024**3,workspace_bytes=2*1024**3,host_free_fraction_min=.15,disk_reserve_bytes=10*1024**3,
        storage={'policy':'ibex_registered_execution_root','physical_output':str(output)},output=str(output))
    validate(spec)
    output.mkdir(parents=True,exist_ok=True);path=output/'spec.json'
    if path.exists():
        if json.loads(path.read_text())!=spec:raise ValueError('Existing scientific identity differs')
    else:atomic_write_json(spec,path)
    return path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parents',required=True);p.add_argument('--source',required=True);p.add_argument('--commit',required=True);p.add_argument('--output',required=True);p.add_argument('--lock-root',required=True);a=p.parse_args()
    print(prepare(a.parents,a.source,a.commit,a.output,a.lock_root))
