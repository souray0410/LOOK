"""One-time accepted native V4-to-V5 parent publication; never a runtime fallback."""
import argparse,fcntl,hashlib,json,os,shutil,tempfile
from pathlib import Path
import torch
from look.models.native_materialization import verify_selected
from look.runtime.state import file_sha256 as sha


def read(p):return json.loads(Path(p).read_text())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')


def validate_replay(source,conversion,replay,tool_sha):
    spec=read(source/'spec.json');original=read(source/'accepted.json');cpu=read(conversion/'cpu_accepted.json')
    if spec['test_used'] is not False or original['test_used'] is not False or original['status']!='accepted':raise ValueError('Accepted train/dev source required')
    if spec['framework']['api']!='V4' or spec['model']['name']!='convnext_base':raise ValueError('Exact registered source family required')
    for name,key in [('best.pt','source_best_sha256'),('spec.json','source_spec_sha256')]:
        if sha(source/name)!=original['files'][name] or sha(source/name)!=cpu[key]:raise ValueError('Source identity changed')
    state=conversion/'v5_selected/checkpoint.pt'
    if sha(state)!=cpu['target_checkpoint_sha256']:raise ValueError('Converted selected state changed')
    for api,status in [('V4','reference_recorded'),('V5','full_dev_and_three_updates_accepted')]:
        r=read(replay/api/'accepted.json')
        if (r.get('api')!=api or r.get('state')!=status or r.get('test_access') is not False or
            r.get('scientific_training_updates')!=0 or r.get('qualification_updates')!=3 or
            r.get('tool_sha256')!=tool_sha or r.get('source_best_sha256')!=cpu['source_best_sha256'] or
            r.get('target_checkpoint_sha256')!=cpu['target_checkpoint_sha256'] or
            r.get('hardware','').find('A100')<0 or
            r.get('development_participants')!=len(read(Path(spec['development_manifest']))['samples']) or
            sha(replay/api/'development.pt')!=r['development_sha256']):raise ValueError('Exact full-development A100 acceptance required')
    return spec,original,cpu


def publish(source,conversion,replay,output,tool_sha):
    source,conversion,replay,output=map(Path,(source,conversion,replay,output))
    spec,original,cpu=validate_replay(source,conversion,replay,tool_sha)
    from mhd_framework.models.artifacts import verify_runtime
    from mhd_framework.models import create_model
    from look.models.observed_participant import ObservedParticipantModel
    verify_runtime(cpu['framework'])
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if output.exists():
            verify_selected(output)
            m=read(output/'migration.json')
            if m['source_spec_sha256']!=sha(source/'spec.json') or m['source_best_sha256']!=sha(source/'best.pt') or m['qualification_tool_sha256']!=tool_sha:raise ValueError('Existing publication belongs to another migration')
            return output
        tmp=Path(tempfile.mkdtemp(prefix=output.name+'.',suffix='.partial',dir=output.parent))
        current={**spec,'framework':cpu['framework'],'migration_provenance':{
            'original_spec_sha256':sha(source/'spec.json'),'original_training_framework':spec['framework'],
            'original_training_identity':original['identity'],'execution_only_conversion':True}}
        identity=digest(current)
        payload=torch.load(conversion/'v5_selected/checkpoint.pt',map_location='cpu',weights_only=False)
        if payload.get('framework_api')!='V5':raise ValueError('Current V5 converted envelope required')
        model=ObservedParticipantModel(create_model(spec['model']));model.load_state_dict(payload['model'],strict=True)
        for key,value in model.state_dict().items():
            if value.dtype!=payload['model'][key].dtype or not torch.equal(value,payload['model'][key]):raise ValueError('State changed during strict consumer load')
        payload['identity']=identity;torch.save(payload,tmp/'best.pt');write(tmp/'spec.json',current)
        for name in ('history.json','development_predictions.npz'):
            if sha(source/name)!=original['files'][name]:raise ValueError('Historical evidence changed')
            shutil.copyfile(source/name,tmp/name)
        shutil.copyfile(source/'accepted.json',tmp/'original_accepted.json');shutil.copyfile(source/'spec.json',tmp/'original_spec.json')
        for api in ('V4','V5'):shutil.copyfile(replay/api/'accepted.json',tmp/(api+'_replay.json'))
        migration={'schema':'look_native_selected_framework_migration_v1','source_best_sha256':sha(source/'best.pt'),'source_spec_sha256':sha(source/'spec.json'),
            'converted_checkpoint_sha256':cpu['target_checkpoint_sha256'],'qualification_tool_sha256':tool_sha,'original_training_framework':spec['framework'],
            'execution_framework':cpu['framework'],'new_training':False,'complete_training_resume':False,'test_access':False}
        write(tmp/'migration.json',migration)
        files={p.name:sha(p) for p in tmp.iterdir() if p.is_file()}
        receipt={**original,'identity':identity,'files':files,'migration':migration,
            'status':'accepted','acceptance_scope':'selected_execution_conversion_with_original_training_history'}
        write(tmp/'source_accepted.json',receipt);files['source_accepted.json']=sha(tmp/'source_accepted.json')
        manifest={'schema':'look_selected_native_v2','test_access':False,'complete_training_resume':False,'complete_task_model':True,
                  'selected_prediction_replay':True,'files':files,'source':migration}
        write(tmp/'selected_artifact.json',manifest);verify_selected(tmp,current)
        validate_replay(source,conversion,replay,tool_sha)
        os.replace(tmp,output)
        return output


if __name__=='__main__':
    p=argparse.ArgumentParser();[p.add_argument('--'+n,required=True) for n in ('source','conversion','replay','output','qualification-tool-sha256')]
    a=p.parse_args();print(publish(a.source,a.conversion,a.replay,a.output,a.qualification_tool_sha256))
