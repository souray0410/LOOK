"""Two immutable priority owners share one GPU-owner step via exec only."""
import os
from pathlib import Path
import pytest

from look.runtime import priority_owner as owner


class FinalOwnerReached(Exception):
    pass


@pytest.fixture
def chain(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '51919716')
    monkeypatch.setenv('SLURM_STEP_ID', '42')
    monkeypatch.setenv('ORIGINAL_ENV', 'unchanged')
    configs={}
    for version in ('v5','v4'):
        pin=tmp_path/(version+'.pin');pin.write_text(version)
        configs[version]=dict(output=str(tmp_path/version),
            fallback_pins={str(pin):owner.sha(pin)},
            fallback_environment={'PYTHONPATH':version+'_original_path'},
            fallback_command=['python','old_wrapper.py','--config','v4','--gpu-owner'] if version=='v5'
                             else ['python','original_owner.py','--gpu-owner'])
    calls=[];priorities=[];states={'v5':('handover',0),'v4':('handover',0)}
    def priority(config, job):
        version=next(k for k,v in configs.items() if v is config)
        priorities.append((version, job))
        state,rc=states[version]
        return dict(state=state,returncode=rc)
    monkeypatch.setattr(owner,'priority',priority)
    monkeypatch.setattr(owner.subprocess,'run',lambda *a,**k:pytest.fail('Unexpected subprocess/srun'))
    def execute(program, command, environment):
        calls.append((program,list(command),dict(environment)))
        if command==configs['v5']['fallback_command']:
            # exec replaces the process environment; emulate that exact boundary.
            with monkeypatch.context() as scoped:
                for key in list(os.environ):
                    if key not in environment:scoped.delenv(key)
                for key,value in environment.items():scoped.setenv(key,value)
                code=owner.gpu_owner(configs['v4'])
            raise SystemExit(code)
        assert command==configs['v4']['fallback_command']
        raise FinalOwnerReached()
    monkeypatch.setattr(owner.os,'execvpe',execute)
    return configs,states,calls,priorities


def test_same_step_two_exec_hops_preserve_each_fallback_environment(chain):
    configs,states,calls,priorities=chain
    with pytest.raises(FinalOwnerReached):owner.gpu_owner(configs['v5'])
    assert priorities==[('v5','51919716'),('v4','51919716')]
    assert len(calls)==2
    for version,(_,command,environment) in zip(('v5','v4'),calls):
        assert command==configs[version]['fallback_command']
        assert '--allocation-owner' not in command
        assert environment['PYTHONPATH']==version+'_original_path'
        assert environment['ORIGINAL_ENV']=='unchanged'
        assert environment['SLURM_JOB_ID']=='51919716'
        assert environment['SLURM_STEP_ID']=='42'


def test_outer_clean_pause_retires_without_inner_owner(chain):
    configs,states,calls,priorities=chain
    states['v5']=('retire_lease',75)
    assert owner.gpu_owner(configs['v5'])==75
    assert calls==[] and priorities==[('v5','51919716')]


@pytest.mark.parametrize('state',['capacity_skipped','owned_elsewhere','quarantined'])
def test_outer_nonfatal_skip_preserves_inner_chain(chain,state):
    configs,states,calls,priorities=chain
    states['v5']=(state,0)
    with pytest.raises(FinalOwnerReached):owner.gpu_owner(configs['v5'])
    assert len(calls)==2 and len(priorities)==2


def test_inner_clean_pause_retires_without_original_owner(chain):
    configs,states,calls,priorities=chain
    states['v4']=('retire_lease',75)
    with pytest.raises(SystemExit) as stop:owner.gpu_owner(configs['v5'])
    assert stop.value.code==75
    assert len(calls)==1 and len(priorities)==2


@pytest.mark.parametrize('version,expected_hops',[('v5',0),('v4',1)])
def test_corrupt_fallback_pin_rejects_before_that_exec(chain,version,expected_hops):
    configs,states,calls,priorities=chain
    # A quarantined priority must still verify its independent fallback pins.
    states[version]=('quarantined',0)
    Path(next(iter(configs[version]['fallback_pins']))).write_text('changed')
    with pytest.raises(ValueError,match='Fallback pin changed'):
        owner.gpu_owner(configs['v5'])
    assert len(calls)==expected_hops


def test_outer_allocation_starts_exactly_one_gpu_owner_step(tmp_path,monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID','51919716')
    monkeypatch.setattr(owner.subprocess,'check_output',lambda *a,**k:
        'JobId=51919716 EndTime=2026-09-20T20:00:00 AllocTRES=cpu=8,mem=128G,gres/gpu=1')
    calls=[]
    def run(command,env):
        calls.append((command,env))
        return type('Result',(),{'returncode':75})()
    monkeypatch.setattr(owner.subprocess,'run',run)
    config=dict(fallback_pins={},python='python',wrapper_program='immutable_v4_wrapper.py')
    assert owner.allocation_owner(config,tmp_path/'v5.json')==75
    assert len(calls)==1
    command,environment=calls[0]
    assert command[0]=='srun' and command.count('--gpu-owner')==1
    assert '--allocation-owner' not in command
    assert command[-3:]==['--config',str(tmp_path/'v5.json'),'--gpu-owner']
    assert environment['SLURM_JOB_ID']=='51919716'
    assert environment['LOOK_ALLOCATION_MEMORY_GIB']=='128.0'


@pytest.mark.parametrize('held_version',['v5','v4'])
def test_actual_priority_locks_are_independent(tmp_path,held_version):
    import fcntl
    import json
    from types import SimpleNamespace
    configs={}
    for version in ('v5','v4'):
        folder=tmp_path/version;folder.mkdir()
        binding=folder/'binding.json';binding.write_text(json.dumps(dict(output=str(folder/'continuation'))))
        configs[version]=dict(output=str(folder),pins={str(binding):owner.sha(binding)},
            continuation_binding=str(binding),continuation_binding_sha256=owner.sha(binding),
            python='python',admission_program='admit.py',continuation_program='resume.py',
            continuation_environment={})
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)
    with (Path(configs[held_version]['output'])/'priority.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert owner.priority(configs[held_version],'51919716',run=run)['state']=='owned_elsewhere'
        assert calls==[]
        other='v4' if held_version=='v5' else 'v5'
        assert owner.priority(configs[other],'51919716',run=run)['state']=='handover'
        assert len(calls)==2  # Admission and continuation only, never another owner.
