"""Resume one pinned search on an existing lease; never allocate or select work."""
import argparse
from contextlib import ExitStack
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_text())


def recoverable(record, present, job_state):
    """Only proved lease loss or a recorded clean pause permits continuation."""
    if not record or not record.get('step'):
        raise ValueError('Missing previous launch identity requires review')
    alive = present(str(record['job_id']), str(record['step']))
    if alive is not False:
        return False
    if record['state'] == 'paused':
        return True
    if record['state'] not in ('running', 'claimed', 'liveness_needs_review', 'failed'):
        raise ValueError('Claim state requires review')
    return job_state(str(record['job_id'])) in ('TIMEOUT', 'PREEMPTED', 'NODE_FAIL')


def allocation_state(job):
    rows = subprocess.check_output(['sacct', '-X', '-j', job, '--noheader', '--parsable2',
                                    '--format=JobIDRaw,State'], text=True, timeout=30)
    states = [r.split('|')[1].split()[0].rstrip('+') for r in rows.splitlines()
              if r.split('|')[0] == job]
    return states[0] if len(states) == 1 else 'UNKNOWN'


def verify_pins(binding, sha):
    for path, digest in binding['pins'].items():
        if sha(path) != digest:
            raise ValueError('Pinned continuation input changed: '+path)
    config = read(binding['config'])
    if sha(config['task']['spec']) != config['task']['spec_sha256']:
        raise ValueError('Scientific specification changed')
    for key, value in {'MALLOC_ARENA_MAX':'1', 'MALLOC_TRIM_THRESHOLD_':'131072',
                       'MALLOC_MMAP_THRESHOLD_':'131072'}.items():
        if config['env'].get(key) != value:
            raise ValueError('Accepted allocator environment changed')
    return config


def publish(config, output):
    from look.runtime.state import atomic_write_json, file_sha256
    from look.studies.search_case import verify_case
    from look.analysis.cumulative_delivery import collect, publish as cumulative
    # One registered configuration; never invoke the matched-package report.
    sequence = Path(output)/'sequence.json'
    value = dict(tasks=[dict(config=config, sha256=file_sha256(config))])
    if sequence.exists() and read(sequence) != value:
        raise ValueError('Cumulative registration changed')
    if not sequence.exists():atomic_write_json(value, sequence)
    return cumulative(collect(sequence, verify_case), Path(output)/'publication')


def execute(binding, job, output, observe=False):
    config_path = binding['config']
    bootstrap = read(config_path)
    import sys
    sys.path[:0] = bootstrap['env']['PYTHONPATH'].split(':')
    from mhd_models.scheduling.policy import Claims
    from mhd_models.scheduling.slurm_liveness import step_presence
    from look.runtime.state import atomic_write_json as write, file_sha256
    from look.studies.search_case import dependencies, verify_case
    from look.analysis.search_delivery import report
    config = verify_pins(binding, file_sha256)
    run = Path(config['task']['run_dir']); out = Path(output); out.mkdir(parents=True, exist_ok=True)
    claims = Claims(config['claims'])
    if str(out.resolve()) != str(Path(binding['output']).resolve()):
        raise ValueError('Continuation output is part of the binding')
    with (out/('observer.lock' if observe else 'continuation.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps(dict(state='already_owned',lock=str(out/'continuation.lock'))),flush=True)
            return 0
        if observe:
            # Existing manager is the sole per-run report writer. This watcher
            # publishes only its accepted delivery and never acquires a claim.
            while True:
                config = verify_pins(binding, file_sha256)
                result = publish(config_path, out)
                write(dict(state='observing_delivery',publication=result,time=time.time()),out/'publication_status.json')
                if (run/'delivery/accepted.json').exists():
                    write(dict(state='accepted_delivery',publication=result,time=time.time()),out/'publication_status.json')
                    return 0
                time.sleep(60)
        record = read(claims.path(run))
        if record.get('spec_sha256') != config['task']['spec_sha256']:
            raise ValueError('Claim specification mismatch')
        if record.get('step') and step_presence(str(record['job_id']),str(record['step'])) is not False:
            write(dict(state='already_running_or_liveness_unknown',claim=record,time=time.time()),out/'status.json')
            return 0
        # Exclude the previous manager even after its scientific step exits.
        manager_dirs = [Path(config['manager'])] + list((out/'attempts').glob('*/manager'))
        matches = [m for m in manager_dirs if (m/'launch.json').exists()
                   and all(read(m/'launch.json').get(k) == record.get(k) for k in ('owner','generation'))]
        if len(matches) != 1:raise ValueError('Previous manager identity is ambiguous')
        with ExitStack() as guards:
            manager_lock = guards.enter_context((matches[0]/'manager.lock').open('a'))
            try:
                fcntl.flock(manager_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                run_lock = guards.enter_context((run/'run.lock').open('a'))
                fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(json.dumps(dict(state='already_owned',run=str(run))),flush=True)
                return 0
            if (run/'accepted.json').exists():
                verify_case(run,read(config['task']['spec']));report(run)
                publish(config_path,out)
                write(dict(state='accepted_delivery',time=time.time()),out/'status.json')
                return 0
            if not recoverable(record,step_presence,allocation_state):
                write(dict(state='waiting_verified_lease_loss_or_pause',claim=record,time=time.time()),out/'status.json')
                return 0
            dependencies(read(config['task']['spec']))
            attempt = out/'attempts'/str(time.time_ns());attempt.mkdir(parents=True)
            write(dict(previous_claim=record,old_step_dead=True,old_allocation_state=allocation_state(str(record['job_id'])),
                       run=str(run),new_job=job,time=time.time()),attempt/'recovery.json')
            def reconcile(current):
                if current != record:raise ValueError('Claim changed during recovery')
                return dict(current,state='paused',updated_at=time.time(),lease_recovery=str(attempt/'recovery.json'))
            pause = run/'pause.json'
            if pause.exists():
                reason=read(pause).get('reason')
                if reason not in ('expiry_or_requested','allocation_expiry'):
                    raise ValueError('Non-lease pause requires explicit review')
            claims.mutate(run,reconcile)
            if pause.exists():pause.rename(attempt/'previous_pause.json')
            renewed=copy.deepcopy(config);renewed['job']=str(job);renewed['manager']=str(attempt/'manager')
            write(renewed,attempt/'config.json')
        # Original accepted finite manager performs admission, claim acquisition,
        # lease checkpointing, scientific verification and per-run delivery.
        with (attempt/'manager.log').open('x') as log:
            result=subprocess.run([config['python'],binding['manager_program'],str(attempt/'config.json')],
                                  env=config['env'],stdout=log,stderr=subprocess.STDOUT)
        write(dict(state='manager_exited',returncode=result.returncode,attempt=str(attempt),time=time.time()),out/'status.json')
        if result.returncode == 0:publish(config_path,out)
        return result.returncode


def main():
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--job')
    p.add_argument('--output',required=True);p.add_argument('--observe',action='store_true');a=p.parse_args()
    if not a.observe and not a.job:p.error('--job required for continuation')
    raise SystemExit(execute(read(a.binding),a.job,a.output,a.observe))


if __name__=='__main__':main()
