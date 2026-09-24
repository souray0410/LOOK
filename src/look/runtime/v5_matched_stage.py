"""Continue the existing LOOK chain through its complete matched study.

The immutable chain module supplies the original paths, shared lock and quota
checks. No separate periodic scheduler or resource policy is introduced.
"""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time


def outcome(terminal, status, accepted):
    if not terminal or terminal['state'] not in {
            'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY'}:
        raise ValueError('Predecessor is not terminal')
    if (terminal['state'] == 'COMPLETED' and terminal['exit_code'] == '0:0'
            and accepted and accepted.get('state') == 'accepted'
            and accepted.get('test_access') is False):
        return 'accepted'
    if (terminal['state'] == 'FAILED' and terminal['exit_code'] == '75:0'
            and status.get('state') == 'paused'):
        return 'resume'
    # A lease timeout releases ownership; completed per-site artifacts are
    # checked by the same scientific executor before they can be reused.
    if terminal['state'] == 'TIMEOUT':
        return 'resume'
    return 'failed_closed'


def monitor(chain, mode, job, claim='none', delay=False):
    args=['sbatch','--parsable',f'--dependency=afterany:{job}',
        '--job-name=look_v5_'+mode+'_finalize','--account=pi-mengy',
        '--partition=batch','--time=00:20:00','--cpus-per-task=2','--mem=4G',
        f'--output={chain.PKG}/{mode}-finalize-{job}-%j.log',
        str(chain.PKG/'monitor.sh'),'--stage',mode,str(job),str(claim)]
    if delay:args.insert(2,'--begin=now+5minutes')
    return subprocess.check_output(args,text=True).strip().split(';')[0]


def main(mode, job, claim, chain):
    pca=chain.OUT.parent/'look_cataract_middle_v5_pca_sharded_v1'
    suite=chain.OUT.parent/'look_cataract_middle_v5_matched_suite_v1'
    event=chain.PKG/f'{mode}-terminal-{job}.json'
    with (chain.PKG/f'{mode}-terminal-{job}.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if event.exists() and chain.read(event).get('state') in ('accepted','suite_submitted','pca_resubmitted'):
            return chain.read(event)
        terminal=chain.terminal(job)
        root=pca if mode=='pca' else suite
        accepted=chain.read(root/'accepted.json') if (root/'accepted.json').exists() else None
        status=chain.read(root/'status.json') if (root/'status.json').exists() else {}
        action=outcome(terminal,status,accepted)
        if mode=='suite':
            c=chain.read(claim)
            if c.get('state')=='submitted':
                chain.validate_claim(c,job=job,task=chain.TASK,owner='mengh')
                chain.close_claim(claim,'completed' if action=='accepted' else 'released',terminal=terminal)
            elif (c.get('state') not in ('completed','released')
                  or str(c.get('job_id'))!=str(job) or c.get('terminal')!=terminal):
                raise ValueError('Suite predecessor claim changed')
        if action=='failed_closed':
            receipt=dict(state=action,terminal=terminal,test_access=False)
            chain.atomic(event,receipt);return receipt
        if mode=='suite' and action=='accepted':
            if (accepted.get('schema')!='look_formal_v5_matched_suite_receipt_v1'
                    or accepted.get('identity_sha256')!=chain.stable_hash(chain.read(suite/'identity.json'))):
                raise ValueError('Suite acceptance identity changed')
            for name,digest in accepted['files'].items():
                path=(suite/name).resolve()
                if not path.is_relative_to(suite.resolve()) or chain.sha(path)!=digest:
                    raise ValueError('Suite accepted evidence changed')
            receipt=dict(state='accepted',terminal=terminal,receipt=str(suite/'accepted.json'),
                receipt_sha256=chain.sha(suite/'accepted.json'),test_access=False)
            chain.atomic(event,receipt);return receipt
        if mode=='pca' and action=='resume':
            next_job=subprocess.check_output(['sbatch','--parsable',str(chain.PKG/'run_pca.sbatch')],text=True).strip().split(';')[0]
            mon=monitor(chain,'pca',next_job)
            receipt=dict(state='pca_resubmitted',job=next_job,monitor=mon,test_access=False)
            chain.atomic(event,receipt);return receipt
        with chain.existing_lock(chain.LOCK):
            policy,allgpu,look=chain.capacity()
            if look>=policy['projects']['LOOK']['reserved_gpus'] or allgpu>=policy['account_ceiling']:
                retry=monitor(chain,mode,job,claim,delay=True)
                receipt=dict(state='waiting_role_capacity',retry_job=retry,test_access=False)
                chain.atomic(event,receipt);return receipt
            cp=chain.PKG/'claims'/f'suite_after_{mode}_{job}.json';cp.parent.mkdir(exist_ok=True)
            if cp.exists():
                c=chain.read(cp)
                if c.get('state')!='submitted':
                    raise ValueError('Unfinished held submission requires reconciliation')
                next_job=c['job_id']
            else:
                c=dict(schema='look_rotation_claim_v1',task=chain.TASK,owner='mengh',
                    state='reserved_for_submission',created_at=time.time(),predecessor_job=job,test_access=False)
                chain.atomic(cp,c)
                next_job=subprocess.check_output(['sbatch','--hold','--parsable',
                    f'--export=ALL,LOOK_ROTATION_CLAIM={cp}',str(chain.PKG/'run_decision.sbatch')],text=True).strip().split(';')[0]
                c.update(state='submitted',job_id=next_job,submitted_at=time.time())
                chain.atomic(cp,c);os.chmod(cp,0o444)
            journal=chain.read(chain.JOURNAL)
            if not any(str(row.get('job_id'))==str(next_job) for row in journal['requests']):
                journal['requests'].append(dict(command=['sbatch','--hold',str(chain.PKG/'run_decision.sbatch')],
                    job_id=next_job,name='look_v5_matched_suite',state='submitted',time=time.time(),
                    claim=str(cp),task=chain.TASK))
                chain.atomic(chain.JOURNAL,journal)
            mon=monitor(chain,'suite',next_job,cp)
            subprocess.check_call(['scontrol','release',str(next_job)])
            receipt=dict(state='suite_submitted',job=next_job,monitor=mon,claim=str(cp),test_access=False)
            chain.atomic(event,receipt);return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('pca','suite'))
    parser.add_argument('job');parser.add_argument('claim');args=parser.parse_args()
    import chain_manager
    main(args.mode,args.job,args.claim,chain_manager)
