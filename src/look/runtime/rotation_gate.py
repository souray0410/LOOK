"""Fail-closed helpers for the existing Ibex project rotation control plane."""
from __future__ import annotations
import contextlib,fcntl,os,pathlib,re,stat

def gpu_count(gres: str) -> int:
    if not gres or gres in {'(null)','N/A'}: return 0
    counts=[]
    for item in gres.split(','):
        if 'gpu' not in item.lower(): continue
        match=re.search(r':(\d+)(?:\([^)]*\))?$',item)
        if not match: raise ValueError(f'Unparseable GPU GRES: {gres}')
        counts.append(int(match.group(1)))
    return sum(counts)

@contextlib.contextmanager
def existing_lock(path):
    path=pathlib.Path(path)
    fd=os.open(path,os.O_RDWR|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd); current=os.stat(path,follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or (before.st_dev,before.st_ino)!=(current.st_dev,current.st_ino):
            raise RuntimeError('Shared lock identity changed')
        fcntl.flock(fd,fcntl.LOCK_EX)
        current=os.stat(path,follow_symlinks=False)
        if (before.st_dev,before.st_ino)!=(current.st_dev,current.st_ino): raise RuntimeError('Shared lock replaced while waiting')
        yield fd
    finally: os.close(fd)

def account_usage(lines, ownership, projects):
    total=0;counts={p:0 for p in projects}
    for line in lines:
        job,gres=line.split('|',1);n=gpu_count(gres)
        if not n: continue
        owners=ownership.get(job,set())
        if len(owners)!=1: raise ValueError(f'GPU job {job} has ambiguous or absent journal ownership')
        owner=next(iter(owners))
        if owner not in counts: raise ValueError(f'Unknown project owner: {owner}')
        total+=n;counts[owner]+=n
    return total,counts

def validate_claim(claim,*,job,task,owner,start_batch=None):
    if claim.get('schema')!='look_rotation_claim_v1' or claim.get('state')!='submitted': raise ValueError('Submitted LOOK claim required')
    if str(claim.get('job_id'))!=str(job) or claim.get('task')!=task or claim.get('owner')!=owner: raise ValueError('Claim/job identity mismatch')
    if start_batch is not None and int(claim.get('start_batch',0))!=int(start_batch): raise ValueError('Claim cursor mismatch')
