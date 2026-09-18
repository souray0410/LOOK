"""Finite immutable family registration; never submits or competes for GPUs."""
import argparse
import fcntl
from pathlib import Path
from look.runtime.state import atomic_write_json, file_sha256, stable_hash
from look.studies.project_case import read
from look.studies.family_search_protocol import VERSION, ARMS, protocol, validate


def register(config):
    """Register four independent cases in an existing deployment-owned run root.

    Config explicitly supplies one accepted reference_search spec, family source
    pins, workspace_bytes and four run_dirs; this creates no GPU claim. The
    existing owner must perform full resource/profile admission before execute.
    """
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    with (out/'registry.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        registration=out/'registration.json'
        identity=dict(config=config,schema=VERSION)
        if registration.exists() and read(registration)!=identity:raise ValueError('Immutable family registration changed')
        ref=read(config['reference_search_spec'])
        if file_sha256(config['reference_search_spec'])!=config['reference_search_sha256']:
            raise ValueError('Reference search spec changed')
        if set(config['run_dirs'])!=set(ARMS) or len(set(config['run_dirs'].values()))!=len(ARMS):
            raise ValueError('Four distinct explicit run directories required')
        prepared=[]
        for arm in ARMS:
            spec=dict(schema=VERSION,protocol=protocol(),test_access=False,
                reference_search=ref,host=ref['host'],arm=arm,candidates=[dict(rank=32,ridge_lambda=None)],
                penalty_policy='prefix_train_pca_gcv',
                candidate_provenance='Existing operator._gcv_lambda on same-prefix train projected residual statistics; continuous log bounds [-13.8,4.6]; not independent RRR tuning',
                source_pins=config['source_pins'],workspace_bytes=config['workspace_bytes'])
            validate(spec)
            run=Path(config['run_dirs'][arm]).resolve();sp=out/'specs'/(arm+'.json')
            if (run/'spec.json').exists() and read(run/'spec.json')!=spec:
                raise ValueError('Run directory has a different scientific identity')
            if sp.exists() and read(sp)!=spec:raise ValueError('Immutable family registration changed')
            prepared.append((arm,spec,run,sp))
        atomic_write_json(identity,registration)
        tasks=[]
        for arm,spec,run,sp in prepared:
            atomic_write_json(spec,sp)
            tasks.append(dict(id='family_search/'+arm+'/'+stable_hash(spec),spec=str(sp.resolve()),
                spec_sha256=file_sha256(sp),run_dir=str(run),execution='look_family_search',
                arm=arm,search_mode='positive_forward_tree',test_access=False,
                command=['python','-m','look.studies.family_search_case','--spec',str(sp.resolve()),'--output',str(run)]))
        feed=dict(schema='look_family_search_feed_v1',tasks=tasks,test_access=False,
            state='registered_not_dispatched',replication_released=False)
        # Exclude polling time: identical registration is byte-identical.
        atomic_write_json(feed,out/'feed.json')
        return feed


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    register(read(a.config))

if __name__=='__main__':main()
