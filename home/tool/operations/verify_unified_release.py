#!/usr/bin/env python3
"""Run the bounded acceptance gates; never launch the formal queue."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from look_core.cli import add_runtime_arguments, resolve_runtime_arguments
from look_core.reproducibility import implementation_sha256
from look_core.state import atomic_write_json, file_sha256, utc_now


def main():
    parser=argparse.ArgumentParser(description=__doc__);add_runtime_arguments(parser);args=parser.parse_args()
    paths=resolve_runtime_arguments(args);project=paths.project_root
    identity=implementation_sha256(project)
    output=paths.runs_root/'maintenance'/'unified_acceptance'/identity[:12];output.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(project/'tool'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',NCCL_P2P_DISABLE=os.environ.get('NCCL_P2P_DISABLE','1'))
    report=dict(status='running',implementation_sha256=identity,test_access=False,started_at_utc=utc_now(),gates={})
    def run(name,arguments):
        log=output/f'{name}.log'
        with log.open('w') as stream:
            process=subprocess.run([sys.executable,*map(str,arguments)],cwd='/tmp',env=env,stdout=stream,stderr=subprocess.STDOUT)
        report['gates'][name]=dict(returncode=process.returncode,log=str(log),log_sha256=file_sha256(log))
        atomic_write_json(report,output/'verification.json')
        print(f'gate={name} returncode={process.returncode}',flush=True)
        if process.returncode:
            report['status']='failed';atomic_write_json(report,output/'verification.json')
            raise RuntimeError(f'{name} failed: {log}')
        return log
    common=['--project-root',project]
    run('unit',['-m','pytest',project/'tool/tests','-q','--disable-warnings'])
    run('topology',[project/'pipeline/25_run_graph_smoke.py',*common,'--gpus','0'])
    run('qualification',[project/'tool/operations/verify_unified_qualification.py',*common])
    run('training',[project/'tool/operations/verify_macro_f1_training.py',*common])
    run('joint',[project/'tool/operations/verify_macro_f1_joint.py',*common])
    run('pipeline',[project/'pipeline/26_run_pipeline_smoke.py',*common,'--gpus','0,1'])
    def artifact_hashes():
        root=paths.runs_root/'smoke'/'pipeline'
        return {str(p):file_sha256(p) for p in root.rglob('*') if p.is_file() and (p.suffix=='.npz' or p.name=='best.pt')}
    before=artifact_hashes()
    run('pipeline_resume',[project/'pipeline/26_run_pipeline_smoke.py',*common,'--gpus','0,1'])
    if not before or before!=artifact_hashes():raise RuntimeError('Resume changed completed weights or predictions')
    report['resume_artifacts_unchanged']=len(before)
    run('ddp_criteria',['-m','torch.distributed.run','--standalone','--nproc-per-node','2',project/'tool/tests/ddp_criteria_smoke.py','--output',output/'ddp_criteria'])
    command=[project/'pipeline/38_run_unified_study.py',*common,'--gpus','0,1']
    first=json.loads(run('plan_first',command).read_text());second=json.loads(run('plan_second',command).read_text())
    if (first!=second or first['expected_screening_backbones']!=7 or
            first['expected_formal_fusion_backbones']!=9 or first['expected_backbones']!=19 or
            first['expected_main_look_cases']!=27 or first['expected_total_stages']!=52):
        raise RuntimeError('Study planning is not deterministic')
    report['planned_study']=first
    report.update(status='passed',completed_at_utc=utc_now())
    atomic_write_json(report,output/'verification.json');print(json.dumps(dict(status='passed',report=str(output/'verification.json')),indent=2))

if __name__=='__main__':main()
