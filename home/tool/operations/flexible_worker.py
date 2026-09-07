"""Execution adapter; all numerical imports resolve to the immutable worker release."""
import fcntl
import os
import time
from pathlib import Path
from look_core.dual_queue import (read, validate_plan, digest, check_source, atomic,
                                 validate_receipt, verify, record)

def worker(project_root, plan_path, output, job_id):
    root, output = Path(project_root).resolve(), Path(output).resolve()
    plan = read(plan_path); jobs = validate_plan(plan)
    job = next(j for j in jobs if j['id'] == job_id)
    identity = dict(plan_sha256=digest(plan), source_manifest_sha256=check_source(root))
    if read(output/'queue_identity.json') != identity:
        raise ValueError('Parent and worker identities differ')
    out = output/'cases'/job_id; out.mkdir(parents=True, exist_ok=True)
    with (out/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt_path = out/'queue_complete.json'
        if receipt_path.exists(): return validate_receipt(receipt_path, identity, job)
        physical = os.environ['LOOK_PHYSICAL_GPU']
        if not physical.isdecimal() or os.environ.get('CUDA_VISIBLE_DEVICES') != os.environ.get('LOOK_ASSIGNED_GPU_UUID'):
            raise ValueError('Worker must expose exactly its assigned physical GPU')
        import torch
        from look_core.bounded_runtime import install_runtime, BudgetRunner
        if torch.cuda.device_count() != 1:
            raise ValueError('An independent case must see exactly one CUDA device')
        from look_core import bounded_runtime
        def allocator_limits():
            # An execution budget, not a numerical/scientific configuration.
            gib = float(os.environ.get('LOOK_ALLOCATOR_GIB', '11'))
            if not 0 < gib <= 12: raise ValueError('Invalid allocator budget')
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(gib*1024**3/total, 1.), 0)
            torch.cuda.set_device(0)
        bounded_runtime.install_allocator_limits = allocator_limits
        install_runtime()
        from look_core.start_study import verify_source
        source = read(verify(job['source']))
        verify_source(source)
        if source['selection']['seed'] != job['seed']:
            raise ValueError('Source seed differs from explicit case')
        begun = time.time()
        if job['kind'] == 'acceptance':
            from look_core.dual_queue_acceptance import probe
            result_path = probe(source, out, job['seed'])
        elif job['kind'] == 'method':
            from look_core.method_study import make_method_cases, run_method_case
            from look_core.self_input_study import make_cases
            spec = job['spec']; case = job['case']
            legal = make_cases(spec) if spec['protocol'].startswith('self_input_') else make_method_cases(spec)
            if case not in legal or case['seed'] != job['seed']:
                raise ValueError('Method case is outside frozen specification')
            if (source['selection']['fusion_position'] != case['fusion_position']
                    or source['selection']['filling_strategy'] != case['filling']):
                raise ValueError('Method source fusion/filling mismatch')
            run_method_case(source, case, spec, out/'result', torch.device('cuda:0'), (0,))
            result_path = out/'result'/'validation_result.json'
        elif job['kind'] == 'selected':
            from look_core.start_study import audit_case, evaluate_selected
            rows = []
            for item in job['records']:
                if 'job_id' in item:
                    predecessor = next(j for j in jobs if j['id'] == item['job_id'])
                    receipt = validate_receipt(output/'cases'/item['job_id']/'queue_complete.json', identity, predecessor)
                    input_result_path = verify(receipt['result'])
                else:
                    input_result_path = verify(item['result'])
                rows.append(audit_case(item['case'], input_result_path))
            rows.sort(key=lambda r:r['start_ordinal'])
            if ([r['start_ordinal'] for r in rows] != list(range(1,10))
                    or any(r['context'] != job['context'] for r in rows)):
                raise ValueError('Selected context requires exactly its nine ordered starts')
            chosen = evaluate_selected(job['context'], rows, source, out/'result', torch.device('cuda:0'), (0,))
            result_path = Path(chosen['path'])
        else:
            from look_core.start_study import sites_for_fusion
            case = job['case']; sites = sites_for_fusion(source['selection']['fusion_position'])
            start = case['start_ordinal']
            if (not 1 <= start <= len(sites) or case['candidate_sites'] != sites
                    or case['eligible_sites'] != sites[start-1:] or case['allowed_start'] != sites[start-1]):
                raise ValueError('Illegal correction start')
            runner = BudgetRunner(source, case, out/'result', out/'cache', torch.device('cuda:0'), (0,))
            runner.run(); result_path = runner.experiment_dir/'validation_result.json'
        result = read(result_path)
        if result.get('status') != 'complete' or result.get('test_access') is not False:
            # Historical LOOK result records encode the sealed boundary as phase/test.
            if not (job['kind'] == 'suffix' and result.get('status') == 'complete'
                    and result.get('phase') == 'validation' and not result.get('test')):
                raise ValueError('Incomplete or unsealed result')
        # Pin all produced model/config/prediction artifacts, including nested suffix banks.
        artifacts = [record(p) for p in sorted((out/'result').rglob('*'))
                     if p.is_file() and p.suffix in ('.json', '.npz', '.pt')]
        artifacts.extend([record(result_path), job['source']])
        for key in ('result', 'manifest', 'checkpoint', 'pca_manifest', 'labels', 'natural_labels'):
            rec = source[key]; artifacts.append({k: rec[k] for k in ('path', 'bytes', 'sha256')})
        for rec in [*source['pcas'], *source['generators'].values()]:
            artifacts.append({k: rec[k] for k in ('path', 'bytes', 'sha256')})
        for rec in result.get('provenance', []):
            # Normalize records to the queue's exact path/size/hash schema.
            verify({k: rec[k] for k in ('path', 'bytes', 'sha256')})
            artifacts.append({k: rec[k] for k in ('path', 'bytes', 'sha256')})
        value = dict(identity=identity, job_sha256=digest(job), status='complete', test_access=False,
                     result=record(result_path),
                     physical_gpu=int(physical), physical_uuid=os.environ['LOOK_ASSIGNED_GPU_UUID'],
                     visible_devices=1, seed=job['seed'],
                     started_at_unix=begun, ended_at_unix=time.time(), artifacts=artifacts,
                     execution_microbatch=int(os.environ['LOOK_EXECUTION_MICROBATCH']),
                     execution_adapter=record(__file__), allocator_gib=float(os.environ.get('LOOK_ALLOCATOR_GIB', '11')))
        atomic(value, out/'queue_complete.json')
        return value
