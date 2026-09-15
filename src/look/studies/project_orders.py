"""Materialize matched selected parents and register the frozen project matrix."""
from pathlib import Path
from look.models.native_materialization import materialize_selected
from look.runtime.state import atomic_write_json,file_sha256,stable_hash
from look.studies.native_prerequisites import MODELS,DISEASES
from look.studies.autoresearch import read,immutable


def advance(config,groups,verify_native,reserve):
    project=config['project'];root=Path(project['output']);root.mkdir(parents=True,exist_ok=True)
    tasks=[];states={}
    gate=project['runtime_gate']
    if file_sha256(Path(gate['path']))!=gate['sha256'] or read(gate['path']).get('status')!='accepted':
        raise ValueError('Full workflow runtime gate is missing or changed')
    for disease in DISEASES:
        for architecture in MODELS:
            name=disease+'/'+architecture
            pair=[groups[name+'/'+track] for track in ('cfp_2d','oct_bscan_2d')]
            if not all(g.get('state') in ('waiting_project_adapter', 'waiting_replications')
                       and g.get('selected') for g in pair):
                states[name]='waiting_locked_paired_parents';continue
            selected={}
            for role,group in zip(('first','second'),pair):
                # A locked, accepted 3416 pair can proceed while replicas train.
                # The controller has verified the nomination; materialization
                # independently verifies each actual native artifact again.
                sources=[group['selected']['run_dir']]+[r['run_dir'] for r in group['replicas']
                    if r.get('state') == 'accepted' or group['state'] == 'waiting_project_adapter']
                selected[role]={}
                for source in sources:
                    spec=read(Path(source)/'spec.json');seed=spec['training']['seed']
                    destination=materialize_selected(source,root/'parents',spec,verify_native)
                    selected[role][seed]=dict(path=str(destination),manifest_sha256=file_sha256(destination/'selected_artifact.json'))
            ready_seeds=set(selected['first']) & set(selected['second'])
            for seed in (3416,3417,3418):
                if seed not in ready_seeds:continue
                for position in ('middle','deep','features'):
                    spec=dict(schema='look_project_case_v1',model={'name':architecture},disease=disease,seed=seed,position=position,
                        parents={role:selected[role][seed] for role in selected},training=project['training'],look=project['look'],
                        methods=['look','single_final','all_on','bias','affine','available_parent'],
                        bootstrap_iterations=10000,source_pins=config['source_pins'],test_access=False,
                        protocol_sha256=config['protocol']['sha256'],catalog_sha256=config['catalog']['sha256'])
                    key=name+f'/seed{seed}/{position}'
                    path=root/'specs'/(stable_hash(key)+'.json');path.parent.mkdir(parents=True,exist_ok=True);immutable(path,spec)
                    run=reserve(root,'look_expanded_host_'+config['catalog']['sha256'][:16],key,spec,
                        source={'protocol':config['protocol'],'native_groups':name},refresh_summary=False)
                    tasks.append(dict(id=key,spec=str(path),spec_sha256=file_sha256(path),run_dir=str(run),role='look_project'))
            states[name]=('project_tasks_registered' if ready_seeds == {3416,3417,3418}
                          else 'pilot_registered_waiting_paired_replications')
    # Prioritize ready pilots across groups without changing task identities.
    tasks.sort(key=lambda t:(read(t['spec'])['seed'], t['id']))
    queue=dict(schema='look_project_work_feed_v1',test_access=False,tasks=tasks,groups=states)
    atomic_write_json(queue,root/'queue.json')
    from look.analysis.project_rollup import summarize
    progress=summarize(tasks,root/'report')
    return dict(queue=str(root/'queue.json'),tasks=len(tasks),groups=states,accepted=progress['accepted'],complete=progress['complete'])
