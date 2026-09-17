"""Matched search policies with the unchanged original PCA/GCV operator."""
from pathlib import Path
import time
import numpy as np
from look.methods.operator import fit_look_node, LOOKArtifact, save_selected_bank
from look.methods.independent_greedy import fit_trajectory, SelectionPaused
from look.runtime.state import atomic_write_json, stable_hash, file_sha256
from look.evaluation.evaluator import evaluate_missing, save_prediction_bundle


def fit_search(graph, train_loader, dev_loader, pattern, sites, factors, latent_dims,
               max_rank, device, output, pca_bank, *, identity, mode, should_pause=lambda:False,
               cache_identity=None, reference_root=None):
    if mode not in ('greedy','best_forward','positive_forward_tree'):raise ValueError('Unregistered search policy')
    if getattr(train_loader.dataset,'split',None)!='train':raise ValueError('Train-only fitting')
    if getattr(dev_loader.dataset,'split',None) not in ('train','development'):
        raise ValueError('Test sealed; train-only profile or development required')
    root=Path(output);root.mkdir(parents=True,exist_ok=True);records=[];hist=[];best=None;winner=None
    for factor in factors:
        folder=root/'factors'/f'x{factor}';basis={}
        for site in sites:
            matches=[b for (n,f),b in pca_bank.items() if n==site and f==(1 if len(b.feature_shape)==1 else factor)]
            if len(matches)!=1:raise ValueError('Missing unique basis')
            basis[site]=matches[0]
        from look.methods.shared_latent import SharedLatentFitter
        from look.methods.linear_operator import fingerprint
        fitter=SharedLatentFitter(graph,train_loader,basis,max_rank,device,folder/'latent_moments',
            reference_root or root/'reference_latents',cache_identity or identity,should_pause)
        ready={}
        def fit(site,upstream,where):
            b=basis[site]
            # Deliberately the original candidate construction and train-GCV fit:
            # policy comparison changes only order/selection, not the operator.
            upstream_key=fingerprint([__import__('dataclasses').asdict(a) for a in upstream])
            multi_site = mode in ('best_forward', 'positive_forward_tree')
            key=(upstream_key, None if multi_site else site)
            if key not in ready:
                start=max((sites.index(a.node_name) for a in upstream),default=-1)+1
                targets=list(sites[start:]) if multi_site else [site]
                ready.clear()  # Earlier upstream moments are durable; bound resident memory.
                ready[key]=fitter.statistics(pattern,upstream,targets)
                atomic_write_json(fitter.metrics,folder/'feature_costs.json')
            candidates=fit_look_node(graph,train_loader,site,pattern,b.factor,latent_dims,
                max_rank,device,b,upstream_artifacts=upstream,latent_statistics=ready[key][site])
            for q in sorted(candidates):yield f'{q:08d}',candidates[q]
        cache={}
        def evaluate(bank):
            if should_pause():raise SelectionPaused()
            # Same in-memory bank is evaluated once; final replay is done by case runner.
            key=tuple(id(a) for a in bank)
            if key in cache and all(a is b for a,b in zip(cache[key][0],bank)):
                return cache[key][1]
            r=evaluate_missing(graph,dev_loader,device,fixed_pattern=pattern,artifact_banks={pattern:bank})
            import hashlib
            digest=hashlib.sha256()
            for name in ('participant_ids','labels','logits'):
                a=np.ascontiguousarray(r[name]);digest.update(str((a.dtype,a.shape)).encode());digest.update(a.tobytes())
            path=folder/'predictions'/(digest.hexdigest()+'.npz')
            if not path.exists():save_prediction_bundle(r,path)
            result=dict(role='development',data_role=dev_loader.dataset.split,score=float(r['metrics']['macro_f1']),
                prediction=str(path),sha256=file_sha256(path),metrics=r['metrics'])
            cache[key]=(tuple(bank),result);return result
        started=time.monotonic()
        bank,r=fit_trajectory(identity=dict(case=identity,pattern=pattern,factor=factor,latent_dims=latent_dims,
                max_rank=max_rank),sites=sites,mode=mode,output=folder,fit_candidates=fit,evaluate=evaluate,
                save_artifact=lambda a,p:a.save(p),load_artifact=LOOKArtifact.load,should_pause=should_pause)
        score=r['final']['score'];hist.append(r)
        records.append(dict(factor=factor,primary_score=score,active_sites=len(bank),
            session_seconds=time.monotonic()-started,site_attempts=r.get('site_attempts',len(r['decisions'])),
            candidate_evaluations=r.get('candidate_evaluations',sum(len(d['candidates']) for d in r['decisions']))))
        rank=(score,-len(bank),factor)
        if winner is None or rank>winner:winner=rank;best=bank
    if best is None:raise ValueError('No factor completed')
    save_selected_bank(best,root)
    atomic_write_json(dict(status='selected',mode=mode,selected_factor=winner[2],selected_primary_score=winner[0],
        tie_break='fewer_active_sites_then_larger_factor',factor_banks=records,test_access=False),root/'factor_selection.json')
    return best,hist
