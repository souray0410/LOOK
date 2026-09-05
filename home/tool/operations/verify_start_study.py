#!/usr/bin/env python3
"""Bounded CPU acceptance on a real frozen backbone; never opens test or trains a resource."""
import argparse
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import torch

from look_core.start_study import read_json, make_cases, source_result, pin_source, SuffixRunner, verify_source
from look_core.paths import ProjectPaths
from look_core.state import atomic_write_json, file_sha256, utc_now
from look_core.look import greedy_fit_look, load_selected_bank, forward_with_look
from look_core.filling import NormalizedMeanFiller, RawZeroFiller, PairedCGANFiller
from look_core.data import UKBBilateralVisitDataset, make_loader
from look_core.evaluate import evaluate_missing
from look_core.reproducibility import seed_everything, implementation_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,required=True)
    p.add_argument('--parent-summary',type=Path,required=True)
    p.add_argument('--parent-source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2); seed_everything(3407)
    parent=read_json(args.parent_summary); runs=args.parent_summary.parents[2]
    spec=read_json(args.parent_source/'configs/unified_study.json')
    context='normalized_mean_feature_3407'
    cases=[c for c in make_cases(spec,parent['selected_fusions']) if c['context']==context]
    source=pin_source(source_result(parent,runs,context),parent,runs,args.project_root)
    runner=SuffixRunner(source,cases[0],args.output/'runtime',args.output/'cache',torch.device('cpu'))
    graph,_=runner._load_frozen_graph(runner._train_or_resume())
    loaders={}
    for split,n in [('train',6),('validation',6)]:
        data=UKBBilateralVisitDataset(runner.config.labels_csv,runner.config.image_root,split,augment=False,limit=n,
            image_size=runner.config.image_size,preprocess_cache_root=runner.config.preprocess_cache_root)
        loaders[split]=make_loader(data,3,0,train=False,seed=3407)
    bank=runner._prepare_shared_pca(graph,loaders['train'],Path(source['checkpoint']['path']))
    # A deterministic generator fixture exercises cGAN filling conversion/inference only.
    # It is not a trained cGAN result and is never entered in the formal study.
    generators=[torch.nn.Sequential(torch.nn.Conv2d(3,3,1),torch.nn.Tanh()) for _ in range(2)]
    fillers=[NormalizedMeanFiller(),RawZeroFiller(),PairedCGANFiller(*generators,torch.device('cpu'))]
    batch=next(iter(loaders['validation']))
    outcomes=[]
    for filler in fillers:
        for pattern in ('oct_missing','cfp_missing'):
            baseline=evaluate_missing(graph,loaders['validation'],torch.device('cpu'),fixed_pattern=pattern,filler=filler)
            off=evaluate_missing(graph,loaders['validation'],torch.device('cpu'),fixed_pattern=pattern,filler=filler,artifact_banks={pattern:[]})
            assert np.array_equal(baseline['logits'],off['logits'])
            for index in [7,8]:
                output=args.output/filler.name/pattern/f'start{index+1}'
                nodes=cases[index]['eligible_sites']
                selected,_=greedy_fit_look(graph,loaders['train'],loaders['validation'],pattern,nodes,[16],[1,2],512,
                    torch.device('cpu'),output,bank,filler=filler,primary_metric='macro_f1')
                before={str(f):file_sha256(f) for f in output.rglob('*') if f.is_file()}
                restored=load_selected_bank(output)
                result=evaluate_missing(graph,loaders['validation'],torch.device('cpu'),fixed_pattern=pattern,filler=filler,artifact_banks={pattern:restored})
                assert result['metrics']['macro_f1']>=baseline['metrics']['macro_f1']
                greedy_fit_look(graph,loaders['train'],loaders['validation'],pattern,nodes,[16],[1,2],512,
                    torch.device('cpu'),output,bank,filler=filler,primary_metric='macro_f1')
                after={str(f):file_sha256(f) for f in output.rglob('*') if f.is_file()}
                # Selected manifests are deterministic; no completed node is refitted on resume.
                assert {k:v for k,v in before.items() if '/decisions/' in k or '/candidates/' in k} == {k:v for k,v in after.items() if '/decisions/' in k or '/candidates/' in k}
                outcomes.append(dict(filling=filler.name,pattern=pattern,start=index+1,enabled=[a.node_name for a in selected],macro_f1=result['metrics']['macro_f1']))
    # Direct input correction must propagate into the collected downstream fitting state.
    from look_core.look import fit_look_node, iter_feature_pairs
    input_artifact=fit_look_node(graph,loaders['train'],'joint_input','oct_missing',16,[1],512,torch.device('cpu'),bank[('joint_input',16)],filler=NormalizedMeanFiller())[1]
    upstream=replace(input_artifact,weight=torch.zeros_like(input_artifact.weight),bias=torch.ones_like(input_artifact.bias))
    original=list(iter_feature_pairs(graph,loaders['train'],'fusion_participant_feature','oct_missing',1,torch.device('cpu'),upstream_artifacts=[],filler=NormalizedMeanFiller()))
    corrected=list(iter_feature_pairs(graph,loaders['train'],'fusion_participant_feature','oct_missing',1,torch.device('cpu'),upstream_artifacts=[upstream],filler=NormalizedMeanFiller()))
    assert any(not torch.equal(a[1],b[1]) for a,b in zip(original,corrected))
    verify_source(source)
    report=dict(status='passed',implementation_sha256=implementation_sha256(args.project_root),completed_at_utc=utc_now(),
        test_access=False,checkpoint=source['checkpoint'],all_off_logits_exact=True,upstream_input_reaches_downstream_fit=True,
        source_resources_unchanged=True,resume_decisions_unchanged=True,results=outcomes,
        interpretation='Technical acceptance on six train and six validation participants; cGAN uses an untrained deterministic fixture. Not scientific evidence.')
    atomic_write_json(report,args.output/'verification.json')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
