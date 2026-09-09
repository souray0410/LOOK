"""Real-data, sealed train/validation acceptance; never emits formal results."""
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from look.studies.starts import read_json, verify_source
from look.runtime.bounded import BudgetRunner, microbatch
from look.studies.methods import make_resource_case
from look.methods.ssf import train_ssf
from look.methods.kernels import fit_control, evaluate_control
from look.training.distributed import module_state_sha256
from look.methods.imputation import NormalizedMeanFiller
from look.runtime.state import atomic_write_json


def run(cfg, output):
    output=Path(output)
    inv=read_json(Path(cfg['historical_suffix']).with_name('source_inventory.json'))
    source=inv['sources']['normalized_mean_layer3_3407'];verify_source(source)
    runner=BudgetRunner(source,make_resource_case(3407),output/'resources',output/'cache',torch.device('cuda:0'),(0,1))
    _,datasets=runner._build_loaders()
    def subset(name):
        ds=datasets[name];indices=[]
        for label in (0,1):indices.extend(np.flatnonzero(ds.labels==label)[:4].tolist())
        result=Subset(ds,indices);result.participant_ids=[ds.participant_ids[i] for i in indices]
        result.set_epoch=ds.set_epoch
        return result
    train,validation=subset('look_train'),subset('validation')
    def loader(ds,batch=None,shuffle=False):
        return DataLoader(ds,batch_size=batch or microbatch(),shuffle=shuffle,num_workers=0,
            generator=torch.Generator().manual_seed(3407))
    from look.runtime.provenance import seed_everything
    seed_everything(3407)
    graph,_=runner._load_frozen_graph(runner._train_or_resume())
    frozen=module_state_sha256(graph);filler=NormalizedMeanFiller();device=torch.device('cuda:0')
    pcas=runner._prepare_shared_pca(graph,loader(train),Path(source['checkpoint']['path']))
    checks={}
    # Both expensive spatial extremes plus a downstream vector node, with real shared PCA.
    config=replace(runner.config,latent_dims=[8,512],downsample_factors=[4,8,16],
        correction_nodes=['joint_input','joint_layer3','fusion_participant_feature'])
    for pattern in ('oct_missing','cfp_missing'):
        bank,result=fit_control(graph,loader(train),loader(validation),device,pcas,config,pattern,
            'missing_only',output/'correction'/pattern,source['checkpoint']['sha256'])
        recovered,again=fit_control(graph,loader(train),loader(validation),device,pcas,config,pattern,
            'missing_only',output/'correction'/pattern,source['checkpoint']['sha256'])
        assert result==again
        # Acceptance must use the legacy evaluation batch, not claim arbitrary batch equivalence.
        a=evaluate_control(graph,loader(validation,runner.config.micro_batch_size),device,{pattern:bank},policy='missing_only',filler=filler,fixed_pattern=pattern)
        b=evaluate_control(graph,loader(validation,runner.config.micro_batch_size),device,{pattern:recovered},policy='missing_only',filler=filler,fixed_pattern=pattern)
        error=float(np.max(np.abs(a['logits']-b['logits'])))
        if not np.allclose(a['logits'],b['logits'],rtol=1e-4,atol=1e-4):raise ValueError('Restored inference mismatch')
        # Full validation baseline must reproduce the original saved checkpoint output.
        base=evaluate_control(graph,loader(datasets['validation'],runner.config.micro_batch_size),device,filler=filler,fixed_pattern=pattern)
        pred=Path(source['result']['path']).parent/'predictions'/f'validation__fill_{pattern}.npz'
        with np.load(pred,allow_pickle=False) as old:
            if not np.array_equal(base['labels'],old['labels']):raise ValueError('Validation ordering differs')
            baseline_error=float(np.max(np.abs(base['logits']-old['logits'])))
            if not np.allclose(base['logits'],old['logits'],rtol=1e-4,atol=1e-4):raise ValueError(f'Historical baseline reproduction failed: {baseline_error}')
        atomic_write_json(dict(baseline_max_abs_error=baseline_error,validation_micro_batch_size=runner.config.micro_batch_size),output/(pattern+'_baseline_reproduction.json'))
        ssf=dict(learning_rates=[1e-5],weight_decay=.01,epochs=1,patience=15,effective_batch_size=128,micro_batch_size=8)
        ssf_result=train_ssf(graph,loader(train,shuffle=True),loader(validation),device,pattern,ssf,
            output/'ssf'/pattern,source['checkpoint']['sha256'],3407)
        assert module_state_sha256(graph)==frozen
        checks[pattern]=dict(inference_max_abs_error=error,baseline_max_abs_error=baseline_error,ssf_candidates=len(ssf_result['candidates']),
            correction_restore=True,frozen_backbone_unchanged=True)
    atomic_write_json(dict(status='complete',test_access=False,acceptance_only=True,train_n=8,validation_n=8,
        execution_micro_batch_size=microbatch(),checks=checks,torch_peak_reserved_bytes={str(i):torch.cuda.max_memory_reserved(i) for i in range(torch.cuda.device_count())}),output/'verification.json')
