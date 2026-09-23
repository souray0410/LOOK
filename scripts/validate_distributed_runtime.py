"""Two-rank real-data MHD V5 DDP acceptance, not a scientific experiment."""
import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from validate_runtime import file_sha, write


def tensor_digest(items):
    digest = hashlib.sha256()
    for name, value in items:
        digest.update(name.encode())
        if value is None:
            digest.update(b'none')
        else:
            value = value.detach().cpu().contiguous()
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.reshape(-1).view(__import__('torch').uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', choices=['LOOK'], required=True,
        help='This repository validates LOOK only; Radon_Bridge owns its independent V5 artifact contract')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--artifact-root', type=Path, required=True, help='Project-owned verified weights and bases')
    parser.add_argument('--gpu-budget-bytes', type=int, required=True,
        help='Accepted complete-lifecycle per-rank budget for this exact workload; no fixed reserve is implied')
    args = parser.parse_args()
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, Subset
    from torch.nn.parallel import DistributedDataParallel
    import mhd_framework
    from mhd_framework.utils import (initialize_mhd_distributed, destroy_mhd_distributed,
        MHD_Trainer, MHD_Monitor, MHD_ParallelConfig)

    assert mhd_framework.__api_version__ == 'V5'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 2
    assert int(os.environ.get('WORLD_SIZE', '1')) == 2
    context = initialize_mhd_distributed('nccl')
    rank, device = context.rank, context.device
    root, output = args.data_root, args.output
    artifacts = args.artifact_root
    artifact_manifest = json.loads((artifacts/'artifacts_manifest.json').read_text())
    assert artifact_manifest['project'] == args.project
    assert artifact_manifest['source_manifest_sha256'] == args.manifest_sha
    assert artifact_manifest['files'] and len(artifact_manifest['files']) == len({row['path'] for row in artifact_manifest['files']})
    for entry in artifact_manifest['files']:
        path = artifacts/entry['path']
        assert path.resolve().is_relative_to(artifacts.resolve()) and not path.is_symlink()
        assert path.stat().st_size == entry['bytes'] and file_sha(path) == entry['sha256']
    out = output / f'rank{rank}'
    out.mkdir(parents=True, exist_ok=True)
    assert file_sha(root / 'manifest.json') == args.manifest_sha
    assert json.loads((root / 'accepted.json').read_text())['manifest_sha256'] == args.manifest_sha
    os.environ['TORCH_HOME'] = str(artifacts / 'weights')
    total = torch.cuda.get_device_properties(device).total_memory
    assert 0 < args.gpu_budget_bytes <= total
    torch.cuda.set_per_process_memory_fraction(args.gpu_budget_bytes / total, device)
    torch.manual_seed(9181)
    started = time.monotonic()
    record = dict(project=args.project, framework=mhd_framework.__version__, rank=rank,
        world_size=2, global_batch=16, local_batch=8, backend=dist.get_backend(),
        test_used=False, scientific_training_result=False, state='running',
        gpu_budget_bytes=args.gpu_budget_bytes, gpu_total_bytes=total,
        gpu_budget_source='explicit_complete_lifecycle_profile', fixed_gpu_reserve_bytes=0,
        manifest_sha256=args.manifest_sha,
        artifact_manifest_sha256=file_sha(artifacts/'artifacts_manifest.json'), pid=os.getpid(), device=str(device), gpu=torch.cuda.get_device_name(device),
        batchnorm='ordinary per-rank BN; rank0 buffers define the checkpoint',
        scope='four disposable DDP updates and finite sharded inference')

    def progress(stage, **values):
        record.update(stage=stage, elapsed_seconds=time.monotonic()-started, **values)
        write(out / 'status.json', record)
        print(json.dumps(dict(rank=rank, stage=stage, **values)), flush=True)

    def identical(value, name):
        gathered = [None] * 2
        dist.all_gather_object(gathered, value)
        assert gathered[0] == gathered[1], f'{name} differs between ranks'
        return value

    progress('load_data')
    from look.data.dataset import UKBBilateralVisitDataset
    def dataset(split):
        return UKBBilateralVisitDataset(root/'look/reference_labels_train_development.csv',
            root/'look/raw_not_included', split, augment=False,
            preprocess_cache_root=root/'look/cache')
    train, dev = dataset('train'), dataset('validation')
    assert (len(train), len(dev)) == (1264, 296)

    def loader(data):
        indices = list(range(rank, len(data), 2))
        gathered = [None, None]
        dist.all_gather_object(gathered, indices)
        assert not set(gathered[0]).intersection(gathered[1])
        assert sorted(gathered[0]+gathered[1]) == list(range(len(data)))
        return DataLoader(Subset(data, indices), batch_size=8, shuffle=False, num_workers=0)
    first = next(iter(loader(train)))
    progress('load_reference_model')
    from look.models.graph import build_resnet50_mhd_graph, optimizer_parameter_groups, reset_and_forward
    from look.runtime.host_checkpoint import read_selected
    selected = json.loads((artifacts/'look/parent/selected.json').read_text())
    assert (selected.get('schema') == 'look_v5_selected_reference_v1'
        and selected.get('framework_api') == 'V5'
        and isinstance(selected.get('identity'), str)
        and isinstance(selected.get('sha256'), str)
        and isinstance(selected.get('config'), dict)), 'Current LOOK V5 selected reference required'
    checkpoint_path = artifacts/'look/parent/best.pt'
    assert file_sha(checkpoint_path) == selected['sha256']
    config = selected['config']
    graph = build_resnet50_mhd_graph('layer3', batch_size=8, device=device, pretrained=False,
        classifier_dropout=config.get('classifier_dropout',0.), label_smoothing=config.get('label_smoothing',0.))
    ids = [(node.id,node.name) for node in sorted(graph.nodes,key=lambda node:node.id)]
    checkpoint = read_selected(checkpoint_path,identity=selected['identity'],node_ids=ids)
    graph.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    optimizer = torch.optim.AdamW(optimizer_parameter_groups(graph,3e-4,.003),weight_decay=.0001)
    input_names, output_names = ('oct_input','cfp_input','label_gt'), ('fusion_logits',)
    forward_levels, backward_levels = graph.forward_levels, graph.backward_levels
    def inputs(batch):
        return dict(oct_input=batch['oct'].to(device),cfp_input=batch['cfp'].to(device),label_gt=batch['label'].to(device))
    def predict(batch):
        values=inputs(batch)
        return {'fusion':reset_and_forward(graph,values['oct_input'],values['cfp_input'])}
    def clip():
        pass  # Preserve the original LOOK acceptance optimizer policy.
    trainer = MHD_Trainer(graph,optimizer,MHD_Monitor(['loss']),forward_levels,backward_levels,
        criteria=lambda g: g.get_node_by_name('loss').feature_message.current_state,
        save_dir=str(output/'trainer'),input_nodes=input_names,output_nodes=output_names,
        parallel=MHD_ParallelConfig(data_parallel='ddp'),distributed_context=context,precision='fp32')
    assert isinstance(trainer.model, DistributedDataParallel)
    record['initial_state_sha256'] = identical(tensor_digest(graph.state_dict().items()), 'initial model state')
    node_ids = sorted((node.id,node.name) for node in graph.nodes)
    identical(node_ids,'Node IDs')
    tracked = next(param for param in graph.parameters() if param.requires_grad)
    before = tracked.detach().cpu().clone()
    gradient_checks = []
    def before_step(optimizer, step_args, step_kwargs):
        gradients = [(name,param.grad) for name,param in graph.named_parameters() if param.requires_grad]
        assert gradients and all(value is not None and torch.isfinite(value).all() for _,value in gradients)
        digest = identical(tensor_digest(gradients), 'DDP reduced gradients')
        gradient_checks.append(digest)
        clip()
    hook = optimizer.register_step_pre_hook(before_step)
    progress('ddp_updates')
    for index,batch in enumerate(loader(train)):
        if index == 4:
            break
        metrics = trainer.train_step(inputs(batch))
        assert __import__('math').isfinite(metrics['loss'])
        identical(tensor_digest(graph.named_parameters()),'updated parameters')
        progress('ddp_update', optimizer_updates=index+1)
    hook.remove()
    assert len(gradient_checks)==4 and not torch.equal(before,tracked.detach().cpu())
    assert node_ids == sorted((node.id,node.name) for node in graph.nodes)
    record.update(gradient_sha256_per_step=gradient_checks, native_parameters_updated=True,
        node_ids_preserved=True, strict_reference_checkpoint_loaded=True, ddp_wrapper_verified=True)
    # Ordinary DDP BN is rank-local. Select rank0 buffers explicitly before evaluation.
    for buffer in graph.buffers():
        dist.broadcast(buffer,src=0)
    record['checkpoint_state_sha256'] = identical(tensor_digest(graph.state_dict().items()),'checkpoint state')
    graph.eval()
    progress('checkpoint_roundtrip')
    with torch.no_grad():
        reference = {key:value.cpu().clone() for key,value in predict(first).items()}
    if rank == 0:
        state = {key:value.detach().cpu() for key,value in graph.state_dict().items()}
        torch.save(dict(model=state,optimizer=optimizer.state_dict(),world_size=2,scope=record['scope']),output/'roundtrip.pt')
        del state
    dist.barrier()
    saved = torch.load(output/'roundtrip.pt',map_location='cpu',weights_only=False)
    graph.load_state_dict(saved['model'],strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    del saved
    with torch.no_grad():
        replay = predict(first)
    for key in reference:
        torch.testing.assert_close(reference[key],replay[key].cpu(),rtol=0,atol=0)
    record['checkpoint_output_exact'] = True
    for split,data in [('train',train),('development',dev)]:
        progress('infer_'+split)
        seen = 0
        digest = hashlib.sha256()
        with torch.no_grad():
            for batch in loader(data):
                prediction = predict(batch)
                for key,value in prediction.items():
                    assert value.ndim==2 and value.shape[1]==2 and torch.isfinite(value).all()
                    digest.update(key.encode());digest.update(value.cpu().float().numpy().tobytes())
                seen += len(next(iter(prediction.values())))
                if seen % 128 == 0:
                    progress('infer_'+split,seen=seen)
        count = torch.tensor(seen,device=device)
        dist.all_reduce(count)
        assert count.item()==len(data)
        record[split+'_inference'] = dict(local_count=seen,total_count=count.item(),local_logit_sha256=digest.hexdigest())
    torch.cuda.synchronize()
    record.update(state='accepted',stage='complete',elapsed_seconds=time.monotonic()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        multigpu_training_tested=True)
    write(out/'accepted.json',record);write(out/'status.json',record)
    gathered = [None,None]
    dist.all_gather_object(gathered,record)
    if rank==0:
        write(output/'accepted.json',dict(state='accepted',project=args.project,world_size=2,ranks=gathered,
            global_batch=16,local_batch=8,test_used=False,scientific_training_result=False))
    destroy_mhd_distributed()

if __name__=='__main__':
    main()
