"""Execution-only memory controls. Source configuration/PCA identity stay unchanged."""
import os
from dataclasses import replace
import torch
from .start_study import SuffixRunner, read_json
from .pipeline import ExperimentRunner
from .data import make_loader


def microbatch():
    value = int(os.environ.get('LOOK_EXECUTION_MICROBATCH', '8'))
    if value not in (8, 4, 2, 1): raise ValueError('Unsupported execution microbatch')
    return value


def install_allocator_limits():
    # 2 GiB is reserved for contexts and allocations outside the torch allocator.
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory
        torch.cuda.set_per_process_memory_fraction(min(12*1024**3/total, 1.), i)
    torch.cuda.set_device(0)


class BudgetRunner(SuffixRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # A single foreground GPU owns numerical execution. Second GPU is an
        # allowed resource, not a requirement; no unbudgeted criterion workers.
        self.options = replace(self.options, gpu_devices=(self.options.gpu_devices[0],))

    def _build_loaders(self):
        # Keep historical inference/fitting batches: cuDNN TF32 changes logits
        # across batch sizes. Only graph allocation and SSF training are capped.
        _, datasets = super()._build_loaders()
        loaders = {name:make_loader(data, self.config.micro_batch_size, 0,
            train=name=='train', seed=self.selection.seed, sampling_strategy=self.config.sampling_strategy)
            for name,data in datasets.items()}
        return loaders, datasets

    def _load_frozen_graph(self, checkpoint_path):
        original = self.config
        try:
            self.config = replace(original, micro_batch_size=min(microbatch(),original.micro_batch_size))
            graph, checkpoint = ExperimentRunner._load_frozen_graph(self, checkpoint_path)
        finally:
            self.config = original
        if read_json(self.source['result']['path'])['checkpoint']['backbone_id'] != self._backbone_id():
            raise RuntimeError('Source backbone identity mismatch')
        return graph, checkpoint


def install_runtime():
    install_allocator_limits()
    from . import method_study, method_logit, start_study
    for module in (method_study, method_logit, start_study): module.SuffixRunner=BudgetRunner
    original = method_study.make_loader
    def ssf_loader(dataset, batch_size, *args, **kwargs):
        return original(dataset, min(microbatch(), batch_size), *args, **kwargs)
    method_study.make_loader=ssf_loader
