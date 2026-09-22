"""Current study GPU configuration: explicit workload cap, no fixed reserve."""
import math


def validate(spec):
    if spec.get('gpu_reserve_bytes') != 0:
        raise ValueError('Current GPU policy requires zero fixed reserve; explicitly migrate the study resource configuration')
    cap = spec.get('gpu_budget_bytes')
    if type(cap) not in (int, float) or not math.isfinite(cap) or cap <= 0:
        raise ValueError('A positive validated workload GPU budget is required')
    return cap


def configure(spec, device=0):
    from mhd_models.scheduling.gpu_budget import configure_allocator
    # Account for actual other-process/context use, without subtracting a fixed
    # reserve. The explicit workload cap remains subject to lifecycle profiling.
    return configure_allocator(device, cap=validate(spec))
