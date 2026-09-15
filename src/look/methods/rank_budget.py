"""Separate a common integer rank budget from method-specific retained quantities."""
import math


def retained_curve(component_values, total, *, quantity):
    """Denominator must cover the ORIGINAL target, never just retained components.

    A truncated decomposition may not reach a requested fraction. A zero-energy
    target has an undefined retained fraction, not 100 percent retention.
    """
    values=[float(v) for v in component_values];total=float(total)
    if not quantity or not math.isfinite(total) or total<0 or any(not math.isfinite(v) or v<0 for v in values):
        raise ValueError('Invalid nonnegative spectrum and target total')
    if sum(values)>total+1e-8*max(1,total):raise ValueError('Components exceed declared total')
    cumulative=0.;rows=[]
    for rank,value in enumerate(values,1):
        cumulative+=value
        rows.append(dict(rank=rank,retained=None if total==0 else min(1.,cumulative/total)))
    return dict(quantity=quantity,total=total,available_rank=len(values),curve=rows,
                defined=total>0,denominator='full_target_not_retained_component_sum')


def rank_for_fraction(curve, fraction):
    fraction=float(fraction)
    if not math.isfinite(fraction) or not 0<fraction<=1:raise ValueError('Fraction must be in (0,1]')
    if not curve['defined']:return dict(state='undefined_zero_target',rank=None,fraction=fraction)
    for row in curve['curve']:
        if row['retained']>=fraction:return dict(state='reached',rank=row['rank'],fraction=fraction)
    return dict(state='unreachable_in_available_basis',rank=None,fraction=fraction)


def matched_ranks(requested, limits):
    """Keep every requested budget with an explicit joint-feasibility decision."""
    if not limits or any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in limits.values()):
        raise ValueError('Integer method-specific feasible rank limits required')
    if any(isinstance(q,bool) or not isinstance(q,int) or q<1 for q in requested):
        raise ValueError('Positive integer rank budgets required')
    return [dict(rank=q,state='feasible' if all(q<=v for v in limits.values()) else 'infeasible',
                 limiting_methods=[k for k,v in limits.items() if q>v]) for q in sorted(set(requested))]
