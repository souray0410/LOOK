"""Label-independent participant subsets, masks and count-stratified permutations."""
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset
from look.runtime.state import stable_hash


def ordered_ids(ids, seed, domain):
    ids = list(map(str, ids))
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('Unique nonempty participant IDs required')
    return sorted(ids, key=lambda x: (hashlib.sha256(f'{domain}:{seed}:{x}'.encode()).hexdigest(), x))


class ParticipantSubset(Dataset):
    def __init__(self, dataset, fraction, seed):
        if dataset.split != 'train' or dataset.augment or not 0 < fraction <= 1:
            raise ValueError('Only unaugmented train subsets may fit corrections')
        ids = dataset.participant_ids
        chosen = set(ordered_ids(ids, seed, 'look_subset_v1')[:max(1, int(len(ids)*fraction+.5))])
        self.indices = [i for i,p in enumerate(ids) if str(p) in chosen]
        self.participant_ids = [ids[i] for i in self.indices]
        self.dataset, self.split, self.augment = dataset, 'train', False
        if hasattr(dataset,'counts'):self.counts=[dataset.counts[i] for i in self.indices]
        self.receipt = dict(fraction=fraction, seed=seed, participants=len(self.indices),
                            participant_sha256=stable_hash(self.participant_ids))
    def __len__(self): return len(self.indices)
    def __getitem__(self, i): return self.dataset[self.indices[i]]


def target_permutation(ids, counts, seed, grouping_counts=None):
    """Permute whole people within equal eye-count strata, preserving eye order.

    Unequal-count swaps cannot pair latent rows without discarding/duplicating an
    eye. A singleton stratum is explicitly infeasible rather than self-paired.
    """
    if len(ids) != len(counts) or any(c not in (1, 2) for c in counts):
        raise ValueError('Invalid participant ownership')
    grouping_counts=list(counts if grouping_counts is None else grouping_counts)
    if len(grouping_counts)!=len(counts):raise ValueError('Grouping ownership mismatch')
    ids = list(map(str, ids)); offsets = np.cumsum([0, *counts]); mapping = {}
    for count in sorted(set(grouping_counts)):
        group = ordered_ids([p for p,c in zip(ids,grouping_counts) if c == count], seed, 'look_target_shuffle_v1')
        if len(group) < 2: raise ValueError('infeasible_singleton_eye_count_stratum')
        mapping.update({p:group[(i+1)%len(group)] for i,p in enumerate(group)})
    index = {p:i for i,p in enumerate(ids)}
    return np.concatenate([np.arange(offsets[index[mapping[p]]], offsets[index[mapping[p]]+1]) for p in ids])


def mask_training_batch(batch, generator):
    """A private generator; no global RNG consumption or participant reordering."""
    states = torch.randint(3, (len(batch['counts']),), generator=generator)
    result = dict(batch, cfp=batch['cfp'].clone(), oct=batch['oct'].clone())
    offset = 0
    for state, count in zip(states.tolist(), batch['counts']):
        if state: result['cfp' if state == 1 else 'oct'][offset:offset+count] = 0
        offset += count
    return result, states
