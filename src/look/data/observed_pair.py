"""Paired participant datasets using the exact selected parents' preprocessing."""
from torch.utils.data import Dataset
import torch
from look.models.observed_participant import verify_pair_rows


class ObservedPair(Dataset):
    def __init__(self, first, second, split, augment=False):
        if split not in ('train','development'):
            raise ValueError('This fitting adapter refuses test data')
        if bool(first.augment) != augment or bool(second.augment) != augment:
            raise ValueError('Paired augmentation declaration mismatch')
        self.pairing = verify_pair_rows(first.rows, second.rows)
        self.counts = [len(row['eyes']) for row in first.rows]
        self.first, self.second = first, second
        self.split, self.augment = split, augment
        self.participant_ids = [str(r['id']) for r in first.rows]
        if split != 'train' and augment:
            raise ValueError('Evaluation cannot use augmentation')

    def __len__(self):
        return len(self.first)

    def __getitem__(self, index):
        cfp, y, i = self.first[index]
        oct_x, other_y, j = self.second[index]
        if y != other_y or i != j or len(cfp) != len(oct_x) or len(cfp) != self.counts[index]:
            raise ValueError('Paired sample identity changed')
        return dict(cfp=cfp, oct=oct_x, label=y, participant_id=self.participant_ids[index])

    def set_epoch(self, epoch):
        self.first.epoch = self.second.epoch = epoch


def collate_observed(rows):
    return dict(cfp=torch.cat([r['cfp'] for r in rows]), oct=torch.cat([r['oct'] for r in rows]),
                counts=[len(r['cfp']) for r in rows], label=torch.tensor([r['label'] for r in rows]),
                participant_id=[r['participant_id'] for r in rows])


def from_parent_specs(first, second, role, inputs_factory, *, augment=False, seed=3416):
    if role not in ('train','development'):
        raise ValueError('Sealed split cannot be requested by training')
    if first[role+'_manifest_sha256'] != second[role+'_manifest_sha256']:
        raise ValueError('Paired UKB parents must share the same versioned visit/cache manifest')
    datasets = []
    for spec, track in ((first,'cfp_2d'), (second,'oct_bscan_2d')):
        if spec.get('test_used') is not False or spec['track'] != track:
            raise ValueError('Unexpected parent modality or test exposure')
        datasets.append(inputs_factory(spec[role+'_manifest'],spec[role+'_manifest_sha256'],track,
            seed=seed,augment=augment,recipe=spec['training'].get('recipe')))
    return ObservedPair(*datasets, role, augment)
