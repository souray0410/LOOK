"""Verified paired array export, with no historical-model or test dependency."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from look.runtime.state import file_sha256


class ArrayPair(Dataset):
    def __init__(self, root, role, *, augment=False, seed=3416):
        if role not in ('train', 'development') or (role != 'train' and augment):
            raise ValueError('Only train/development authorized')
        self.root=Path(root).resolve(); self.split=role; self.augment=augment
        self.seed=seed; self.epoch=0; self.rows={}; self.verified=set()
        audit=json.loads((self.root/'accepted.json').read_text())
        if audit.get('status')!='accepted' or audit.get('test_used') is not False:
            raise ValueError('Accepted train/dev-only export required')
        for track in ('cfp_2d', 'oct_bscan_2d'):
            p=self.root/track/(role+'.json')
            if file_sha256(p)!=audit['manifest_sha256'][track][role]:
                raise ValueError('Manifest changed')
            obj=json.loads(p.read_text())
            if obj['role']!=role:raise ValueError('Split role mismatch')
            self.rows[track]=obj['samples']
        a,b=self.rows.values()
        if [(r['id'],r['label']) for r in a]!=[(r['id'],r['label']) for r in b]:
            raise ValueError('Paired identity/label mismatch')
        self.participant_ids=[str(r['id']) for r in a]
        if len(set(self.participant_ids))!=len(a):raise ValueError('Duplicate participant')
        self.counts=[2]*len(a)
        self.audit=audit

    def __len__(self):return len(self.participant_ids)
    def set_epoch(self, epoch):self.epoch=epoch
    def __getitem__(self, i):
        out={}
        flip=self.augment and (hashlib.sha256(f'flip:{self.seed}:{self.epoch}:{self.participant_ids[i]}'.encode()).digest()[0]&1)
        for track,key in (('cfp_2d','cfp'),('oct_bscan_2d','oct')):
            r=self.rows[track][i]; p=(self.root/track/r['file']).resolve()
            if not p.is_relative_to(self.root/track):raise ValueError('Escaping array path')
            if (track,i) not in self.verified:
                if file_sha256(p)!=r['sha256']:raise ValueError('Array changed')
                self.verified.add((track,i))
            a=np.load(p,allow_pickle=False)
            if a.shape!=(2,3,224,224) or a.dtype!=np.uint8:raise ValueError('Unexpected paired input')
            if flip:a=a[...,::-1]
            x=torch.from_numpy(a.copy()).float()/255
            pre=self.audit['tracks'][track]
            out[key]=(x-torch.tensor(pre['mean']).view(1,3,1,1))/torch.tensor(pre['std']).view(1,3,1,1)
        return dict(out,label=int(self.rows['cfp_2d'][i]['label']),participant_id=self.participant_ids[i])
