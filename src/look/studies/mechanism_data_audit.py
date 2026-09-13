"""Read-only provenance and metadata availability for accepted parent manifests."""
import json
from pathlib import Path
from collections import Counter
from look.runtime.state import file_sha256,stable_hash


def audit_parent_specs(parent_specs):
    checks=[]
    for spec in parent_specs:
        sets={};rows={}
        for role in ('train','development'):
            path=Path(spec[role+'_manifest'])
            if file_sha256(path)!=spec[role+'_manifest_sha256']:raise ValueError('Parent data manifest changed')
            m=json.loads(path.read_text());samples=m['samples'];ids=[str(r['id']) for r in samples]
            if len(ids)!=len(set(ids)):raise ValueError('Repeated participant in split')
            if any(int(r['label']) not in (0,1) for r in samples):raise ValueError('Changed disease label domain')
            sets[role]=set(ids);rows[role]=samples
        if sets['train']&sets['development']:raise ValueError('Train/development participant overlap')
        columns=set().union(*(r.keys() for rr in rows.values() for r in rr))
        fields={key:sorted(columns&set(aliases)) for key,aliases in dict(
            center=['center','centre','assessment_center','assessment_centre'],
            device=['device','device_model','scanner'],time=['date','imaging_date','assessment_date'],
            natural_missingness=['missing_cfp','missing_oct','availability']).items()}
        checks.append(dict(track=spec['track'],split_counts={role:len(v) for role,v in sets.items()},
            cases={role:sum(int(r['label']) for r in samples) for role,samples in rows.items()},
            manifest_sha256={role:spec[role+'_manifest_sha256'] for role in rows},
            participant_sha256={role:stable_hash(sorted(v)) for role,v in sets.items()},
            metadata_columns=fields,missing_metadata=[k for k,v in fields.items() if not v],
            natural_missingness_interpretation='paired_cache_cannot_estimate_full_source_population_missingness'))
    return dict(schema='look_source_data_audit_v1',checks=checks,test_access=False,
        centre_split_changed=False,scope='current_train_dev_manifests; missing_metadata_requires_raw_source_audit')
