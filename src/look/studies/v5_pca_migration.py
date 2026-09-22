"""Offline PCA bank rebinding after exact input-site equivalence.

Normal PCA readers retain one framework-independent mathematical format. This
importer changes dependency identities, never refits or alters numerical state.
"""
from dataclasses import replace
import fcntl
import json
from pathlib import Path
import shutil
import tempfile

import torch
from look.methods import operator, joint
from look.methods.operator import FullFeaturePCA
from look.runtime.state import file_sha256, stable_hash
from look.runtime.provenance import write_json_atomic


def migrate(source_manifest, output_root, *, feature_receipt, feature_receipt_sha256,
            target_checkpoint, target_checkpoint_sha256):
    source_manifest = Path(source_manifest).resolve()
    source = source_manifest.parent
    output_root = Path(output_root).resolve()
    if output_root == source or output_root.is_relative_to(source):
        raise ValueError('Source bank cannot be overwritten')
    if file_sha256(feature_receipt) != feature_receipt_sha256:
        raise ValueError('Feature replay receipt changed')
    replay = json.loads(Path(feature_receipt).read_text())
    bank = json.loads(source_manifest.read_text())
    identity = bank['identity']
    if (replay.get('schema') != 'look_node_feature_replay_v1'
            or replay.get('state') != 'accepted' or replay.get('test_access') is not False
            or replay.get('all_site_values_bitwise_equal') is not True
            or replay.get('framework_api') != 'V5'
            or replay.get('original_bank_sha256') != file_sha256(source_manifest)
            or replay.get('source_checkpoint_sha256') != identity['host_best_sha256']
            or replay.get('target_checkpoint_sha256') != target_checkpoint_sha256
            or file_sha256(target_checkpoint) != target_checkpoint_sha256):
        raise ValueError('Exact PCA input and host equivalence required')
    if bank['bank_id'] != stable_hash(identity)[:16] or source.name != bank['bank_id']:
        raise ValueError('Original bank identity changed')
    modules = {'look.methods.operator': ('pca_code_sha256', operator),
               'look.methods.joint': ('joint_code_sha256', joint)}
    for name, (key, module) in modules.items():
        current = file_sha256(module.__file__)
        if identity[key] != current or replay['imported_look_source_sha256'].get(name) != current:
            raise ValueError('PCA/joint operator changed; separate numerical migration required')
    entries = bank['entries']
    if (not entries or len({(row['node'], row['factor']) for row in entries}) != len(entries)
            or set(replay['sites']) != {row['node'] for row in entries}
            or any(type(replay['counts'].get(role)) is not int or replay['counts'][role] <= 0
                   for role in ('train', 'development'))):
        raise ValueError('Incomplete site coverage')
    target_identity = dict(identity, host_best_sha256=target_checkpoint_sha256)
    bank_id = stable_hash(target_identity)[:16]
    destination = output_root / bank_id
    if destination == source:
        raise ValueError('Source bank cannot be overwritten')
    contract = dict(schema='look_pca_bank_migration_v1',
        source_manifest_sha256=file_sha256(source_manifest),
        feature_receipt_sha256=feature_receipt_sha256,
        source_identity=identity, target_identity=target_identity,
        state='candidate_input_equivalence_and_tensor_preservation',
        original_fit_preserved=True, test_access=False, production_dispatch_authorized=False)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root/(bank_id+'.migration.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            prior = json.loads((destination/'migration.json').read_text())
            if prior['contract'] != contract:
                raise ValueError('Existing migration has different dependencies')
            for name, digest in prior['files'].items():
                if file_sha256(destination/name) != digest:
                    raise ValueError('Existing converted bank changed')
            return destination
        stage = Path(tempfile.mkdtemp(prefix=bank_id+'.', dir=output_root))
        try:
            target_entries = []; files = {}
            for row in entries:
                name = row['node']+'_x'+str(row['factor'])+'.pt'
                if Path(name).name != name or row['node'] in ('.', '..'):
                    raise ValueError('Invalid PCA site filename')
                path = Path(row['path'])
                if path.is_symlink() or path.resolve() != source/name or file_sha256(path) != row['sha256']:
                    raise ValueError('Original PCA entry changed or escaped bank')
                old = FullFeaturePCA.load(path)
                if (old.source_id != row['source_id'] or old.node_name != row['node']
                        or old.factor != row['factor']):
                    raise ValueError('Original PCA metadata changed')
                new = replace(old, source_id=bank_id+'/'+name.removesuffix('.pt'))
                new.save(stage/name)
                loaded = FullFeaturePCA.load(stage/name)
                for key, value in vars(old).items():
                    other = getattr(loaded, key)
                    if isinstance(value, torch.Tensor):
                        if value.dtype != other.dtype or not torch.equal(value, other):
                            raise ValueError('PCA tensor changed: '+key)
                    elif key != 'source_id' and value != other:
                        raise ValueError('PCA fit metadata changed: '+key)
                digest = file_sha256(stage/name)
                metadata = dict(source_id=new.source_id, sha256=digest,
                                bytes=(stage/name).stat().st_size, samples=old.sample_count)
                write_json_atomic(metadata, (stage/name).with_suffix('.json'))
                files[name] = digest
                files[Path(name).with_suffix('.json').name] = file_sha256((stage/name).with_suffix('.json'))
                target_entries.append(dict(row, path=str(destination/name),
                                           source_id=new.source_id, sha256=digest))
            write_json_atomic(dict(bank_id=bank_id, identity=target_identity, entries=target_entries),
                              stage/'bank_manifest.json')
            files['bank_manifest.json'] = file_sha256(stage/'bank_manifest.json')
            write_json_atomic(dict(contract=contract, files=files), stage/'migration.json')
            stage.rename(destination)
        except BaseException:
            shutil.rmtree(stage)
            raise
    return destination
