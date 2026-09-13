"""Explicit access boundary for frozen-model test and later dataset adapters."""
from pathlib import Path
from look.runtime.state import file_sha256,stable_hash


def validate_release(lock,model_registry,comparison_registry,data_manifest,audit):
    if lock.get('schema')!='look_test_release_v1' or lock.get('state')!='locked':
        raise ValueError('Formal test is sealed: missing release lock')
    for key,value in (('models',model_registry),('comparisons',comparison_registry),('data',data_manifest)):
        if lock.get(key+'_sha256')!=stable_hash(value):raise ValueError('Locked '+key+' changed')
    if data_manifest.get('role')!='test' or audit.get('state')!='accepted' or audit.get('unexplained_exposure') is not False:
        raise ValueError('Test data/exposure audit not accepted')
    if lock.get('audit_sha256')!=stable_hash(audit):raise ValueError('Exposure audit changed')
    if not model_registry.get('models') or not model_registry.get('all_training_resolved'):
        raise ValueError('Incomplete model registry')
    for m in model_registry['models']:
        if m.get('state')!='accepted' or m.get('development_replay')!='accepted':
            raise ValueError('Unaccepted model or development replay')
        if file_sha256(Path(m['checkpoint']))!=m['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    return True


def validate_transfer(contract):
    required=('participant_split_sha256','label_definition_sha256','preprocessing_sha256',
              'modality_dimensions','node_mapping_sha256','source_dataset','target_dataset')
    if any(not contract.get(k) for k in required):raise ValueError('Incomplete dataset/model adapter contract')
    if contract.get('mode')=='frozen_transfer':
        if contract.get('target_fitting') is not False or contract.get('compatible_labels_inputs') is not True:
            raise ValueError('Frozen transfer cannot fit on target or assume compatibility')
    elif contract.get('mode')!='method_replication':raise ValueError('Unknown transfer interpretation')
    return True
