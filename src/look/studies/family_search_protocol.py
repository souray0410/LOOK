"""Explicit finite matched fitting-family search, separate from legacy train GCV."""
from look.studies.search_protocol import validate as validate_reference, PATTERNS
from look.methods.family_greedy import validate_candidates
VERSION = 'look_family_search_v1'
ARMS = ('residual_rrr', 'shared_pca_ridge', 'pca_free_mean', 'rrr_shared_intercept')


def protocol():
    return dict(schema=VERSION, test_access=False, patterns=list(PATTERNS),
        mode='positive_forward_tree', fitting='train_only_centered_residual_moments',
        selection='train_PCA_reference_GCV_per_prefix_then_dev_positive_tree',
        penalty_policy='prefix_train_pca_gcv',
        penalty_limitation='controlled_PCA_reference_penalty_not_independently_optimal_RRR_penalty',
        legacy_gcv_equivalent=False, baseline='frozen_uncorrected_host',
        ties='off_then_registered_tree_tiebreak', repeats='blocked_until_weekly_package_acceptance')


def validate(spec):
    if spec.get('schema') != VERSION or spec.get('protocol') != protocol() or spec.get('test_access') is not False:
        raise ValueError('Unregistered family search or test access')
    ref = spec['reference_search']
    validate_reference(ref)
    if spec.get('host')!=ref['host']:raise ValueError('Family host identity differs from reference')
    if ref['host'] != dict(disease='cataract', architecture='resnet50', position='deep', seed=3416):
        raise ValueError('Only registered reference host and first seed allowed')
    if ref['mode'] != 'positive_forward_tree' or ref.get('spatial_factors') != [16] or ref.get('latent_dims') != [32]:
        raise ValueError('Matched x16 rank32 positive tree required')
    if spec.get('arm') not in ARMS:
        raise ValueError('Unregistered matched fitting arm')
    if spec.get('penalty_policy') != 'prefix_train_pca_gcv':
        raise ValueError('Registered train-only penalty rule required')
    validate_candidates(spec['arm'],spec['candidates'],penalty_policy=spec['penalty_policy'])
    if spec['candidates'] != [dict(rank=32, ridge_lambda=None)]:
        raise ValueError('Every finite candidate requires rank32')
    if not spec.get('candidate_provenance') or not spec.get('source_pins'):
        raise ValueError('Candidate rationale and immutable family source pins required')
    if type(spec.get('workspace_bytes')) is not int or spec['workspace_bytes'] <= 0:
        raise ValueError('Explicit full-dimensional fitting workspace required')
    return spec
