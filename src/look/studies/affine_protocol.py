"""Finite additive mechanism family; no change to existing scientific identities."""
from look.methods.affine_family import VERSION, ARMS, NEW_ARMS
from look.studies.spatial_protocol import matrix

COMPARISONS = [
    ('pca_free_mean', 'shared_pca_ridge'),
    ('residual_rrr', 'rrr_shared_intercept'),
    ('rrr_shared_intercept', 'shared_pca_ridge'),
    ('residual_rrr', 'pca_free_mean'),
    ('residual_ridge', 'residual_rrr'),
    ('pls_svd_ridge', 'residual_rrr'),
    ('residual_ridge', 'diagonal_ridge'),
    ('residual_ridge', 'orthogonal_alignment'),
    *[(m, 'host') for m in ARMS],
]


def protocol(scope):
    if scope not in ('terminal', 'progressive'):
        raise ValueError('Unknown family scope')
    return dict(schema=VERSION, scope=scope, arms=list(ARMS), new_arms=list(NEW_ARMS),
        test_access=False, additional_neural_training=False, seeds=[3416,3417,3418],
        patterns=['oct_missing','cfp_missing'], mean_subspace_factorial=True,
        fixed=['host','participants','normalization','spatial_representation','reference_rank','reference_lambda','sites','return'],
        selection='conditional_on_original_PCA_selected_configuration_no_new_method_selection',
        upstream='sequential_refit_within_arm' if scope=='progressive' else 'single_terminal_node',
        pls='SVD_of_centered_X_transpose_D_then_ridge_in_left_subspace_not_iterative_PLSRegression',
        orthogonal='Q_transpose_Q_identity_in_standardized_coordinates_reflections_allowed_no_ridge_penalty',
        capacity='q_matching_for_low_rank_subset_only_full_diagonal_orthogonal_not_equal_capacity',
        comparisons=[list(x) for x in COMPARISONS],
        interaction='(RRR_free-RRR_constrained)-(PCA_free-PCA_constrained)',
        bootstrap_iterations=10000, bootstrap_seed=7341618, reference_margin=.01,
        pilot_gate='complete_3416_all_arms_and_report_technical_acceptance_not_effect_size',
        reuse='accepted_reference_predictions_and_terminal_train_cache_by_digest',
        resource_failure='needs_review_no_silent_compression',
        scope_limit='affine_does_not_imply_bijection_or_physiological_causality')


def validate(spec):
    if spec.get('schema')!=VERSION or spec.get('test_access') is not False:
        raise ValueError('Unsealed affine family')
    if spec.get('protocol')!=protocol(spec.get('scope')) or spec.get('host') not in matrix():
        raise ValueError('Unregistered affine configuration')
    if spec['host']['seed']!=3416 and not spec.get('pilot'):
        raise ValueError('Matched first-seed acceptance required')


def counts():
    return dict(hosts=81, scopes=2, new_case_tasks=162, new_method_views=810,
                new_pure_missing_views=1620, additional_neural_training=0,
                reference_method_views_reused=486,
                note='One case contains five fits per missing pattern; progressive fits contain variable selected sites.')
