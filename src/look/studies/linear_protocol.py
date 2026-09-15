"""Additive conditional mechanism study; never selects favorable pilot effects."""
from look.methods.linear_operator import VERSION,ARMS
from look.studies.spatial_protocol import matrix


def protocol():
    return dict(schema=VERSION,arms=list(ARMS),test_access=False,additional_neural_training=False,
        seeds=[3416,3417,3418],patterns=['oct_missing','cfp_missing'],
        host_states=['original'],template='accepted_standard_LOOK_per_pattern',
        fixed=['host','data_order','normalization','sites','spatial_method','factor','rank','ridge_lambda','return_operator'],
        upstream='sequential_refit_per_arm_at_fixed_sites_no_new_dev_selection',
        budget_axis='common_integer_slope_rank_upper_bound_q_not_equal_capacity',
        native_retention=dict(pca='complete_feature_explained_variance',rrr='regularized_centered_residual_objective_gain'),
        fraction_policy='descriptive_train_only_not_equal_energy_across_methods_not_new_selection',
        penalty='sum_squared_standardized_residual_error_plus_lambda_frobenius_squared',
        comparisons=[['rrr_shared_intercept','shared_pca_ridge'],['residual_rrr','rrr_shared_intercept'],
                     ['residual_rrr','shared_pca_ridge']],
        interpretation='conditional_on_LOOK_selected_configuration_not_independent_best_method',
        resource_failure='needs_resource_review_no_silent_rank_reduction_or_site_skipping',
        replication_gate='complete_matched_3416_acceptance_no_effect_threshold',
        bootstrap_iterations=10000,bootstrap_seed=7341618,reference_margin=.01)


def validate(spec):
    if spec.get('schema')!=VERSION or spec.get('test_access') is not False or spec.get('protocol')!=protocol():
        raise ValueError('Unregistered linear study or test access')
    if spec.get('host') not in matrix():raise ValueError('Unregistered host')
    if spec['host']['seed']!=3416 and not spec.get('pilot'):
        raise ValueError('Complete matched pilot required for replication')
