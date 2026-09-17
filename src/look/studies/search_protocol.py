"""Weekly search-policy amendment; legacy all-start research remains archived."""
from look.studies.suffix_protocol import sites, PATTERNS
VERSION='look_search_policy_v1'
TREE_MODE='positive_forward_tree'
TREE_VERSION='look_positive_forward_tree_v1'
SEARCH_MODES=('greedy','best_forward',TREE_MODE)


def representative_starts(architecture, position):
    ordered = sites(architecture, position)
    # Anatomical/network order fixed before reading scores; median canonical node.
    return [1, 2, (len(ordered)+1)//2, len(ordered)]


def protocol(mode=None):
    if mode == TREE_MODE:
        return dict(schema=TREE_VERSION,test_access=False,patterns=list(PATTERNS),
            main=TREE_MODE,controls=['greedy','best_forward'],baseline='frozen_uncorrected_host',
            search='at_each_prefix_try_every_strictly_downstream_site; keep_site_best_candidate; recurse_all_positive_extensions',
            branch_acceptance='site_best_dev_score_strictly_greater_than_current_prefix; equal_score_rejected',
            pruning='rejected_extensions_have_no_descendants',
            node_order='fixed_candidate_sites_order',
            candidate_ties='lexicographic_candidate_key',
            selection='all_retained_prefixes_including_empty; higher_dev_score; fewer_corrections; lexicographic_position_candidate_path',
            operator='original_complete_train_PCA_and_GCV; unchanged_parent_case_grid',
            seed_gate='matched_3416_technical_acceptance_and_existing_whole_weekly_delivery_not_performance',
            registration='independent_first_seed; legacy_finite_package_unchanged',
            claims='exploratory_dev; optimum_only_over_positive_site_best_tree; no_global_optimum_or_lower_walltime_guarantee')
    if mode not in (None, 'greedy', 'best_forward'):
        raise ValueError('Unregistered search mode')
    return dict(schema=VERSION,test_access=False,patterns=list(PATTERNS),
        main='best_forward',control='greedy',baseline='frozen_uncorrected_host',
        search='all_remaining_sites_on_same_accepted_bank; select_one; strictly_downstream_refit',
        stopping='maximum_remaining_dev_gain_not_positive',
        ties='off; earlier_site; smaller_latent_dim',
        operator='original_complete_train_PCA_and_GCV; unchanged_parent_case_grid',
        seed_gate='matched_3416_technical_acceptance_not_performance',
        all_starts='retained_followup_not_weekly_prerequisite',
        claims='exploratory_dev; no guarantee_of_global_optimum_or_lower_walltime')


def validate(spec):
    if spec.get('schema')!=VERSION or spec.get('protocol')!=protocol(spec.get('mode')) or spec.get('test_access') is not False:
        raise ValueError('Unregistered search policy or test access')
    h=spec['host'];ordered=sites(h['architecture'],h['position'])
    if h['seed'] not in (3416,3417,3418) or spec.get('mode') not in SEARCH_MODES:
        raise ValueError('Invalid seed or search mode')
    if spec['mode']==TREE_MODE and h['seed']!=3416:
        raise ValueError('Tree repeat seeds await registered whole weekly package acceptance; old pilot cannot release them')
    start = spec.get('start_ordinal', 1)
    if start != 1 and (spec['mode'] != 'greedy' or start not in representative_starts(h['architecture'],h['position'])):
        raise ValueError('Only preregistered representative sequential starts are allowed')
    if spec['candidate_sites']!=ordered or spec['eligible_sites']!=ordered[start-1:]:
        raise ValueError('Search candidate sites or suffix changed')
    if h['seed']!=3416 and not spec.get('pilot'):raise ValueError('Matched 3416 acceptance required')
    if type(spec.get('worker_threads', 2)) is not int or spec.get('worker_threads', 2) not in (1, 2):
        raise ValueError('Worker threads must be explicitly registered as one or two')
    if 'latent_dims' in spec and (not isinstance(spec['latent_dims'], list) or len(spec['latent_dims']) != 1
            or type(spec['latent_dims'][0]) is not int or spec['latent_dims'][0] <= 0):
        raise ValueError('One explicit positive latent dimension per serial configuration')
    if 'spatial_factors' in spec and spec['spatial_factors'] != [16]:
        raise ValueError('Only the explicitly authorized fixed16 weekly variant is registered')


def execution_factors(spec, parent_config):
    """Explicit new identity; absent override retains the original scientific grid."""
    validate(spec)
    selected = spec.get('spatial_factors', parent_config['factors'])
    if not set(selected).issubset(parent_config['factors']):
        raise ValueError('Requested factors lack the same-host fitted bases')
    return list(selected)


def execution_latent_dims(spec, parent_config):
    """Finite single-configuration delivery; never change the archived parent grid."""
    validate(spec)
    selected = spec.get('latent_dims', parent_config['latent_dims'])
    if not set(selected).issubset(parent_config['latent_dims']):
        raise ValueError('Requested latent dimension is outside the registered parent grid')
    return list(selected)
