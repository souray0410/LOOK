"""Expanded UKB suffix starts: fixed hosts, independent greedy trajectories."""
VERSION = 'look_ukb_suffix_starts_v1'
PATTERNS = ('oct_missing', 'cfp_missing')


def sites(architecture, position):
    from look.models.native_host import STAGE_MAP, STARTS
    stages = STAGE_MAP[architecture]; cut = STARTS[position]
    return (['joint_input'] + ['joint_'+s for s in stages[:cut+1]] +
            ['fusion_'+s for s in stages[cut:]] + ['fusion_participant_feature'])


def protocol():
    return dict(schema=VERSION, test_access=False, patterns=list(PATTERNS),
        method='original_PCA_residual_greedy', starts='all_canonical_sites',
        earlier_sites='disabled; independent refit from each allowed start',
        reuse='frozen_host_and_complete_train_PCA; first_and_last_from_original_case',
        selection='dev_macro_f1_desc_then_start_ordinal_asc_per_missing_direction',
        selection_scope='all_starts_complete_before_selection; never_per_missing_ratio',
        factors_and_dimensions='unchanged_parent_case_grid', bootstrap_iterations=10000,
        seed_gate='all_3416_starts_technically_accepted_regardless_of_scores',
        resume='accepted_node_decisions; interrupted_unsaved_node_recomputed',
        scope='same_UKB_cohort; not_small_cohort_reuse_or_independent_test')


def validate(spec):
    if spec.get('schema') != VERSION or spec.get('protocol') != protocol() or spec.get('test_access') is not False:
        raise ValueError('Unregistered suffix protocol or test access')
    h=spec['host']; ordered=sites(h['architecture'],h['position']); n=spec['start_ordinal']
    if h['seed'] not in (3416,3417,3418) or not 1 < n < len(ordered):
        raise ValueError('Only internal starts require new fitting; endpoints are aliases')
    if spec['candidate_sites'] != ordered or spec['eligible_sites'] != ordered[n-1:]:
        raise ValueError('Suffix order or disabled prefix changed')
    if h['seed'] != 3416 and not spec.get('pilot'):
        raise ValueError('Complete first-seed technical acceptance required')
