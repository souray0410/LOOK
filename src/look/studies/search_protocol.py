"""Weekly search-policy amendment; legacy all-start research remains archived."""
from look.studies.suffix_protocol import sites, PATTERNS
VERSION='look_search_policy_v1'


def protocol():
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
    if spec.get('schema')!=VERSION or spec.get('protocol')!=protocol() or spec.get('test_access') is not False:
        raise ValueError('Unregistered search policy or test access')
    h=spec['host'];ordered=sites(h['architecture'],h['position'])
    if h['seed'] not in (3416,3417,3418) or spec.get('mode') not in ('greedy','best_forward'):
        raise ValueError('Invalid seed or search mode')
    if spec['candidate_sites']!=ordered or spec['eligible_sites']!=ordered:
        raise ValueError('Search requires all canonical candidate sites')
    if h['seed']!=3416 and not spec.get('pilot'):raise ValueError('Matched 3416 acceptance required')
