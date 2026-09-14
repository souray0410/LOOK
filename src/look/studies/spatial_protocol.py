"""Matched spatial preprocessing comparison; technical pilot never selects a winner."""
from itertools import product
from look.runtime.state import stable_hash

VERSION='look_spatial_v1'
ROUTES=('interpolate','direct','average_pool')
SEEDS=(3416,3417,3418)


def protocol():
    return dict(schema=VERSION,test_access=False,routes=list(ROUTES),seeds=list(SEEDS),
        factors={'interpolate':[4,8,16],'direct':[1],'average_pool':[4,8,16]},
        latent_dims=[8,16,32,64,96,128,192,256,384,512],max_rank=512,
        host_states=['original'],patterns=['oct_missing','cfp_missing'],
        selection='independent_same_greedy_rules_per_route',return_operator='bilinear_align_corners_false',
        replication_gate='all_three_routes_technical_acceptance_and_matched_report_no_score_threshold',
        bootstrap_iterations=10000,bootstrap_seed=7341618,reference_margin=.01,
        rank_policy='same_feasible_rank_per_site_all_routes_explicit_manifest',additional_neural_training=False)


def matrix():
    return [dict(disease=d,architecture=a,position=p,seed=s) for d,a,p,s in product(
        ('glaucoma','cataract','macular_degeneration'),('resnet50','densenet121','swin_b'),
        ('middle','deep','features'),SEEDS)]


def validate(spec):
    if spec.get('schema')!=VERSION or spec.get('test_access') is not False or spec.get('protocol')!=protocol():
        raise ValueError('Unregistered spatial protocol or test access')
    if spec['host'] not in matrix():raise ValueError('Unknown spatial host')
    if spec['host']['seed']!=3416 and not spec.get('pilot'):
        raise ValueError('Replication requires the complete matched pilot receipt')
