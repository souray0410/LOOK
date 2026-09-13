"""Finite, source-bound supplement to the accepted 81-host UKB study."""
from itertools import product
from look.runtime.state import stable_hash

VERSION = 'look_ukb_mechanisms_v1'
DISEASES = ('glaucoma', 'cataract', 'macular_degeneration')
ARCHITECTURES = ('resnet50', 'densenet121', 'swin_b')
POSITIONS = ('middle', 'deep', 'features')
SEEDS = (3416, 3417, 3418)
PATTERNS = ('oct_missing', 'cfp_missing')
FRACTIONS = (.01, .05, .10, .25, .50, 1.)
TRAINING = dict(epochs=100, patience=15, minimum_epochs=8, warmup_epochs=5,
    microbatch=16, effective_batch=128, pretrained_lr=1e-5, new_layer_lr=1e-4,
    weight_decay=1e-4, clip=5., precision='fp32', num_workers=0,
    loss='unweighted_cross_entropy', primary_metric='macro_f1')
MLP = dict(lr=1e-3, weight_decay=1e-4, batch=256, epochs=200,
           patience=20, relative_improvement=1e-4, fitting_fraction=.9)


def matrix():
    """Logical positions are not runs: only ready content-bound tasks are reserved."""
    tasks = []
    for disease, architecture, position, seed in product(DISEASES, ARCHITECTURES, POSITIONS, SEEDS):
        host = dict(disease=disease, architecture=architecture, position=position, seed=seed)
        for arm in ('continue_complete', 'continue_missing'):
            tasks.append(dict(kind='host_training', arm=arm, host=host))
        for track in ('cfp', 'oct'):
            tasks.append(dict(kind='student_training', arm='distill', track=track, host=host))
        for pattern in PATTERNS:
            for arm in ('independent', 'missing_readout', 'available_readout',
                        'missing_refit', 'available_refit', 'mlp', 'affine_equivalence'):
                tasks.append(dict(kind='correction', arm=arm, pattern=pattern, host=host))
            for repeat in (0, 1, 2):
                tasks.append(dict(kind='correction', arm='shuffle', repeat=repeat, pattern=pattern, host=host))
            for fraction, repeat, selection in product(FRACTIONS, (0, 1, 2), ('fixed', 'reselect')):
                # Stable row order and identical full data: repeats at 100% are aliases.
                if fraction == 1. and repeat != 0:
                    continue
                tasks.append(dict(kind='sample_curve', arm=selection, fraction=fraction,
                                  repeat=repeat, pattern=pattern, host=host))
    for disease, architecture, seed, track in product(DISEASES, ARCHITECTURES, SEEDS, ('cfp', 'oct')):
        tasks.append(dict(kind='student_training', arm='ce', track=track,
            host=dict(disease=disease, architecture=architecture, seed=seed)))
    for task in tasks:
        task['id'] = stable_hash(dict(protocol=VERSION, **task))
    assert len({t['id'] for t in tasks}) == len(tasks)
    assert sum(t['kind'].endswith('_training') for t in tasks) == 378
    return tasks


def protocol():
    tasks = matrix()
    return dict(schema=VERSION, test_access=False, training=TRAINING, mlp=MLP,
        teacher_temperature=2., distillation_weights=dict(ce=1., kl=1.),
        patterns=list(PATTERNS), missing_probabilities=[1/3]*3,
        fractions=list(FRACTIONS), subset_repeats=3, base_host_cases=81,
        neural_training_tasks=378, logical_tasks=len(tasks), matrix_sha256=stable_hash(tasks),
        seed=7341618, bootstrap_iterations=10000, reference_margin=.01,
        external_candidates=['MMTM', 'EyeMoSt+', 'EDRL'],
        external_training='blocked_until_separate_adaptation_protocol',
        current_split_changes=False, deployment='source_fit_frozen_target')


def validate_task(spec):
    if spec.get('schema') != VERSION or spec.get('test_access') is not False:
        raise ValueError('Unknown supplement protocol or test access')
    task = spec['task']
    identity = stable_hash(dict(protocol=VERSION, **{k:v for k,v in task.items() if k != 'id'}))
    if task['id'] != identity or not any(t == task for t in matrix()):
        raise ValueError('Task is outside the fixed matrix')
    if spec['protocol'] != protocol():
        raise ValueError('Scientific protocol changed')
