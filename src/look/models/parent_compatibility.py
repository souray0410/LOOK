"""Explicit inference-only equivalence for earlier V4 registry bundles.

The parent receipt is never rewritten. Training resumes always use its complete
original runtime. This exception requires identical graph/operator source bytes,
strict full state/node loading, followed by complete saved development replay.
"""

def verify_inference_equivalence(spec):
    from mhd_framework.models.artifacts import runtime_source_sha256
    current=runtime_source_sha256();original=spec['framework']['source_sha256']
    allowed={'3559caa8d596d4438533a69d39d8a2c32eb21e46','56065555e96d39711871db5eba0367c5db84e3fd','c0a27abb3e0f2153bfd273b1d05d5b7dae9784f0'}
    if spec['framework'].get('commit') not in allowed:raise ValueError('Unreviewed parent framework release')
    name=spec['model']['name']
    implementation={'resnet50':'models/resnet.py','densenet121':'models/densenet.py','swin_b':'models/vision_transformer.py'}
    if name not in implementation:raise ValueError('Unreviewed inference architecture')
    paths={'core.py','utils.py','__init__.py','models/graph.py','models/resnet.py',implementation[name]}
    if name=='swin_b':paths.add('models/vision_transformer.py')
    if any(not original.get(p) or original[p]!=current.get(p) for p in paths):
        raise ValueError('Native inference operator source changed')
    return dict(schema='look_inference_equivalence_v1',original_framework=spec['framework']['commit'],
                inference_source_sha256={p:original[p] for p in sorted(paths)},
                scope='inference_only_requires_strict_state_and_full_development_replay',
                ignored_non_inference_registry_files=['models/__init__.py','models/artifacts.py','models/training.py'])
