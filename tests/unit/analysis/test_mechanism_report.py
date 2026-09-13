import numpy as np
from look.analysis.mechanism_report import overlap_bootstrap,classify,comparison_registry


def test_shared_people_and_zero_variance():
    views=[dict(participant_ids=['a','b','c','d'],labels=[0,1,0,1],predictions=[0,0,0,1]),
           dict(participant_ids=['a','b','c','d'],labels=[0,1,0,1],predictions=[0,0,0,1]),
           dict(participant_ids=['a','b','c','d'],labels=[1,0,1,0],predictions=[1,0,0,0])]
    stats=overlap_bootstrap(views,[[1,-1,0],[1,0,-1]],['a','b'],100,3)
    assert stats['union_participants']==4
    assert stats['contrasts'][0]['zero_variance']
    assert stats['contrasts'][0]['global_simultaneous_95'] is None
    assert classify([.011,.03])=='substantial_improvement'
    assert classify([-.001,.002])=='practically_close'
    assert classify([-.02,.02])=='unresolved'
    assert 'original' in comparison_registry()['families']


def test_reused_comparison_retains_all_statistical_families():
    import numpy as np
    from look.analysis.mechanism_report import overlap_bootstrap
    a=dict(participant_ids=np.array(['a','b','c','d']),labels=np.array([0,1,0,1]),predictions=np.array([0,1,1,1]))
    b=dict(a,predictions=np.array([0,0,0,1]))
    result=overlap_bootstrap([a,b],[[1,-1]],[['original','regime']],iterations=100)
    assert set(result['contrasts'][0]['all_family_simultaneous_95'])=={'original','regime'}
