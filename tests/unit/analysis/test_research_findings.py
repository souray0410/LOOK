from look.analysis.research_findings import summarize


def test_no_models_is_not_completed(tmp_path):
    report=summarize([],tmp_path)
    assert not report['study_complete']
    assert report['complete_diseases']==0
