import numpy as np
import pytest
from look.evaluation.correction_gate import fit_gate, apply_gate, gated_bank


def test_independent_switches_ties_and_no_test_reselection():
    y=np.array([0,0,1,1]);good=np.eye(2)[y];bad=np.eye(2)[1-y]
    for name,candidate,host,enabled in [('gain',good,bad,True),('loss',bad,good,False),('tie',good,good,False)]:
        g=fit_gate(y,candidate,host,data_role='development',identity='host1',method=name,pattern='cfp_missing')
        assert g['enabled'] is enabled
        assert np.array_equal(apply_gate(candidate,host,g,identity='host1',method=name,pattern='cfp_missing'),good)
        assert gated_bank([1,2],g,identity='host1',method=name,pattern='cfp_missing')==([1,2] if enabled else [])
        # Test-like predictions reverse the benefit: decision remains frozen.
        assert np.array_equal(apply_gate(host,candidate,g,identity='host1',method=name,pattern='cfp_missing'),bad if name!='tie' else good)
        with pytest.raises(ValueError):apply_gate(candidate,host,g,identity='other',method=name,pattern='cfp_missing')
        with pytest.raises(ValueError):fit_gate(y,candidate,host,data_role='test',identity='host1',method=name,pattern='cfp_missing')
        g['enabled']=not enabled
        with pytest.raises(ValueError):apply_gate(candidate,host,g,identity='host1',method=name,pattern='cfp_missing')
