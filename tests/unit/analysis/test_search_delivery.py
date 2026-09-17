import numpy as np
import pytest
from look.analysis.search_delivery import validate_predictions

def test_delivery_pairing_rejects_order_and_duplicates():
    a={'participant_ids':np.array([1,2]),'labels':np.array([0,1])}
    validate_predictions(a,a)
    with pytest.raises(ValueError):validate_predictions(a,dict(a,participant_ids=np.array([2,1])))
    b=dict(a,participant_ids=np.array([1,1]))
    with pytest.raises(ValueError):validate_predictions(b,b)
