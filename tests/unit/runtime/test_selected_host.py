import pytest
import torch
from look.runtime.host_checkpoint import atomic_save, read_selected, save_selected


def test_current_selected_preserves_buffers_and_extra_state(tmp_path):
    model = torch.nn.BatchNorm1d(3)
    path = tmp_path / 'best.pt'
    save_selected(path, model=model, identity='case', epoch=2, node_ids=[(1,'x')],
                  sampling_state={'counter': 7})
    state = read_selected(path, identity='case', node_ids=[(1,'x')])
    target = torch.nn.BatchNorm1d(3)
    target.load_state_dict(state['model'], strict=True)
    assert state['sampling_state'] == {'counter': 7}
    for key, tensor in model.state_dict().items():
        assert torch.equal(target.state_dict()[key], tensor)
        assert target.state_dict()[key].dtype == tensor.dtype


@pytest.mark.parametrize('change', [
    {'framework_api': 'V4'}, {'framework_api': None}, {'identity': 'other'},
    {'node_ids': [(2,'x')]}, {'epoch': -1}, {'optimizer': {}},
    {'schema': 'optimizer_boundary_v2'},
])
def test_rejects_unconverted_or_wrong_selected(tmp_path, change):
    path = tmp_path / 'best.pt'
    state = save_selected(path, model=torch.nn.Linear(2,1), identity='case',
                          epoch=2, node_ids=[(1,'x')])
    atomic_save(path, dict(state, **change))
    with pytest.raises(ValueError, match='Current V5 selected host'):
        read_selected(path, identity='case', node_ids=[(1,'x')])
