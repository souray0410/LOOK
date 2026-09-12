import copy
import pytest
import torch
from torch import nn

from look.models.observed_participant import ObservedParticipantModel, verify_pair_rows


class Graph(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 5)
        self.head = nn.Linear(5, 2)

    def forward_until(self, node, x):
        assert node == "features"
        return self.encoder(x).tanh()

    def forward_from(self, node, x):
        assert node == "features"
        return self.head(x)


def test_observed_eye_mean_and_backward_match_direct_reference():
    torch.manual_seed(17)
    graph = Graph(); ref = copy.deepcopy(graph)
    model = ObservedParticipantModel(graph)
    x = torch.randn(3, 4, requires_grad=True)
    rx = x.detach().clone().requires_grad_(True)
    actual = model(x, [1, 2])
    features = ref.encoder(rx).tanh()
    expected = ref.head(torch.stack([features[0], features[1:].mean(0)]))
    assert torch.equal(actual, expected)
    actual.square().sum().backward(); expected.square().sum().backward()
    assert torch.equal(x.grad, rx.grad)
    for p, q in zip(graph.parameters(), ref.parameters()):
        assert torch.equal(p.grad, q.grad)
    restored = ObservedParticipantModel(Graph())
    restored.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(restored(x, [1, 2]), actual)
    for counts in ([2, 2], [0, 3], [True, 2], []):
        with pytest.raises(ValueError): model(x, counts)


def test_pair_identity_label_and_eye_order_must_match():
    rows = [dict(id="synthetic_a", eyes=["L"], label=0),
            dict(id="synthetic_b", eyes=["L", "R"], label=1)]
    assert verify_pair_rows(rows, copy.deepcopy(rows)) == dict(participants=2, one_eye=1, two_eyes=1)
    for field, value in (("id", "different"), ("label", 0), ("eyes", ["R", "L"])):
        other = copy.deepcopy(rows); other[1][field] = value
        with pytest.raises(ValueError): verify_pair_rows(rows, other)
    duplicated = [rows[0], rows[0]]
    with pytest.raises(ValueError): verify_pair_rows(duplicated, duplicated)
