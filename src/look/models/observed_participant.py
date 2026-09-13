"""A complete native graph with explicit observed-eye participant aggregation.

This is an independent-parent consumer, not the fused LOOK host. It retains the
native graph's state keys and head, and does not create correction locations.
"""
import torch
from torch import nn


class ObservedParticipantModel(nn.Module):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph

    def to(self, *args, **kwargs):
        # Module.to recurses through _apply; it does not call the nested graph's
        # to override, which also migrates MHD messages and topology tensors.
        self.graph.to(*args, **kwargs)
        return self

    def cpu(self):
        return self.to(torch.device('cpu'))

    def cuda(self, device=None):
        return self.to(torch.device('cuda', torch.cuda.current_device() if device is None else device))

    def forward(self, x, counts):
        if (not counts or any(type(n) is not int or n not in (1, 2) for n in counts)
                or sum(counts) != len(x)):
            raise ValueError("Each participant must have one or two observed eyes")
        features = self.graph.forward_until("features", x)
        pooled = torch.stack([part.mean(0) for part in features.split(counts)])
        return self.graph.forward_from("features", pooled)


def verify_pair_rows(first, second):
    """Both modalities must identify the same participant, eye and label order.

    Input rows stay in authorized storage; the function returns aggregate counts.
    No intersection/reordering or bilateral duplication is silently performed.
    """
    if not first or len(first) != len(second):
        raise ValueError("Paired manifest lengths differ or are empty")
    seen = set()
    counts = {1: 0, 2: 0}
    for left, right in zip(first, second):
        for key in ("id", "eyes", "label"):
            if left[key] != right[key]:
                raise ValueError("Paired identity/eye order/label mismatch")
        if left["id"] in seen:
            raise ValueError("Repeated participant")
        seen.add(left["id"])
        eyes = left["eyes"]
        if len(eyes) not in (1, 2) or len(set(eyes)) != len(eyes):
            raise ValueError("Invalid observed-eye ownership")
        counts[len(eyes)] += 1
    return dict(participants=len(first), one_eye=counts[1], two_eyes=counts[2])
