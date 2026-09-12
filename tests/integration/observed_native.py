"""GPU native-consumer equivalence; not a fused-host or LOOK efficacy test.

Pass the pinned independent trainer on PYTHONPATH. Data is the accepted small
train/development runtime fixture, never the expanded test partition.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from look.models.observed_participant import ObservedParticipantModel
from look.runtime.state import atomic_write_json, file_sha256


def run(data, output):
    from expanded.native import ParticipantModel
    from mhd_framework.models import create_model
    from runtime.checkpoint import capture_rng, restore_rng
    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((data / "accepted.json").read_text())
    assert receipt["status"] == "accepted" and receipt["test_used"] is False
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    results = []
    for name in ("resnet50", "densenet121", "swin_b"):
        config = dict(name=name, spatial_dims=2, views=1)
        torch.manual_seed(3416)
        native = ParticipantModel(config, device="cuda:0")
        adapted = ObservedParticipantModel(create_model(config, device="cuda:0"))
        adapted.load_state_dict(native.state_dict(), strict=True)
        native.eval(); adapted.eval()
        nodes = [(n["id"], n["name"]) for n in native.graph.describe_nodes()]
        assert nodes == [(n["id"], n["name"]) for n in adapted.graph.describe_nodes()]
        for track in ("cfp_2d", "oct_bscan_2d"):
            path = data / track / "train.json"
            assert file_sha256(path) == receipt["manifest_sha256"][track]["train"]
            samples = json.loads(path.read_text())["samples"][:2]
            arrays = []
            for index, row in enumerate(samples):
                image = path.parent / row["file"]
                assert file_sha256(image) == row["sha256"]
                a = np.load(image, allow_pickle=False)
                arrays.append(a[:1 if index == 0 else 2])
            counts = [len(a) for a in arrays]
            assert counts == [1, 2], "Fixture must exercise actual single and bilateral pooling"
            x = torch.from_numpy(np.concatenate(arrays).copy()).float().cuda() / 255
            if x.shape[1] == 1: x = x.repeat(1, 3, 1, 1)
            x.requires_grad_(True); other = x.detach().clone().requires_grad_(True)
            native.zero_grad(set_to_none=True); adapted.zero_grad(set_to_none=True)
            state = capture_rng(); first = native(x, counts)
            restore_rng(state); second = adapted(other, counts)
            assert torch.equal(first, second)
            first.square().sum().backward(); second.square().sum().backward()
            assert torch.equal(x.grad, other.grad)
            for (n, p), (m, q) in zip(native.named_parameters(), adapted.named_parameters()):
                assert n == m and (p.grad is None) == (q.grad is None)
                if p.grad is not None: assert torch.equal(p.grad, q.grad), n
            # Identical meaningful updates in both wrappers retain full state.
            left = torch.optim.SGD(native.parameters(), lr=.001)
            right = torch.optim.SGD(adapted.parameters(), lr=.001)
            left.step(); right.step()
            assert all(torch.equal(v, adapted.state_dict()[k]) for k, v in native.state_dict().items())
            results.append(dict(model=name, track=track, counts=counts,
                                logits_exact=True, gradients_exact=True, update_exact=True,
                                original_node_ids=True, manifest_sha256=file_sha256(path)))
            del first, second, x, other, left, right
        torch.save(adapted.state_dict(), output / (name + ".pt"))
        native.load_state_dict(torch.load(output / (name + ".pt"), weights_only=True), strict=True)
        del native, adapted
        torch.cuda.empty_cache()
    atomic_write_json(dict(status="accepted", test_used=False, formal_updates=0,
        scope="small_real_inputs_native_consumer_equivalence_not_LOOK_or_A100_admission",
        initialization="random", results=results), output / "accepted.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True); args = parser.parse_args()
    run(args.data, args.output)
