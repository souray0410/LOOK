"""Resumable, train/development-only feature replay for offline V5 PCA rebinding."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from look.data.array_pair import ArrayPair
from look.data.observed_pair import collate_observed
from look.methods.joint import read_site
from look.methods.operator import forward_with_look
from look.runtime.provenance import write_json_atomic
from look.runtime.state import file_sha256, stable_hash
from look.studies.cohort_delivery import make_graph


ROLES = ("train", "development")


def tensor_digest(value: torch.Tensor) -> dict:
    value = value.detach().contiguous().cpu()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest()}


def batch_record(role: str, index: int, participant_ids, graph, sites) -> dict:
    return {"role": role, "batch": index,
            "participants": hashlib.sha256(json.dumps(participant_ids).encode()).hexdigest(),
            "sites": {name: tensor_digest(read_site(graph, name)) for name in sites}}


class ReplayJournal:
    """One immutable JSON record per batch; exact reruns resume, drift fails closed."""

    def __init__(self, root, mode):
        self.root = Path(root) / "records" / mode
        self.root.mkdir(parents=True, exist_ok=True)

    def commit(self, record, *, expected=None):
        if expected is not None and record != expected:
            raise ValueError("Feature replay differs from the reference")
        folder = self.root / record["role"]
        folder.mkdir(exist_ok=True)
        path = folder / f"{record['batch']:06d}.json"
        if path.exists():
            if json.loads(path.read_text()) != record:
                raise ValueError("Existing replay journal record changed")
        else:
            write_json_atomic(record, path)
        return path

    def read(self, role, index):
        path = self.root / role / f"{index:06d}.json"
        if not path.is_file():
            raise ValueError("Reference replay batch is missing")
        return json.loads(path.read_text())


def _source_contract(run_root, manifest, pca_entry_root=None):
    run_root = Path(run_root).resolve(); manifest = Path(manifest).resolve()
    if not manifest.is_file() or not manifest.is_relative_to(run_root):
        raise ValueError("PCA manifest must belong to the immutable source run")
    spec = json.loads((run_root / "spec.json").read_text())
    accepted = json.loads((run_root / "host" / "accepted.json").read_text())
    bank = json.loads(manifest.read_text())
    if spec.get("test_access") is not False or accepted.get("state") != "accepted":
        raise ValueError("Accepted train/development source required")
    if accepted.get("identity") != stable_hash(spec):
        raise ValueError("Source host identity changed")
    for name, digest in accepted["files"].items():
        if file_sha256(run_root / "host" / name) != digest:
            raise ValueError("Source host artifact changed")
    entries = bank.get("entries", [])
    sites = list(dict.fromkeys(row["node"] for row in entries))
    entry_root = Path(pca_entry_root).resolve() if pca_entry_root else manifest.parent
    relocated = [entry_root / Path(row["path"]).name for row in entries]
    if (len(sites) != len(entries) or not sites
            or bank["identity"]["host_best_sha256"] != file_sha256(run_root / "host" / "best.pt")
            or any(path.is_symlink() or not path.resolve().is_relative_to(entry_root)
                   or file_sha256(path) != row["sha256"]
                   for path, row in zip(relocated, entries, strict=True))):
        raise ValueError("PCA source identity or site coverage changed")
    return spec, accepted, bank, sites


def run(mode, run_root, manifest, checkpoint, output, device,
        *, data_root=None, pca_entry_root=None, gpu_budget_bytes=None,
        reference_receipt=None, reference_receipt_sha256=None):
    from mhd_framework import __api_version__
    from mhd_framework.models.artifacts import runtime_source_sha256

    run_root = Path(run_root).resolve(); output = Path(output).resolve()
    if output == run_root or output.is_relative_to(run_root):
        raise ValueError("Replay output must be isolated from the source run")
    spec, accepted, bank, sites = _source_contract(run_root, manifest, pca_entry_root)
    data_root = Path(data_root or spec["data_root"]).resolve()
    if file_sha256(data_root / "accepted.json") != spec["data_audit_sha256"]:
        raise ValueError("Relocated data audit differs from the source spec")
    expected_api = "V4" if mode == "reference" else "V5"
    if __api_version__ != expected_api:
        raise ValueError(f"{mode} requires framework {expected_api}")
    if device.type == "cuda":
        if not gpu_budget_bytes or gpu_budget_bytes <= 0:
            raise ValueError("CUDA replay requires an explicit measured workload budget")
        total = torch.cuda.get_device_properties(device).total_memory
        if gpu_budget_bytes >= total:
            raise ValueError("GPU workload budget must retain transient device headroom")
        torch.cuda.set_per_process_memory_fraction(gpu_budget_bytes / total, device)
    checkpoint = Path(checkpoint).resolve()
    graph = make_graph(spec); node_ids = [(n.id, n.name) for n in sorted(graph.nodes, key=lambda n: n.id)]
    if mode == "reference":
        if checkpoint != (run_root / "host" / "best.pt").resolve():
            raise ValueError("Reference must use the original selected host")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    else:
        from look.runtime.host_checkpoint import read_selected
        state = read_selected(checkpoint, identity=accepted["identity"], node_ids=node_ids)
    graph.load_state_dict(state["model"], strict=True); graph.to(device); graph.eval()
    reference = None
    if mode == "check":
        if not reference_receipt or file_sha256(reference_receipt) != reference_receipt_sha256:
            raise ValueError("Pinned reference receipt required")
        reference = json.loads(Path(reference_receipt).read_text())
        if (reference.get("schema") != "look_node_feature_replay_v1"
                or reference.get("state") != "reference_exported"
                or reference.get("test_access") is not False
                or reference.get("source_checkpoint_sha256") != bank["identity"]["host_best_sha256"]
                or reference.get("original_bank_sha256") != file_sha256(manifest)
                or reference.get("sites") != sites):
            raise ValueError("Reference receipt identity changed")
    torch.set_num_threads(2); torch.manual_seed(spec["seed"]); random.seed(spec["seed"]); np.random.seed(spec["seed"])
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    journal = ReplayJournal(output, mode); counts = {}; batches = {}
    for role in ROLES:
        dataset = ArrayPair(data_root, role, augment=False, seed=spec["seed"])
        counts[role] = len(dataset); batches[role] = 0
        loader = DataLoader(dataset, batch_size=spec["training"]["microbatch"], shuffle=False,
            collate_fn=collate_observed, num_workers=0,
            generator=torch.Generator().manual_seed(spec["seed"]))
        with torch.no_grad():
            for index, batch in enumerate(loader):
                forward_with_look(graph, batch["oct"].to(device), batch["cfp"].to(device), counts=batch["counts"])
                record = batch_record(role, index, batch["participant_id"], graph, sites)
                expected = ReplayJournal(output, "reference").read(role, index) if mode == "check" else None
                journal.commit(record, expected=expected); batches[role] += 1
                write_json_atomic({"mode": mode, "role": role, "batch": index,
                                   "updated_at": time.time(), "test_access": False}, output / "status.json")
    if mode == "check" and (counts != reference["counts"] or batches != reference["batches"]):
        raise ValueError("Replay cohort or batch coverage changed")
    receipt = {"schema": "look_node_feature_replay_v1",
        "state": "reference_exported" if mode == "reference" else "accepted",
        "source_checkpoint_sha256": bank["identity"]["host_best_sha256"],
        "target_checkpoint_sha256": file_sha256(checkpoint),
        "original_bank_sha256": file_sha256(manifest), "sites": sites,
        "counts": counts, "batches": batches, "all_site_values_bitwise_equal": mode == "check",
        "framework_api": __api_version__, "framework_source_sha256": runtime_source_sha256(),
        "imported_look_source_sha256": {name: file_sha256(module.__file__) for name, module in sys.modules.items()
            if name.startswith("look.") and getattr(module, "__file__", None) and str(module.__file__).endswith(".py")},
        "test_access": False, "production_cutover": False, "completed_at": time.time()}
    write_json_atomic(receipt, output / f"{mode}_receipt.json")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("reference", "check"), required=True)
    parser.add_argument("--run-root", required=True); parser.add_argument("--pca-manifest", required=True)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--gpu-budget-bytes", type=int)
    parser.add_argument("--data-root"); parser.add_argument("--pca-entry-root")
    parser.add_argument("--reference-receipt"); parser.add_argument("--reference-receipt-sha256")
    args = parser.parse_args(argv)
    result = run(args.mode, args.run_root, args.pca_manifest, args.checkpoint, args.output,
                 torch.device(args.device), reference_receipt=args.reference_receipt,
                 reference_receipt_sha256=args.reference_receipt_sha256,
                 data_root=args.data_root, pca_entry_root=args.pca_entry_root,
                 gpu_budget_bytes=args.gpu_budget_bytes)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
