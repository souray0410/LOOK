"""Fit train-only PCA from a complete, site-sharded formal-V5 cache.

This is a one-time migration stage.  It refuses partial caches and verifies each
selected shard before it contributes to either moments or PCA fitting.
"""
from __future__ import annotations
import argparse, json, pathlib, time
from collections.abc import Callable, Iterator
import torch
from look.methods.joint import correction_sites
from look.methods.operator import downsample_flatten, fit_complete_pca, forward_with_look
from look.runtime.provenance import write_json_atomic
from look.runtime.state import file_sha256, stable_hash
from look.studies.v5_project_full_replay import FRAMEWORK, MODELS, _parent, read
from look.studies.v5_project_feature_replay import datasets
from look.studies.v5_project_feature_materialize_sharded import participant_sha

SCHEMA = "look_formal_v5_train_feature_materialization_v2_site_sharded"

def _chunks(cache: pathlib.Path, total: int) -> list[pathlib.Path]:
    chunks = cache / "chunks"
    rows = sorted(p for p in chunks.iterdir() if p.is_dir() and p.name.isdigit())
    if len(rows) != total or [p.name for p in rows] != [f"{i:06d}" for i in range(total)]:
        raise ValueError("Complete contiguous cache required")
    if any(p.is_dir() for p in chunks.glob("*.partial")):
        raise ValueError("Incomplete cache directory present")
    return rows

def validate_complete_cache(cache: pathlib.Path, expected_identity: dict | None = None):
    cache = pathlib.Path(cache).resolve()
    accepted = read(cache / "accepted.json")
    identity = read(cache / "identity.json")
    if accepted.get("schema") != SCHEMA or accepted.get("state") != "accepted_complete":
        raise ValueError("Accepted complete sharded cache required")
    if accepted.get("completed_batches") != accepted.get("total_batches"):
        raise ValueError("Cache cursor is incomplete")
    if stable_hash(identity) != accepted.get("identity_sha256"):
        raise ValueError("Cache identity receipt mismatch")
    if identity.get("framework_commit") != FRAMEWORK or identity.get("models_commit") != MODELS:
        raise ValueError("Formal V5 double pin required")
    if identity.get("test_access") is not False or accepted.get("test_access") is not False:
        raise ValueError("Train-only cache required")
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("Cache identity differs from requested replay")
    rows = _chunks(cache, int(accepted["total_batches"]))
    sites = list(identity["sites"])
    participant_count = 0
    for index, path in enumerate(rows):
        receipt = read(path / "receipt.json")
        if receipt.get("batch") != index or receipt.get("sites") != sites or receipt.get("test_access") is not False:
            raise ValueError("Chunk receipt identity mismatch")
        if set(receipt.get("files", {})) != set(sites):
            raise ValueError("Chunk does not contain every site")
        participant_ids = receipt.get("participant_ids", [])
        if receipt.get("participants_sha256") != participant_sha(participant_ids):
            raise ValueError("Chunk participant identity mismatch")
        participant_count += len(participant_ids)
    if participant_count != accepted.get("participants_total"):
        raise ValueError("Participant cursor mismatch")
    return identity, accepted, rows

def cached_feature_factory(rows: list[pathlib.Path], site: str, factor: int) -> Callable[[], Iterator]:
    def features():
        for path in rows:
            receipt = read(path / "receipt.json")
            item = receipt["files"][site]
            shard = path / item["path"]
            if shard.parent != path or shard.is_symlink() or file_sha256(shard) != item["sha256"]:
                raise ValueError("Site shard hash mismatch")
            feature = torch.load(shard, map_location="cpu", weights_only=True)
            flat, down_shape = downsample_flatten(feature, factor)
            yield flat, tuple(feature.shape[1:]), down_shape
    return features

def execute(*, source_run, checkpoint, cache, output, inputs_factory):
    source_run = pathlib.Path(source_run).resolve(); checkpoint = pathlib.Path(checkpoint).resolve()
    cache = pathlib.Path(cache).resolve(); output = pathlib.Path(output).resolve()
    spec = read(source_run / "spec.json"); source_accepted = read(source_run / "host/accepted.json")
    expected = {
        "schema": "look_formal_v5_train_feature_cache_v2_site_sharded",
        "framework_commit": FRAMEWORK, "models_commit": MODELS,
        "source_spec_sha256": file_sha256(source_run / "spec.json"),
        "target_checkpoint_sha256": file_sha256(checkpoint),
        "train": None, "sites": None, "test_access": False,
    }
    data, cohort = datasets(source_run, inputs_factory, seed=spec["seed"])
    expected["train"] = cohort["roles"]["train"]
    identity, accepted, rows = validate_complete_cache(cache)
    expected["sites"] = identity["sites"]
    if identity != expected: raise ValueError("Cache is not the requested source/checkpoint/cohort")
    parents = [_parent(spec["parents"][k]["path"], spec["parents"][k]["manifest_sha256"]) for k in ("first", "second")]
    from look.models.native_host import build_native_host
    from look.runtime.host_checkpoint import read_selected
    graph = build_native_host(*parents, spec["position"], device="cpu"); del parents
    ids = [(n.id, n.name) for n in sorted(graph.nodes, key=lambda n:n.id)]
    state = read_selected(checkpoint, identity=source_accepted["identity"], node_ids=ids)
    graph.load_state_dict(state["model"], strict=True); graph.eval()
    # Populate exact member shapes from one original, unaugmented train batch.
    from torch.utils.data import DataLoader
    from look.data.observed_pair import collate_observed
    loader = DataLoader(data["train"], batch_size=spec["training"]["microbatch"], shuffle=False,
        num_workers=0, collate_fn=collate_observed)
    first = next(iter(loader))
    with torch.no_grad(): forward_with_look(graph, first["oct"], first["cfp"], counts=first["counts"])
    sites = correction_sites(graph)
    if sites != identity["sites"]: raise ValueError("Graph/cache site order mismatch")
    cfg = spec["look"]; entries=[]; bank_root=output/"bank"; bank_root.mkdir(parents=True, exist_ok=True)
    stage_identity={"schema":"look_formal_v5_sharded_pca_v1","cache_identity_sha256":stable_hash(identity),
        "cache_acceptance_sha256":file_sha256(cache/"accepted.json"),"factors":cfg["factors"],
        "max_rank":cfg["max_rank"],"sites":sites,"test_access":False}
    identity_path=output/"identity.json"
    if identity_path.exists() and read(identity_path)!=stage_identity: raise ValueError("PCA stage identity changed")
    write_json_atomic(stage_identity,identity_path)
    for site in sites:
        sample=torch.load(rows[0]/read(rows[0]/"receipt.json")["files"][site]["path"],map_location="cpu",weights_only=True)
        factors=[1] if sample.ndim==2 else list(dict.fromkeys(cfg["factors"]))
        for factor in factors:
            path=bank_root/f"{site}_x{factor}.pt"; meta=path.with_suffix(".json")
            source_id=f"{stable_hash(stage_identity)[:16]}/{site}_x{factor}"
            if path.exists() or meta.exists():
                from look.methods.operator import FullFeaturePCA
                m=read(meta)
                if m.get("sha256")!=file_sha256(path) or m.get("source_id")!=source_id: raise ValueError("Existing PCA entry invalid")
                basis=FullFeaturePCA.load(path)
                if basis.source_id!=source_id: raise ValueError("Existing PCA source changed")
            else:
                basis=fit_complete_pca(graph, (), site, factor, cfg["max_rank"], torch.device("cpu"), source_id,
                    strict_rank=True, feature_factory=cached_feature_factory(rows,site,factor))
                basis.save(path); write_json_atomic({"source_id":source_id,"sha256":file_sha256(path),
                    "samples":basis.sample_count,"test_access":False},meta)
            entries.append({"site":site,"factor":factor,"path":str(path.relative_to(output)),"sha256":file_sha256(path),"source_id":source_id})
            write_json_atomic({"state":"running","completed_entries":len(entries),"test_access":False},output/"status.json")
    receipt={"schema":"look_formal_v5_sharded_pca_receipt_v1","state":"accepted","identity_sha256":stable_hash(stage_identity),
        "entries":entries,"train_participants":accepted["participants_total"],"completed_at":time.time(),"test_access":False}
    write_json_atomic(receipt,output/"accepted.json"); return receipt

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--source-run",required=True);p.add_argument("--checkpoint",required=True)
    p.add_argument("--cache",required=True);p.add_argument("--output",required=True);a=p.parse_args(argv)
    from expanded.native import Inputs
    print(json.dumps(execute(source_run=a.source_run,checkpoint=a.checkpoint,cache=a.cache,output=a.output,inputs_factory=Inputs)),flush=True)
if __name__=="__main__": main()
