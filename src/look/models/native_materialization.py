"""Project-owned selected native models, without modifying independent runs.

This format intentionally copies selected inference evidence, not optimizer resume
state. It retains the original acceptance receipt as source evidence, not as a
claim that every native-training artifact exists in the project copy.
"""
import fcntl
import json
import os
import shutil
import tempfile
from pathlib import Path

from look.runtime.state import atomic_write_json, file_sha256, stable_hash


def read(path):
    return json.loads(Path(path).read_text())


def verify_selected(root, expected_spec=None):
    root = Path(root).resolve()
    manifest = read(root / "selected_artifact.json")
    if (manifest.get("schema") != "look_selected_native_v1" or
            manifest.get("test_access") is not False or
            manifest.get("complete_training_resume") is not False):
        raise ValueError("Unknown selected-artifact contract")
    required = {"best.pt", "spec.json", "history.json", "development_predictions.npz",
                "source_accepted.json"}
    if not required.issubset(manifest["files"]):
        raise ValueError("Selected model evidence is incomplete")
    for name, checksum in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != checksum:
            raise ValueError("Selected model artifact changed")
    spec = read(root / "spec.json")
    receipt = read(root / "source_accepted.json")
    import hashlib
    identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    if (spec.get("test_used") is not False or receipt.get("test_used") is not False or
            receipt.get("identity") != identity or receipt.get("status") != "accepted"):
        raise ValueError("Selected model identity or test provenance mismatch")
    if expected_spec is not None and spec != expected_spec:
        raise ValueError("Different parent training specification")
    if any(receipt["files"][name] != manifest["files"][name]
           for name in required - {"source_accepted.json"}):
        raise ValueError("Selected files disagree with native acceptance")
    return manifest, spec, receipt


def materialize_selected(source, destination_root, expected_spec, verify_completion):
    source = Path(source).resolve()
    verify_completion(source, expected_spec)
    receipt_path = source / "accepted.json"
    receipt_sha = file_sha256(receipt_path)
    receipt = read(receipt_path)
    if (expected_spec.get("test_used") is not False or receipt.get("test_used") is not False or
            read(source / "spec.json") != expected_spec):
        raise ValueError("Parent specification/test provenance is not eligible")
    identity = dict(source_run=str(source), accepted_sha256=receipt_sha,
                    spec_sha256=receipt["files"]["spec.json"],
                    best_sha256=receipt["files"]["best.pt"])
    destination_root = Path(destination_root).resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    key = stable_hash(identity)
    target = destination_root / key
    with (destination_root / (key + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            record, _, _ = verify_selected(target, expected_spec)
            if record["source"] != identity:
                raise ValueError("Conflicting parent-copy identity")
            return target
        temporary = Path(tempfile.mkdtemp(prefix=key + ".", suffix=".partial", dir=destination_root))
        try:
            names = ["best.pt", "spec.json", "history.json", "development_predictions.npz"]
            names += [n for n in receipt["files"] if n.startswith("provenance/")]
            copied = {}
            for name in names:
                src = (source / name).resolve()
                dst = temporary / name
                if not src.is_relative_to(source) or not dst.resolve().is_relative_to(temporary):
                    raise ValueError("Unsafe native artifact path")
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                checksum = file_sha256(dst)
                if checksum != receipt["files"][name]:
                    raise ValueError("Source changed while copying")
                with dst.open("rb") as f:
                    os.fsync(f.fileno())
                copied[name] = checksum
            shutil.copyfile(receipt_path, temporary / "source_accepted.json")
            if file_sha256(temporary / "source_accepted.json") != receipt_sha:
                raise ValueError("Native receipt changed while copying")
            with (temporary / "source_accepted.json").open("rb") as f:
                os.fsync(f.fileno())
            copied["source_accepted.json"] = receipt_sha
            manifest = dict(schema="look_selected_native_v1", source=identity, files=copied,
                            test_access=False, complete_task_model=True,
                            complete_training_resume=False, selected_prediction_replay=False,
                            source_acceptance_verification="native_verifier_and_copy_hashes")
            atomic_write_json(manifest, temporary / "selected_artifact.json")
            verify_selected(temporary, expected_spec)
            os.replace(temporary, target)
            fd = os.open(destination_root, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            return target
        except Exception:
            # Preserve the failed attempt for diagnosis; it is never returned or
            # mistaken for a valid model by the manifest verifier.
            raise


def load_selected(root, graph_factory, expected_spec=None, device="cpu"):
    """Strict complete state loading; numerical development replay is separate."""
    import torch
    from mhd_framework.models.artifacts import verify_runtime
    from look.models.observed_participant import ObservedParticipantModel
    _, spec, receipt = verify_selected(root, expected_spec)
    verify_runtime(spec["framework"])
    graph = graph_factory(spec["model"], device=device)
    actual_nodes = [[node["id"], node["name"]] for node in graph.describe_nodes()]
    if actual_nodes != [list(row) for row in receipt["node_ids"]]:
        raise ValueError("Native MHD node identity mismatch")
    model = ObservedParticipantModel(graph)
    state = torch.load(Path(root) / "best.pt", map_location="cpu", weights_only=False)
    if state["identity"] != receipt["identity"] or state["epoch"] != receipt["best_epoch"]:
        raise ValueError("Selected checkpoint identity/epoch mismatch")
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    return model
