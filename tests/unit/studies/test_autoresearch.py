import json
import fcntl
from pathlib import Path

import pytest

from look.runtime.state import file_sha256
from look.studies.native_prerequisites import collect
from look.studies.autoresearch import Controller, discover, replica_spec


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def task(root, name, score=None):
    run = root / "runs" / name
    run.mkdir(parents=True)
    spec = dict(model=dict(name="resnet50", spatial_dims=2), track="cfp_2d",
                disease="cataract", test_used=False, training=dict(seed=3416),
                train_manifest_sha256="train", development_manifest_sha256="dev",
                cache_receipt_sha256="cache", aggregation="valid_eye_feature_mean_v1")
    path = write(root / "specs" / (name + ".json"), spec)
    if score is not None:
        write(run / "accepted.json", dict(metrics=dict(macro_f1=score, auroc=.8),
                                          files={"best.pt": "fixture"}))
    return dict(id=name, spec=str(path), run_dir=str(run), spec_sha256=file_sha256(path))


def setup(tmp_path, scores=(.8, None)):
    tasks = [task(tmp_path, str(i), score) for i, score in enumerate(scores)]
    q = write(tmp_path / "queues" / "initial" / "queue.json", dict(tasks=tasks))
    catalog = write(tmp_path / "catalog.json", collect([q]))
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Frozen fixture")
    cfg = dict(schema="look_autoresearch_v1", test_access=False,
               catalog=dict(path=str(catalog), sha256=file_sha256(catalog)),
               protocol=dict(path=str(protocol), sha256=file_sha256(protocol)),
               source_pins=[], output=str(tmp_path / "workflow"),
               queues_root=str(tmp_path / "queues"), models_root=str(tmp_path / "models"))
    def reserve(root, namespace, identity, spec, **_):
        path = Path(root) / identity.replace("/", "_")
        path.mkdir(parents=True, exist_ok=True)
        existing = path / "configuration.json"
        if existing.exists():
            assert json.loads(existing.read_text()) == spec
        else:
            write(existing, spec)
        return path
    return Controller(cfg, lambda *a: None, reserve), tasks


def group(status):
    return status["groups"]["cataract/resnet50/cfp_2d"]


def test_wait_then_auto_select_register_two_replicas_and_restart(tmp_path):
    c, tasks = setup(tmp_path)
    assert group(c.tick())["state"] == "waiting_native_screen"
    assert not list(c.root.glob("groups/*/queue.json"))
    write(Path(tasks[1]["run_dir"]) / "accepted.json",
          dict(metrics=dict(macro_f1=.7, auroc=.9), files={"best.pt": "fixture"}))
    first = group(c.tick())
    assert first["state"] == "waiting_replications"
    queue = json.loads(Path(first["queue"]).read_text())
    assert len(queue["tasks"]) == 2
    seeds = [json.loads(Path(t["spec"]).read_text())["training"]["seed"] for t in queue["tasks"]]
    assert seeds == [3417, 3418]
    assert group(c.tick())["queue_sha256"] == first["queue_sha256"]
    assert len(list((tmp_path / "models").iterdir())) == 2
    for t in queue["tasks"]:
        write(Path(t["run_dir"]) / "accepted.json", {"fixture": True})
    assert group(c.tick())["state"] == "waiting_project_adapter"
    assert not c.tick()["formal_project_dispatch_ready"]


def test_new_better_candidate_does_not_change_locked_round(tmp_path):
    c, tasks = setup(tmp_path, (.8, .7))
    before = group(c.tick())["selected"]
    new = task(tmp_path, "later_better", .99)
    write(tmp_path / "queues" / "later" / "queue.json", dict(tasks=[new]))
    after = c.tick()
    assert group(after)["selected"] == before
    assert after["next_round_candidates"] == 1
    assert after["current_round_candidates"] == 2


def test_changed_spec_and_receipt_do_not_create_new_selection(tmp_path):
    c, tasks = setup(tmp_path, (.8, .7))
    first = group(c.tick())
    path = Path(tasks[0]["run_dir"]) / "accepted.json"
    write(path, dict(metrics=dict(macro_f1=.5, auroc=.8), files={"best.pt": "fixture"}))
    assert group(c.tick())["state"] == "needs_review"
    assert json.loads(Path(first["queue"]).read_text())["nomination"] == first["selected"]


def test_config_pin_mutation_fails_without_touching_workers(tmp_path):
    c, _ = setup(tmp_path)
    c.tick()
    Path(c.config["protocol"]["path"]).write_text("Changed")
    with pytest.raises(ValueError, match="Pinned input"):
        c.tick()
    status = json.loads((c.root / "status.json").read_text())
    assert status["healthy_workers_untouched"]


def test_atomic_controller_lock_prevents_second_reconciler(tmp_path):
    c, _ = setup(tmp_path)
    with (c.root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert c.tick()["state"] == "another_controller_active"


def test_bad_discovery_queue_isolated_and_test_only_metadata(tmp_path):
    c, _ = setup(tmp_path)
    bad = tmp_path / "queues" / "broken" / "queue.json"
    bad.parent.mkdir(); bad.write_text("bad json")
    t = task(tmp_path, "exposed", .9)
    path = Path(t["spec"]); spec = json.loads(path.read_text()); spec["test_used"] = True
    write(path, spec); t["spec_sha256"] = file_sha256(path)
    write(tmp_path / "queues" / "exposed" / "queue.json", dict(tasks=[t]))
    data = discover(c.config["queues_root"], set())
    assert len(data["issues"]) == 1
    assert next(x for x in data["models"] if x["run_dir"] == t["run_dir"])["state"] == "ineligible_test_provenance"
    assert not data["test_predictions_read"]


def test_replication_changes_seed_and_seed_provenance_only(tmp_path):
    _, tasks = setup(tmp_path)
    s = json.loads(Path(tasks[0]["spec"]).read_text())
    s["literature"] = dict(setting_provenance=dict(seed=dict(actual=3416)))
    r = replica_spec(s, 3417, {"parent": "fixture"})
    assert s["training"]["seed"] == 3416
    assert r["literature"]["setting_provenance"]["seed"]["actual"] == 3417
    assert r["train_manifest_sha256"] == s["train_manifest_sha256"]
    with pytest.raises(ValueError):
        replica_spec(s, 3420, {})
