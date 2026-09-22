"""Continuous discovery with immutable research rounds and idempotent work orders.

This CPU controller never owns a GPU or overrides a training worker. Native
queues it publishes are consumed by the existing resource/claim scheduler.
Unimplemented project adapters stay explicit prerequisites, never fake success.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import logging
import signal
import time
from pathlib import Path

from look.runtime.state import atomic_write_json, file_sha256, stable_hash, utc_now
from look.studies.native_prerequisites import audit


def read(path):
    return json.loads(Path(path).read_text())


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Immutable research artifact changed: {path}")
    else:
        atomic_write_json(value, path)


def discover(queues_root, frozen_runs):
    """Inspect specifications only; never open image arrays or test predictions."""
    found, issues = {}, []
    for queue in sorted(Path(queues_root).glob("*/queue.json")):
        try:
            tasks = read(queue)["tasks"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append(dict(queue=str(queue), reason=str(error)))
            continue
        for task in tasks:
            try:
                path = Path(task["spec"])
                if file_sha256(path) != task["spec_sha256"]:
                    raise ValueError("Spec checksum changed")
                spec = read(path)
                run = str(Path(task["run_dir"]).resolve())
                value = dict(run_dir=run, spec=str(path), spec_sha256=task["spec_sha256"],
                             model=spec["model"], track=spec["track"],
                             disease=spec["disease"], seed=spec["training"]["seed"],
                             test_access_declared=spec.get("test_used"),
                             in_locked_round=run in frozen_runs,
                             state="registered_not_accepted",
                             queue_sources=[str(queue)])
                if spec.get("test_used") is not False:
                    value["state"] = "ineligible_test_provenance"
                elif run not in frozen_runs:
                    value["state"] = "next_round_candidate_not_selected"
                if run in found:
                    if found[run]["spec_sha256"] != value["spec_sha256"]:
                        raise ValueError("One run has conflicting specifications")
                    found[run]["queue_sources"].append(str(queue))
                else:
                    found[run] = value
            except (OSError, ValueError, KeyError, TypeError) as error:
                issues.append(dict(queue=str(queue), reason=str(error),
                                   task_id=task.get("id") if isinstance(task, dict) else None))
    return dict(updated_at=utc_now(), models=sorted(found.values(), key=lambda x: x["run_dir"]),
                issues=issues, test_predictions_read=False)


def replica_spec(spec, seed, nomination):
    if seed not in (3417, 3418) or spec["training"]["seed"] != 3416:
        raise ValueError("Only the two protocol replication seeds are permitted")
    child = copy.deepcopy(spec)
    child["training"]["seed"] = seed
    settings = child.get("literature", {}).get("setting_provenance", {})
    if "seed" in settings:
        settings["seed"]["actual"] = seed
    child["recipe_selection"] = nomination
    # No relabeling of screening eligibility or training-completion semantics.
    return child


class Controller:
    def __init__(self, config, verify_completion, reserve):
        self.config = config
        self.root = Path(config["output"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.verify = verify_completion
        self.reserve = reserve

    def _validate(self):
        c = self.config
        if c.get("schema") != "look_autoresearch_v1" or c.get("test_access") is not False:
            raise ValueError("Unrecognized or unsealed controller")
        for item in [c["catalog"], c["protocol"], *c["source_pins"]]:
            if file_sha256(Path(item["path"])) != item["sha256"]:
                raise ValueError(f"Pinned input changed: {item['path']}")
        catalog = read(c["catalog"]["path"])
        if catalog.get("replication_seeds") != [3416, 3417, 3418]:
            raise ValueError("Unexpected seed protocol")
        return catalog

    def tick(self):
        with (self.root / "controller.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {"state": "another_controller_active"}
            try:
                immutable(self.root / "controller_config.json", self.config)
                return self._tick()
            except Exception as error:
                atomic_write_json(dict(state="needs_implementation_review", error=repr(error),
                    updated_at=utc_now(), healthy_workers_untouched=True), self.root / "status.json")
                raise

    def _tick(self):
        catalog = self._validate()
        discovery = discover(self.config["queues_root"], {r["run_dir"] for r in catalog["candidates"]})
        atomic_write_json(discovery, self.root / "discovery.json")
        assessment = audit(catalog, self.verify)
        atomic_write_json(assessment, self.root / "native_readiness.json")
        groups, work = {}, []
        for name, group in assessment["groups"].items():
            key = stable_hash(name)[:20]
            dest = self.root / "groups" / key
            dest.mkdir(parents=True, exist_ok=True)
            try:
                result = self._group(name, group, dest)
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
                result = dict(state="needs_review", reason=str(error))
            groups[name] = result
            if result.get("queue"):
                work.append({k: result[k] for k in ("queue", "queue_sha256", "state")})
        # Put unresolved, already registered parents behind replications in the
        # same atomic-claim scheduler. No new scientific run or recipe is made.
        if self.config.get("prioritize_existing_parents", False):
            tasks = [dict(id="parent__"+stable_hash(r["run_dir"])[:20],role="model",
                spec=r["spec"],spec_sha256=r["spec_sha256"],run_dir=r["run_dir"])
                for r in catalog["candidates"]]
            queue=self.root/"existing_parent_queue.json"
            immutable(queue,dict(schema="existing_native_parent_priority_v1",tasks=tasks,test_used=False))
            work.append(dict(queue=str(queue),queue_sha256=file_sha256(queue),state="existing_parent_priority"))
        # This feed is a durable handoff to the existing resource scheduler. It
        # is not an alternative lease mechanism and never mutates live queues.
        atomic_write_json(dict(schema="look_native_work_feed_v1", updated_at=utc_now(),
                              queues=work, test_access=False), self.root / "native_work_feed.json")
        projects = None
        if self.config.get("project"):
            from look.studies.project_orders import advance
            projects = advance(self.config, groups, self.verify, self.reserve)
        status = dict(schema="look_autoresearch_status_v1", updated_at=utc_now(),
                      state="active", groups=groups, discovered=len(discovery["models"]),
                      discovery_issues=len(discovery["issues"]),
                      current_round_candidates=len(catalog["candidates"]),
                      accepted_native=sum(g["accepted_count"] for g in assessment["groups"].values()),
                      next_round_candidates=sum(not r["in_locked_round"] for r in discovery["models"]),
                      formal_project_dispatch_ready=bool(projects and projects["tasks"]), test_access=False,
                      projects=projects,
                      remaining_implementation=[] if self.config.get("project") else
                          ["full_fusion_host_adapter", "full_LOOK_correction_adapter", "project_replay_and_resource_receipts"])
        previous = read(self.root / "status.json") if (self.root / "status.json").exists() else {}
        transitions = [dict(group=name, before=previous.get("groups", {}).get(name, {}).get("state"),
                            after=row["state"]) for name, row in groups.items()
                       if previous.get("groups", {}).get(name, {}).get("state") != row["state"]]
        if transitions:
            events = self.root / "events"
            events.mkdir(exist_ok=True)
            atomic_write_json(dict(time=utc_now(), transitions=transitions,
                                   notify_user=any(t["after"] == "needs_review" for t in transitions)),
                              events / f"{time.time_ns()}.json")
        atomic_write_json(status, self.root / "status.json")
        self._report(status)
        return status

    def _group(self, name, group, dest):
        nomination_path = dest / "nomination.json"
        if group["state"] != "screen_closed":
            if nomination_path.exists():
                raise ValueError("Previously locked parent no longer passes acceptance")
            bad = [r for r in group["candidates"] if r["state"] == "needs_review"]
            return dict(state="needs_review" if bad else "waiting_native_screen",
                        accepted=group["accepted_count"], total=group["candidate_count"],
                        issues=[dict(run_dir=r["run_dir"], reason=r["reason"]) for r in bad])
        selected = group["selected"]
        nomination = dict(group=name, catalog_sha256=self.config["catalog"]["sha256"],
                          protocol_sha256=self.config["protocol"]["sha256"],
                          run_dir=selected["run_dir"], spec_sha256=selected["spec_sha256"],
                          accepted_sha256=selected["accepted_sha256"],
                          best_sha256=selected["best_sha256"],
                          selection="development_macro_f1_then_auroc_then_spec_sha256_then_run_id")
        immutable(nomination_path, nomination)
        spec = read(selected["spec"])
        tasks = []
        for seed in (3417, 3418):
            child = replica_spec(spec, seed, nomination)
            path = dest / f"seed{seed}.json"
            immutable(path, child)
            namespace = "look_parent_replication_" + self.config["catalog"]["sha256"][:16]
            run = self.reserve(self.config["models_root"], namespace, name + f"/seed{seed}", child,
                               source=dict(nomination=str(nomination_path)), refresh_summary=False)
            tasks.append(dict(id=name.replace("/", "__") + f"__seed{seed}", role="model",
                              spec=str(path), spec_sha256=file_sha256(path),
                              run_dir=str(run), state="pending"))
        queue = dest / "queue.json"
        immutable(queue, dict(schema="look_selected_native_replication_v1", tasks=tasks,
                              nomination=nomination, test_used=False))
        evidence = []
        for task in tasks:
            root = Path(task["run_dir"])
            if not (root / "accepted.json").exists():
                evidence.append(dict(run_dir=str(root), state="awaiting_native_acceptance"))
            else:
                self.verify(str(root), read(task["spec"]))
                evidence.append(dict(run_dir=str(root), state="accepted",
                                     accepted_sha256=file_sha256(root / "accepted.json")))
        ready = all(r["state"] == "accepted" for r in evidence)
        return dict(state="waiting_project_adapter" if ready else "waiting_replications",
                    selected=nomination, queue=str(queue), queue_sha256=file_sha256(queue),
                    replicas=evidence, formal_project_dispatch_ready=False)

    def _report(self, status):
        rows = ["# LOOK 自动研究进度", "", f"更新时间：{status['updated_at']}", "",
                "持续发现新模型；当前研究轮候选固定。验收不等于项目性能达标。", "",
                "| 分组 | 阶段 | 当前轮验收 |", "|---|---|---|"]
        for name, group in status["groups"].items():
            rows.append(f"| {name} | {group['state']} | {group.get('accepted', '—')}/{group.get('total', '—')} |")
        rows += ["", "新模型进入后续研究轮，不替换已锁定的父模型。",
                 ("融合训练、LOOK修正与匹配开发评价已接通；任务等待匹配父模型，GPU资源验收在正式训练之前自动执行。"
                  if self.config.get("project") else "完整融合/LOOK适配器尚未验收，当前不会派发正式LOOK训练，也不会读取test。")]
        if status.get("projects"):
            rows += ["", f"已登记项目任务：{status['projects']['tasks']}，完整队列及逐任务阶段在projects目录。"]
        from look.runtime.state import atomic_write_text
        atomic_write_text("\n".join(rows) + "\n", self.root / "REPORT.zh-CN.md")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=1800)
    args = parser.parse_args()
    if args.interval < 60:
        raise ValueError("Avoid frequent full artifact hashing")
    # Model_Training is an explicitly supplied runtime dependency, not bundled
    # into the model framework or the independent scientific model itself.
    from mhd_models.runtime.training_state import verify_completion
    from mhd_models.runtime.run_registry import reserve
    controller = Controller(read(args.config), verify_completion, reserve)
    daemon_lock = (controller.root / "daemon.lock").open("a")
    try:
        fcntl.flock(daemon_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logging.warning("An automatic research process already owns this round")
        return
    running = True
    def stop(*_):
        nonlocal running
        running = False
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while running:
        try:
            result = controller.tick()
            logging.warning("Research reconciliation: %s", result.get("state"))
        except Exception:
            logging.exception("Research reconciliation failed; GPU workers preserved")
            if args.once:
                raise
        if args.once:
            break
        deadline = time.monotonic() + args.interval
        while running and time.monotonic() < deadline:
            time.sleep(min(1, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    main()
