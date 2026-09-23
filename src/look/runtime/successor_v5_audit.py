"""Conservative formal-V5 admission for successor dispatcher feeds."""
import json
from pathlib import Path

FORMAL_FRAMEWORK = "1287681c08846e11364c81653048435482e772a7"
FORMAL_COMPANION = "cc16e74a8cfc705d69b3d31efe2daeec9404471f"
FEED_KEYS = ("project_feed", "native_feed", "additional_native_feeds", "supplement_feeds",
             "spatial_feeds", "linear_feeds", "terminal_feeds", "search_feeds",
             "family_search_feeds", "feature_replay_feeds", "suffix_feeds", "affine_feeds")


def read(path):
    return json.loads(Path(path).read_text())


def identity(spec):
    framework = spec.get("framework_commit") or spec.get("framework", {}).get("commit")
    companion = spec.get("mhd_models_commit") or spec.get("companion_commit")
    if framework != FORMAL_FRAMEWORK:
        return False, "missing_or_nonformal_framework_identity"
    if companion != FORMAL_COMPANION:
        return False, "missing_or_nonformal_companion_identity"
    text = json.dumps(spec, sort_keys=True)
    if "legacy_project_pythonpath" in text or "framework_c0a27ab" in text:
        return False, "legacy_execution_path"
    return True, "formal_v5_identity"


def audit(config):
    config = read(config) if not isinstance(config, dict) else config
    if config.get("legacy_project_pythonpath"):
        raise ValueError("Successor config retains legacy_project_pythonpath")
    admitted, quarantined = [], []
    for key in FEED_KEYS:
        paths = config.get(key, [])
        if isinstance(paths, str):
            paths = [paths]
        for feed_path in paths:
            feed = read(feed_path)
            tasks = list(feed.get("tasks", []))
            if key in ("native_feed", "additional_native_feeds"):
                tasks = [task for row in feed.get("queues", []) for task in read(row["queue"]).get("tasks", [])]
            for task in tasks:
                spec_path = task.get("spec")
                if not spec_path or not Path(spec_path).is_file():
                    quarantined.append({"feed_key": key, "task": task.get("id"), "reason": "missing_spec"})
                    continue
                ok, reason = identity(read(spec_path))
                row = {"feed_key": key, "task": task.get("id"), "spec": spec_path, "reason": reason}
                (admitted if ok else quarantined).append(row)
    return {"framework_commit": FORMAL_FRAMEWORK, "companion_commit": FORMAL_COMPANION,
            "admitted": admitted, "quarantined": quarantined}

