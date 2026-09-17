"""Score-independent release of repeat seeds after a whole weekly delivery.

This is scheduling metadata, never part of a scientific specification. Old live
workers retain ownership. A missing or stale delivery fails closed for NEW work.
"""
import json
from pathlib import Path
from look.runtime.state import stable_hash, file_sha256

SECTIONS = ("matched_results", "diagnostics", "statistics", "figures", "report")

def read(path):
    return json.loads(Path(path).read_text())

def release_state(policy_path):
    policy = read(policy_path)
    if policy.get("schema") != "look_weekly_delivery_v1" or policy.get("test_access") is not False:
        raise ValueError("Invalid weekly delivery policy")
    first = policy["first_seed"]
    if not isinstance(first, int) or not policy.get("package_id"):
        raise ValueError("Missing first seed or finite package identity")
    try:
        if policy.get("unresolved_requirements"):
            raise ValueError("Weekly implementation/coverage requirements remain unresolved")
        receipt = read(policy["release_receipt"])
        if (receipt.get("state") != "accepted" or receipt.get("profile") is not False
            or receipt.get("test_access") is not False
            or receipt.get("policy_sha256") != stable_hash(policy)):
            raise ValueError("Delivery receipt identity mismatch")
        # The report producer supplies explicit bindings to the accepted cases.
        required = policy.get("required_cases", [])
        if not required:
            raise ValueError("Empty weekly package cannot release repeat seeds")
        evidence = receipt["cases"]
        if set(evidence) != {r["run_dir"] for r in required}:
            raise ValueError("Incomplete matched package")
        for row in required:
            if file_sha256(row["spec"]) != row["spec_sha256"]:
                raise ValueError("Required scientific specification changed")
            path = Path(row["run_dir"]) / "accepted.json"
            accepted = read(path)
            if (file_sha256(path) != evidence[row["run_dir"]]
                or accepted.get("state") != "accepted"
                or accepted.get("profile", False) is not False
                or accepted.get("test_access") is not False):
                raise ValueError("Scientific acceptance changed")
            verify_case_evidence(row)
        for section in SECTIONS:
            artifacts = receipt["sections"][section]
            if not artifacts:
                raise ValueError("Missing delivery section: " + section)
            for artifact in artifacts:
                if file_sha256(artifact["path"]) != artifact["sha256"]:
                    raise ValueError("Delivery artifact changed: " + section)
        return dict(released=True, first_seed=first, reason="whole_weekly_package_accepted")
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(released=False, first_seed=first,
                    reason="waiting_whole_weekly_delivery", detail=str(error))


def verify_case_evidence(row):
    from look.runtime.project_dispatch import verify_delivery_dependency
    spec=read(row['spec'])
    kinds={'look_search_policy_v1':'look_search'}
    kind=row.get('execution',kinds.get(spec.get('schema')))
    if kind is None:raise ValueError('Weekly case needs its owning execution verifier')
    verify_delivery_dependency(dict(row,execution=kind))

def task_seed(task):
    spec = read(task["spec"])
    seed = spec.get("seed", spec.get("host", {}).get("seed"))
    if seed is None:
        seed = spec.get("task", {}).get("seed")
    return seed

def filter_tasks(tasks, policy_path):
    gate = release_state(policy_path)
    if gate["released"]:
        return tasks, []
    ready, held = [], []
    for task in tasks:
        # Generic native training is independent. Project parents remain managed
        # by their explicit prerequisite queue; this gate owns project dispatch.
        if task.get("execution") == "native" or task_seed(task) == gate["first_seed"]:
            ready.append(task)
        else:
            held.append(dict(run=task["run_dir"], reason=gate["reason"], seed=task_seed(task)))
    return ready, held
