"""Build an immutable dispatch-release receipt from accepted family searches.

This module does not dispatch work or alter the historical weekly policy.  It
turns already accepted, strictly verified family-search cases into the evidence
object consumed by :func:`look.runtime.project_dispatch.verify_priority_release`.
"""
import argparse
from pathlib import Path

from look.runtime.state import atomic_write_json, file_sha256
from look.studies.project_case import read
from look.studies.family_search_protocol import ARMS, validate
from look.studies.family_search_case import verify_case


CORE_ARMS = ("residual_rrr", "pca_free_mean")
RELEASED_MODES = ("best_forward", "greedy")


def build(feed_path, output, *, required_arms=CORE_ARMS):
    """Verify accepted family cases and write one deterministic release receipt."""
    feed_path = Path(feed_path).resolve()
    output = Path(output)
    feed = read(feed_path)
    if (feed.get("schema") != "look_family_search_feed_v1"
            or feed.get("test_access") is not False):
        raise ValueError("Sealed family-search feed required")
    required_arms = tuple(required_arms)
    if required_arms not in (CORE_ARMS, tuple(ARMS)):
        raise ValueError("Only the approved core or full family package can release search")

    by_arm = {}
    for task in feed.get("tasks", []):
        if (task.get("execution") != "look_family_search"
                or task.get("test_access") is not False):
            raise ValueError("Unexpected family-search task")
        spec_path = Path(task["spec"]).resolve()
        if file_sha256(spec_path) != task.get("spec_sha256"):
            raise ValueError("Family-search specification changed")
        spec = read(spec_path)
        validate(spec)
        arm = spec["arm"]
        if arm in by_arm or task.get("arm") != arm:
            raise ValueError("Duplicate or mismatched family-search arm")
        by_arm[arm] = (task, spec_path, spec)
    if set(by_arm) != set(ARMS):
        raise ValueError("Exactly four registered fitting arms required")

    requirements = []
    for arm in required_arms:
        task, spec_path, spec = by_arm[arm]
        run = Path(task["run_dir"]).resolve()
        if read(run / "spec.json") != spec:
            raise ValueError("Accepted run spec differs from registered task")
        verify_case(run, spec)
        accepted = run / "accepted.json"
        requirements.append({
            "execution": "look_family_search",
            "arm": arm,
            "run_dir": str(run),
            "spec": str(spec_path),
            "spec_sha256": file_sha256(spec_path),
            "accepted_sha256": file_sha256(accepted),
        })
    receipt = {
        "schema": "look_priority_release_v1",
        "state": "accepted",
        "test_access": False,
        "released_search_modes": list(RELEASED_MODES),
        "family_feed": str(feed_path),
        "family_feed_sha256": file_sha256(feed_path),
        "required_arms": list(required_arms),
        "requirements": requirements,
    }
    if output.exists() and read(output) != receipt:
        raise ValueError("Existing priority release has a different identity")
    atomic_write_json(receipt, output)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feed", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--full-family", action="store_true")
    args = parser.parse_args()
    build(args.feed, args.output,
          required_arms=tuple(ARMS) if args.full_family else CORE_ARMS)


if __name__ == "__main__":
    main()
