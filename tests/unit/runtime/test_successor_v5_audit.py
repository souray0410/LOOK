import json

import pytest

from look.runtime.successor_v5_audit import FORMAL_COMPANION, FORMAL_FRAMEWORK, audit


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return str(path)


def test_only_explicit_formal_v5_identity_is_admitted(tmp_path):
    good = put(tmp_path / "good.json", {"framework_commit": FORMAL_FRAMEWORK,
                                        "mhd_models_commit": FORMAL_COMPANION})
    old = put(tmp_path / "old.json", {"framework": {"commit": "c0a27ab"}})
    feed = put(tmp_path / "feed.json", {"tasks": [{"id": "good", "spec": good},
                                                     {"id": "old", "spec": old}]})
    result = audit({"project_feed": feed})
    assert [row["task"] for row in result["admitted"]] == ["good"]
    assert [row["task"] for row in result["quarantined"]] == ["old"]


def test_legacy_dispatcher_config_is_rejected_before_feed_scan(tmp_path):
    with pytest.raises(ValueError, match="legacy_project_pythonpath"):
        audit({"legacy_project_pythonpath": "old"})
