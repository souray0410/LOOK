from __future__ import annotations

import json
import sys

import pytest

from look_core.batch_runner import CommandCase, build_flag_cases, run_command_cases
from look_core.file_selection import execute_filter_plan, plan_keyword_filter, write_filter_plan


def test_batch_runner_plans_and_reuses_successful_case(tmp_path):
    cases = build_flag_cases((sys.executable, "-c", "print('ok')"), [{"seed": 1}])
    assert cases[0].argv[-2:] == ("--seed", "1")
    runnable = [CommandCase("success", (sys.executable, "-c", "print('ok')"))]
    planned = run_command_cases(runnable, tmp_path / "runs")
    assert planned[0]["status"] == "planned"
    completed = run_command_cases(runnable, tmp_path / "runs", execute=True)
    assert completed[0]["status"] == "completed"
    reused = run_command_cases(runnable, tmp_path / "runs", execute=True)
    assert reused[0]["status"] == "reused"


def test_keyword_filter_requires_stable_allowlist_and_quarantines(tmp_path):
    root = tmp_path / "source"
    folder = root / "participant"
    folder.mkdir(parents=True)
    keep = folder / "image_resnet50.png"
    remove = folder / "image_resnet18.png"
    keep.write_bytes(b"keep")
    remove.write_bytes(b"remove")
    plan = plan_keyword_filter(root, ["resnet18"], mode="delete_matches")
    plan_path = tmp_path / "plan.json"
    write_filter_plan(plan, plan_path)
    assert execute_filter_plan(plan_path, tmp_path / "quarantine")["status"] == "dry_run"
    assert remove.exists()
    result = execute_filter_plan(plan_path, tmp_path / "quarantine", execute=True)
    assert result["moved_files"] == ["participant/image_resnet18.png"]
    assert keep.exists() and not remove.exists()


def test_keyword_filter_refuses_changed_file(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    target = root / "old.txt"
    target.write_text("before")
    plan_path = tmp_path / "plan.json"
    write_filter_plan(plan_keyword_filter(root, ["old"], mode="delete_matches"), plan_path)
    target.write_text("after")
    with pytest.raises(ValueError, match="changed"):
        execute_filter_plan(plan_path, tmp_path / "quarantine", execute=True)
