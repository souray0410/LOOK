import json

import pytest

from look.runtime.successor_drain import finalize, request


def write(path, value):
    path.write_text(json.dumps(value) + "\n")


def test_drain_preserves_workers_and_records_all_terminal_jobs(tmp_path):
    write(tmp_path / "requests.json", {"schema": "look_workflow_requests_v1", "requests": [
        {"job_id": "9"}, {"job_id": "7"}, {"job_id": "9"}]})
    value = request(tmp_path, "V5 successor handover", now=lambda: 1.0)
    assert value["cancel_allocations"] is False
    assert value["signal_gpu_workers"] is False
    receipt = finalize(tmp_path, tmp_path / "terminal.json", now=lambda: 2.0)
    assert receipt["state"] == "dispatcher_drained"
    assert receipt["terminal_job_ids"] == ["7", "9"]
    assert finalize(tmp_path, tmp_path / "terminal.json", now=lambda: 3.0) == receipt


def test_drain_request_is_exclusive(tmp_path):
    request(tmp_path, "first")
    with pytest.raises(FileExistsError):
        request(tmp_path, "second")


def test_terminal_receipt_rejects_unidentified_submission(tmp_path):
    write(tmp_path / "requests.json", {"schema": "look_workflow_requests_v1", "requests": [
        {"state": "submitted_waiting_identity"}]})
    request(tmp_path, "drain")
    with pytest.raises(ValueError, match="Slurm identity"):
        finalize(tmp_path, tmp_path / "terminal.json")
