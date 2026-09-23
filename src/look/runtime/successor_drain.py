"""Fail-closed drain receipt for an immutable LOOK dispatcher."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from look.runtime.state import atomic_write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def request(output, reason, *, now=time.time):
    output = Path(output)
    stop = output / "stop.json"
    value = {"schema": "look_dispatcher_drain_request_v1", "reason": reason,
             "requested_at": now(), "cancel_allocations": False,
             "signal_gpu_workers": False}
    fd = os.open(stop, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return value


def finalize(output, receipt, *, now=time.time):
    """Write a terminal receipt only while holding the released dispatcher lock."""
    output, receipt = Path(output), Path(receipt)
    stop, journal_path = output / "stop.json", output / "requests.json"
    if not stop.is_file() or not journal_path.is_file():
        raise ValueError("Drain request and request journal are required")
    journal = read(journal_path)
    if journal.get("schema") != "look_workflow_requests_v1":
        raise ValueError("Unknown request journal")
    jobs = []
    for row in journal.get("requests", []):
        job = str(row.get("job_id", ""))
        if not job.isdigit():
            raise ValueError("Every submitted request needs a Slurm identity")
        jobs.append(job)
    with (output / "dispatcher.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Dispatcher has not exited") from error
        identity = {
            "schema": "look_dispatcher_terminal_receipt_v1",
            "state": "dispatcher_drained",
            "stop_sha256": sha(stop),
            "request_journal_sha256": sha(journal_path),
            "terminal_job_ids": sorted(set(jobs), key=int),
            "cancel_allocations": False,
            "signal_gpu_workers": False,
        }
        if receipt.exists():
            existing = read(receipt)
            if {k: existing.get(k) for k in identity} != identity:
                raise ValueError("Conflicting terminal receipt")
            return existing
        value = dict(identity, observed_at=now())
        atomic_write_json(value, receipt)
        return value
