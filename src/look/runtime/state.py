from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import socket
import tempfile
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable



def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o644)
        durable_replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o644)
        durable_replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def quarantine(path: Path, quarantine_root: Path, reason: str) -> Path:
    quarantine_root.mkdir(parents=True, exist_ok=True)
    destination = quarantine_root / f"{path.name}.{stable_hash(reason)[:8]}.{int(datetime.now().timestamp())}"
    shutil.move(str(path), destination)
    atomic_write_json({"source": str(path), "reason": reason, "moved_at_utc": utc_now()}, destination.with_suffix(destination.suffix + ".json"))
    return destination


class PipelineState(AbstractContextManager["PipelineState"]):
    """Atomic, resumable state and single-process locking for one pipeline stage."""

    def __init__(self, state_root: Path, stage: str, config: Any, inputs: Iterable[Path] = ()):
        self.stage = stage
        self.directory = state_root / stage
        self.state_path = self.directory / "state.json"
        self.lock_path = self.directory / "stage.lock"
        self.config_hash = stable_hash(config)
        self.inputs = [Path(path) for path in inputs]
        self._previous_handlers: dict[int, Any] = {}
        self.payload: dict[str, Any] = {}

    def __enter__(self) -> "PipelineState":
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.lock_path.is_file():
            try:
                lock = json.loads(self.lock_path.read_text(encoding="utf-8"))
                if lock.get("host") == socket.gethostname():
                    try:
                        os.kill(int(lock["pid"]), 0)
                    except (ProcessLookupError, ValueError, KeyError):
                        self.lock_path.unlink(missing_ok=True)
            except (OSError, json.JSONDecodeError):
                self.lock_path.unlink(missing_ok=True)
        try:
            descriptor = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            lock = json.loads(self.lock_path.read_text(encoding="utf-8"))
            raise RuntimeError(f"Stage {self.stage} is locked: {lock}") from error
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "host": socket.gethostname(), "started_at_utc": utc_now()}, handle)
        prior = self.read()
        from look.runtime.artifacts import build_file_manifest

        existing_inputs = [path for path in self.inputs if path.is_file()]
        self.payload = {
            "stage": self.stage,
            "status": "running",
            "started_at_utc": utc_now(),
            "config_hash": self.config_hash,
            "input_files": [str(path) for path in self.inputs],
            "input_manifest": build_file_manifest(existing_inputs),
            "progress": prior.get("progress", {}) if prior.get("config_hash") == self.config_hash else {},
        }
        atomic_write_json(self.payload, self.state_path)
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle_signal)
        return self

    def read(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def checkpoint(self, **progress: Any) -> None:
        self.payload.setdefault("progress", {}).update(progress)
        self.payload["updated_at_utc"] = utc_now()
        atomic_write_json(self.payload, self.state_path)

    def completed_outputs_valid(self, outputs: Iterable[Path] = ()) -> bool:
        from look.runtime.artifacts import validate_file_manifest

        prior = self.read()
        if prior.get("status") != "completed" or prior.get("config_hash") != self.config_hash:
            return False
        manifest = prior.get("outputs", {})
        expected = {str(Path(path)) for path in outputs}
        if expected and not expected.issubset(manifest):
            return False
        return not validate_file_manifest(manifest)

    def complete(self, outputs: Iterable[Path] = ()) -> None:
        from look.runtime.artifacts import build_file_manifest

        output_manifest = build_file_manifest(outputs)
        self.payload.update(status="completed", completed_at_utc=utc_now(), outputs=output_manifest)
        atomic_write_json(self.payload, self.state_path)

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        self.payload.update(status="interrupted", interrupted_at_utc=utc_now(), signal=signum)
        atomic_write_json(self.payload, self.state_path)
        self._release()
        raise SystemExit(128 + signum)

    def _release(self) -> None:
        self.lock_path.unlink(missing_ok=True)
        for signum, handler in self._previous_handlers.items():
            signal.signal(signum, handler)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if exc is not None:
            self.payload.update(status="failed", failed_at_utc=utc_now(), error=repr(exc))
            atomic_write_json(self.payload, self.state_path)
        self._release()
        return False


def durable_replace(temporary: Path, destination: Path) -> None:
    """Commit a fully written file and its directory entry before reporting completion."""
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(destination)
    descriptor = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
