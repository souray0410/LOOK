"""Concrete Ibex adapter for the LOOK V5 monitor-only handover.

Normal imports and inspection are read-only. Production mutation is exposed only
through :func:`apply_production` and requires an exact immutable binding plus the
explicit production flag in the CLI wrapper.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from look.runtime.rotation_gate import gpu_count
from look.runtime.v5_monitor_handover import run_handover, spec_sha256
from look.runtime.v5_role_policy_transition import verify_look_only_delta


PRODUCTION_ROOT = Path(
    "/ibex/project/c2377/souray/home/mengh/operations/2026_09_10_11_11_31"
)
PRODUCTION_LOCK = PRODUCTION_ROOT / "gpu_account_submission.lock"
PRODUCTION_POLICY = PRODUCTION_ROOT / "research_gpu_roles.json"
PRODUCTION_USER = "mengh"
PRODUCTION_ACCOUNT = "pi-mengy"


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _parse_scontrol(line: str) -> dict[str, str]:
    return dict(token.split("=", 1) for token in shlex.split(line) if "=" in token)


def _base_state(value: str) -> str:
    return value.split()[0].split("+")[0]


class SlurmAdapter:
    """Exact scontrol/sacct/sbatch adapter with bounded state convergence waits."""

    def __init__(
        self,
        *,
        user: str,
        account: str,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        wait_seconds: float = 20.0,
        command_timeout_seconds: float = 20.0,
    ) -> None:
        self.user, self.account, self.runner = user, account, runner
        self.wait_seconds = wait_seconds
        self.command_timeout_seconds = command_timeout_seconds

    def _run(self, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        return self.runner(
            args,
            text=True,
            capture_output=True,
            check=check,
            timeout=self.command_timeout_seconds,
        )

    def _has_step(self, job_id: str) -> bool:
        proc = self._run(
            ["sstat", "-j", f"{job_id}.batch", "--noheader", "--parsable2", "--format=JobIDRaw"],
            check=False,
        )
        return proc.returncode == 0 and bool(proc.stdout.strip())

    def observe(self, job_id: str) -> dict[str, Any]:
        proc = self._run(["scontrol", "show", "job", "-o", str(job_id)], check=False)
        if proc.returncode == 0 and proc.stdout.strip():
            row = _parse_scontrol(proc.stdout.strip())
            if row.get("JobId") == str(job_id):
                return {
                    "job_id": str(job_id),
                    "state": _base_state(row["JobState"]),
                    "reason": row.get("Reason"),
                    "dependency": row.get("Dependency", "(null)"),
                    "user": row["UserId"].split("(", 1)[0],
                    "account": row["Account"],
                    "req_tres": row["ReqTRES"],
                    "command": row["Command"],
                    "name": row["JobName"],
                    "submit_time": row["SubmitTime"],
                    "exit_code": row["ExitCode"],
                    "has_step": self._has_step(str(job_id)),
                }
        accounting = self._run(
            [
                "sacct", "-X", "-nP", "-j", str(job_id),
                "--format=JobIDRaw,JobName,State,ExitCode,User,Account,ReqTRES",
            ],
            check=False,
        )
        rows = [line.split("|") for line in accounting.stdout.splitlines() if line.split("|")[0] == str(job_id)]
        if len(rows) != 1:
            raise ValueError(f"Slurm identity is absent or ambiguous: {job_id}")
        row = rows[0]
        return {
            "job_id": str(job_id),
            "name": row[1],
            "state": _base_state(row[2]),
            "exit_code": row[3],
            "user": row[4],
            "account": row[5],
            "req_tres": row[6],
            "has_step": self._has_step(str(job_id)),
        }

    @staticmethod
    def job_name(spec: dict[str, Any]) -> str:
        prefix = "lookv22_" if spec.get("mode") == "production_v22" else "lookv21_"
        return prefix + spec_sha256(spec)[:14]

    def find_held(self, spec: dict[str, Any]) -> list[str]:
        proc = self._run(
            ["squeue", "-u", self.user, "-h", "-t", "PENDING", "-n", self.job_name(spec), "-o", "%A"],
        )
        found = []
        for job_id in proc.stdout.split():
            observed = self.observe(job_id)
            if (
                observed.get("state") == "PENDING"
                and observed.get("reason") == "JobHeldUser"
                and observed.get("dependency") == spec["dependency"]
                and observed.get("command") == spec["replacement_command"]
            ):
                found.append(job_id)
        return found

    def submit_held(self, spec: dict[str, Any], claim_path: str | Path, output: str | Path) -> str:
        proc = self._run(
            [
                "sbatch", "--hold", "--parsable", f"--dependency=afterany:{spec['gpu_job_id']}",
                f"--job-name={self.job_name(spec)}", f"--account={self.account}",
                "--partition=batch", "--time=00:20:00", "--cpus-per-task=2", "--mem=4G",
                f"--output={Path(output).resolve()}/monitor-%j.log",
                spec["replacement_command"], str(spec["gpu_job_id"]), str(Path(claim_path).resolve()),
            ]
        )
        job_id = proc.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise ValueError("sbatch returned a malformed job id")
        return job_id

    def account_gpu_count(self) -> int:
        proc = self._run(
            ["squeue", "-u", self.user, "-h", "-t", "RUNNING,PENDING", "-o", "%A|%b"]
        )
        total = 0
        for line in proc.stdout.splitlines():
            _, gres = line.split("|", 1)
            total += gpu_count(gres)
        return total

    def _wait(self, job_id: str, predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
        deadline = time.monotonic() + self.wait_seconds
        last: dict[str, Any] | None = None
        while time.monotonic() <= deadline:
            last = self.observe(job_id)
            if predicate(last):
                return last
            time.sleep(0.2)
        raise TimeoutError(f"Slurm state did not converge for {job_id}: {last}")

    def cancel(self, job_id: str) -> None:
        current = self.observe(job_id)
        if current.get("state") == "CANCELLED":
            return
        self._run(["scancel", str(job_id)])
        self._wait(str(job_id), lambda row: row.get("state") == "CANCELLED")

    def release(self, job_id: str) -> None:
        current = self.observe(job_id)
        if current.get("state") == "PENDING" and current.get("reason") != "JobHeldUser":
            return
        self._run(["scontrol", "release", str(job_id)])
        self._wait(
            str(job_id),
            lambda row: row.get("state") == "PENDING" and row.get("reason") != "JobHeldUser",
        )


class ArtifactInstaller:
    """Validate and idempotently install one LOOK-only policy transition."""

    REQUIRED = {
        "schema", "active_policy", "base_policy_snapshot", "base_policy_sha256",
        "candidate_policy", "candidate_policy_sha256", "dedicated_journal",
        "dedicated_journal_sha256", "claim", "claim_sha256", "provenance",
        "provenance_sha256", "replacement_command", "replacement_source_sha256",
        "gpu_job_id", "test_access",
    }

    def __init__(self, binding_path: str | Path, *, expected_sha256: str) -> None:
        self.path = Path(binding_path).resolve()
        if sha256(self.path) != expected_sha256:
            raise ValueError("Handover binding identity changed")
        self.binding = read_json(self.path)
        if set(self.binding) != self.REQUIRED or self.binding.get("schema") != "look_v5_policy_binding_v1":
            raise ValueError("Unknown policy binding")
        if self.binding.get("test_access") is not False:
            raise ValueError("Test access is forbidden")

    def _path(self, field: str, sha_field: str) -> Path:
        path = Path(self.binding[field]).resolve()
        if sha256(path) != self.binding[sha_field]:
            raise ValueError(f"Bound artifact changed: {field}")
        return path

    def verify(self, spec: dict[str, Any]) -> dict[str, Path]:
        base = self._path("base_policy_snapshot", "base_policy_sha256")
        candidate = self._path("candidate_policy", "candidate_policy_sha256")
        journal = self._path("dedicated_journal", "dedicated_journal_sha256")
        claim = self._path("claim", "claim_sha256")
        provenance = self._path("provenance", "provenance_sha256")
        replacement = self._path("replacement_command", "replacement_source_sha256")
        if str(replacement) != str(Path(spec["replacement_command"]).resolve()):
            raise ValueError("Replacement command path changed")
        if self.binding["replacement_source_sha256"] != spec["replacement_source_sha256"]:
            raise ValueError("Replacement command digest changed")
        if str(self.path) == str(provenance):
            raise ValueError("Binding and provenance must be independent artifacts")
        base_value, candidate_value = read_json(base), read_json(candidate)
        verify_look_only_delta(base_value, candidate_value, journal)
        request_rows = read_json(journal).get("requests", [])
        matching = [row for row in request_rows if str(row.get("job_id")) == str(spec["gpu_job_id"])]
        if len(matching) != 1 or matching[0].get("v5_claim_sha256") != self.binding["claim_sha256"]:
            raise ValueError("Dedicated journal does not uniquely register the live V5 GPU job")
        if self.binding["gpu_job_id"] != str(spec["gpu_job_id"]):
            raise ValueError("Binding GPU identity changed")
        if read_json(provenance).get("test_access") is not False:
            raise ValueError("Provenance permits test access")
        return {
            "base": base,
            "candidate": candidate,
            "journal": journal,
            "claim": claim,
            "provenance": provenance,
        }

    @staticmethod
    def _install_bytes(active: Path, candidate: Path) -> None:
        active.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{active.name}.", dir=active.parent)
        temporary_path = Path(temporary)
        try:
            payload = candidate.read_bytes()
            if os.write(fd, payload) != len(payload):
                raise OSError("short policy install write")
            os.fsync(fd)
            os.close(fd)
            fd = -1
            os.chmod(temporary_path, candidate.stat().st_mode & 0o777)
            os.replace(temporary_path, active)
            directory_fd = os.open(active.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if fd >= 0:
                os.close(fd)
            if temporary_path.exists():
                temporary_path.unlink()

    def commit(self, spec: dict[str, Any], _: str) -> dict[str, str]:
        artifacts = self.verify(spec)
        active = Path(self.binding["active_policy"]).resolve()
        current = sha256(active)
        if current == self.binding["base_policy_sha256"]:
            self._install_bytes(active, artifacts["candidate"])
        elif current != self.binding["candidate_policy_sha256"]:
            raise ValueError("Active role policy is neither the bound base nor installed candidate")
        if sha256(active) != self.binding["candidate_policy_sha256"]:
            raise ValueError("LOOK-only role policy installation did not persist")
        return {
            "policy_sha256": self.binding["candidate_policy_sha256"],
            "journal_sha256": self.binding["dedicated_journal_sha256"],
            "claim_sha256": self.binding["claim_sha256"],
            "provenance_sha256": self.binding["provenance_sha256"],
        }

    def preflight(self, spec: dict[str, Any]) -> dict[str, str]:
        self.verify(spec)
        active = Path(self.binding["active_policy"]).resolve()
        if sha256(active) != self.binding["base_policy_sha256"]:
            raise ValueError("Active role policy no longer matches the bound base")
        return {
            "policy_sha256": self.binding["candidate_policy_sha256"],
            "journal_sha256": self.binding["dedicated_journal_sha256"],
            "claim_sha256": self.binding["claim_sha256"],
            "provenance_sha256": self.binding["provenance_sha256"],
        }


def build_policy_binding(
    *, active_policy: str | Path, base_policy_snapshot: str | Path,
    candidate_policy: str | Path, dedicated_journal: str | Path,
    claim: str | Path, provenance: str | Path, replacement_command: str | Path,
    gpu_job_id: str,
) -> dict[str, Any]:
    values = {
        "base_policy_snapshot": Path(base_policy_snapshot).resolve(),
        "candidate_policy": Path(candidate_policy).resolve(),
        "dedicated_journal": Path(dedicated_journal).resolve(),
        "claim": Path(claim).resolve(),
        "provenance": Path(provenance).resolve(),
        "replacement_command": Path(replacement_command).resolve(),
    }
    for path in values.values():
        if not path.is_file():
            raise ValueError(f"Bound artifact is missing: {path}")
    result: dict[str, Any] = {
        "schema": "look_v5_policy_binding_v1",
        "active_policy": str(Path(active_policy).resolve()),
        "gpu_job_id": str(gpu_job_id),
        "test_access": False,
    }
    digest_fields = {
        "base_policy_snapshot": "base_policy_sha256",
        "candidate_policy": "candidate_policy_sha256",
        "dedicated_journal": "dedicated_journal_sha256",
        "claim": "claim_sha256",
        "provenance": "provenance_sha256",
        "replacement_command": "replacement_source_sha256",
    }
    for field, path in values.items():
        result[field] = str(path)
        result[digest_fields[field]] = sha256(path)
    return result


def build_handover_spec(
    *, old_job: dict[str, Any], gpu_job: dict[str, Any], replacement_command: str | Path,
    binding_path: str | Path, shared_lock: str | Path, mode: str = "production",
) -> dict[str, Any]:
    replacement, binding = Path(replacement_command).resolve(), Path(binding_path).resolve()
    if not replacement.is_file() or not binding.is_file():
        raise ValueError("Replacement command and binding must exist")
    dependency = old_job.get("dependency")
    result = {
        "schema": "look_v5_monitor_handover_spec_v2",
        "mode": mode,
        "transaction_id": f"look-v5-{'v22' if mode == 'production_v22' else 'v21'}-{gpu_job['job_id']}-{sha256(binding)[:12]}",
        "old_job_id": str(old_job["job_id"]),
        "gpu_job_id": str(gpu_job["job_id"]),
        "dependency": dependency,
        "old_command": str(old_job["command"]),
        "gpu_command": str(gpu_job["command"]),
        "replacement_command": str(replacement),
        "replacement_source_sha256": sha256(replacement),
        "binding_sha256": sha256(binding),
        "expected_user": old_job["user"],
        "expected_account": old_job["account"],
        "expected_monitor_req_tres": old_job["req_tres"],
        "expected_gpu_req_tres": gpu_job["req_tres"],
        "shared_lock": str(Path(shared_lock).resolve()),
        "max_account_gpus": 24,
        "test_access": False,
    }
    if gpu_job["user"] != result["expected_user"] or gpu_job["account"] != result["expected_account"]:
        raise ValueError("GPU and monitor Slurm owners differ")
    if dependency != f"afterany:{gpu_job['job_id']}(unfulfilled)":
        raise ValueError("Monitor dependency does not bind the exact GPU predecessor")
    return result

def apply_production(
    *, spec_path: str | Path, binding_path: str | Path, receipt_dir: str | Path,
    allow_production: bool,
) -> dict[str, Any]:
    if not allow_production:
        raise PermissionError("Explicit production apply flag is required")
    spec_path, binding_path = Path(spec_path).resolve(), Path(binding_path).resolve()
    spec = read_json(spec_path)
    if spec.get("mode") != "production":
        raise ValueError("Production apply requires a production spec")
    if Path(spec["shared_lock"]).resolve() != PRODUCTION_LOCK:
        raise ValueError("Production shared lock path changed")
    installer = ArtifactInstaller(binding_path, expected_sha256=spec["binding_sha256"])
    if Path(installer.binding["active_policy"]).resolve() != PRODUCTION_POLICY:
        raise ValueError("Production role policy path changed")
    if spec["expected_user"] != PRODUCTION_USER or spec["expected_account"] != PRODUCTION_ACCOUNT:
        raise ValueError("Production Slurm identity changed")
    slurm = SlurmAdapter(user=PRODUCTION_USER, account=PRODUCTION_ACCOUNT)
    return run_handover(
        receipt_dir,
        lock_path=PRODUCTION_LOCK,
        spec=spec,
        observe=slurm.observe,
        find_held=slurm.find_held,
        submit_held=lambda value: slurm.submit_held(value, installer.binding["claim"], receipt_dir),
        cancel=slurm.cancel,
        preflight_artifacts=installer.preflight,
        commit_artifacts=installer.commit,
        release=slurm.release,
        account_gpu_count=slurm.account_gpu_count,
    )
