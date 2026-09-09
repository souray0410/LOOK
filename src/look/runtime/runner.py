from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from look.runtime.state import PipelineState, atomic_write_json, atomic_write_text, stable_hash, utc_now


@dataclass(frozen=True)
class CommandCase:
    """One immutable command invocation in a reproducible batch."""

    name: str
    argv: tuple[str, ...]
    environment: Mapping[str, str] = field(default_factory=dict)

    @property
    def case_id(self) -> str:
        return f"{self.name}__{stable_hash({'argv': self.argv, 'environment': dict(self.environment)})[:12]}"


def run_command_cases(
    cases: Iterable[CommandCase],
    output_root: Path,
    *,
    state_root: Path | None = None,
    execute: bool = False,
    resume: bool = True,
) -> list[dict[str, object]]:
    """Plan or run argv-only commands without source rewriting or a shell."""

    root = Path(output_root).resolve()
    states = Path(state_root or root / ".state").resolve()
    results: list[dict[str, object]] = []
    for case in cases:
        if not case.name or not case.argv:
            raise ValueError("Each command case requires a name and a non-empty argv")
        case_dir = root / case.case_id
        result_path = case_dir / "result.json"
        stdout_path = case_dir / "stdout.log"
        stderr_path = case_dir / "stderr.log"
        config = {"name": case.name, "argv": list(case.argv), "environment": dict(case.environment)}
        state = PipelineState(states, f"batch_{case.case_id}", config)
        if resume and state.completed_outputs_valid([result_path, stdout_path, stderr_path]):
            results.append({"case_id": case.case_id, "status": "reused", "result": str(result_path)})
            continue
        if not execute:
            results.append({"case_id": case.case_id, "status": "planned", "argv": list(case.argv)})
            continue
        case_dir.mkdir(parents=True, exist_ok=True)
        with state:
            started = utc_now()
            completed = subprocess.run(
                list(case.argv),
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, **dict(case.environment)},
            )
            atomic_write_text(completed.stdout, stdout_path)
            atomic_write_text(completed.stderr, stderr_path)
            payload = {
                **config,
                "case_id": case.case_id,
                "started_at_utc": started,
                "completed_at_utc": utc_now(),
                "returncode": completed.returncode,
            }
            atomic_write_json(payload, result_path)
            if completed.returncode != 0:
                raise subprocess.CalledProcessError(completed.returncode, case.argv)
            state.complete([result_path, stdout_path, stderr_path])
        results.append({"case_id": case.case_id, "status": "completed", "result": str(result_path)})
    return results


def build_flag_cases(base_argv: Sequence[str], parameter_sets: Iterable[Mapping[str, object]]) -> list[CommandCase]:
    """Convert structured mappings into deterministic CLI cases."""

    cases: list[CommandCase] = []
    for parameters in parameter_sets:
        argv = list(base_argv)
        for key in sorted(parameters):
            value = parameters[key]
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    argv.append(flag)
            elif isinstance(value, (list, tuple)):
                argv.extend([flag, *map(str, value)])
            elif value is not None:
                argv.extend([flag, str(value)])
        case_id = stable_hash(dict(parameters))[:12]
        cases.append(CommandCase(name=f"case_{case_id}", argv=tuple(argv)))
    return cases
