from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


def parse_gpu_devices(values: Sequence[int | str] | str) -> tuple[int, ...]:
    tokens: Iterable[int | str]
    tokens = values.split(",") if isinstance(values, str) else values
    result = []
    for value in tokens:
        text = str(value).strip().lower()
        if text.startswith("cuda:"):
            text = text.split(":", 1)[1]
        if not text.isdigit():
            raise ValueError(f"Invalid GPU device: {value!r}")
        result.append(int(text))
    if not result or len(set(result)) != len(result):
        raise ValueError("GPU devices must be a non-empty unique list")
    return tuple(result)


def launch_ddp_stage(
    project_root: Path,
    payload: dict,
    stage: str,
    gpu_devices: Sequence[int],
    payload_path: Path,
) -> None:
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = payload_path.with_suffix(payload_path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(payload_path)
    devices = parse_gpu_devices(gpu_devices)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
    # The ws RTX 5000 Ada pair hangs on NCCL's default P2P/CUMEM transport.
    # Other deployments can explicitly restore P2P with NCCL_P2P_DISABLE=0.
    environment.setdefault("NCCL_P2P_DISABLE", "1")
    command = [
        sys.executable,
        "-m", "torch.distributed.run",
        "--standalone",
        "--nproc-per-node", str(len(devices)),
        "-m", "look_core.distributed_worker",
        "--project-root", str(project_root),
        "--payload", str(payload_path),
        "--stage", stage,
    ]
    process = subprocess.Popen(command, cwd=project_root, env=environment, start_new_session=True)
    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        os.killpg(process.pid, signal.SIGINT)
        return_code = process.wait()
        raise
    if return_code:
        raise RuntimeError(f"DDP stage {stage!r} failed with exit code {return_code}")
