from __future__ import annotations

import json
import hashlib
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.distributed as dist


def module_state_sha256(module: torch.nn.Module) -> str:
    """Hash every parameter and persistent buffer in deterministic name order."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def all_gather_object(value, context):
    if not context.distributed:
        return [value]
    gathered = [None] * context.world_size
    dist.all_gather_object(gathered, value)
    return gathered


def assert_module_state_identical(module: torch.nn.Module, context) -> str:
    state_hash = module_state_sha256(module)
    gathered = all_gather_object(state_hash, context)
    if len(set(gathered)) != 1:
        raise RuntimeError(f"Module state differs across ranks: {gathered}")
    return state_hash


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
    environment.setdefault("NCCL_SHM_DISABLE", "0")
    environment.setdefault("PYTHONFAULTHANDLER", "1")
    environment.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    environment.setdefault("TORCH_FR_BUFFER_SIZE", "2000")
    environment.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")
    command = [
        sys.executable,
        "-m", "torch.distributed.run",
        "--standalone",
        "--nproc-per-node", str(len(devices)),
        "-m", "look.training.worker",
        "--project-root", str(project_root),
        "--payload", str(payload_path),
        "--stage", stage,
    ]
    process = subprocess.Popen(command, cwd=project_root, env=environment, start_new_session=True)
    received_signal: int | None = None

    def forward_signal(signum, _frame) -> None:
        nonlocal received_signal
        received_signal = signum
        if process.poll() is None:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass

    handled_signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    previous_handlers = {signum: signal.getsignal(signum) for signum in handled_signals}
    for signum in handled_signals:
        signal.signal(signum, forward_signal)
    try:
        return_code = process.wait()
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    if received_signal == signal.SIGINT:
        raise KeyboardInterrupt
    if received_signal is not None:
        raise SystemExit(128 + received_signal)
    if return_code:
        raise RuntimeError(f"DDP stage {stage!r} failed with exit code {return_code}")
