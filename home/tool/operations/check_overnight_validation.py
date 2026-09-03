#!/usr/bin/env python3
"""Read overnight progress without loading training packages or changing state."""
import argparse
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
parser.add_argument("--runs-root", type=Path)
parser.add_argument("--session", default="look-overnight")
parser.add_argument("--follow", action="store_true")
args = parser.parse_args()
config = json.loads((args.project_root / "project.json").read_text())
runs = args.runs_root or Path(config["deployment_defaults"]["data_root"]) / config["directories"]["runs"]
active = subprocess.run(["tmux", "has-session", "-t", args.session], capture_output=True).returncode == 0
print(f"session={args.session} running={active}")
summaries = sorted((runs / "overnight").glob("*/summary.json"), key=lambda p: p.stat().st_mtime)
if summaries:
    path = summaries[-1]
    result = json.loads(path.read_text())
    print(f"summary={path}")
    for key in ("status", "stage", "updated_at_utc", "reason", "error", "quality_gate_checks", "ranking", "look_started"):
        if key in result:
            print(f"{key}={json.dumps(result[key])}")
progress = sorted((runs / "sweeps").glob("validation__*/progress.json"), key=lambda p: p.stat().st_mtime)
if progress:
    result = json.loads(progress[-1].read_text())
    print(f"latest_sweep={progress[-1].parent.name}")
    for key in ("status", "completed_count", "configuration_count", "current_index", "current_configuration"):
        print(f"{key}={json.dumps(result.get(key))}")
histories = sorted((runs / "backbones").glob("*/history.json"), key=lambda p: p.stat().st_mtime)
if histories:
    rows = json.loads(histories[-1].read_text())
    if rows:
        print(f"latest_backbone={histories[-1].parent.name}")
        print(json.dumps(rows[-1], indent=2))
subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"])
logs = sorted((runs / "logs").glob(f"{args.session}_*.log"), key=lambda p: p.stat().st_mtime)
if logs:
    print(f"log={logs[-1]}", flush=True)
    command = ["tail", "-n", "25"]
    if args.follow:
        command.append("-F")
    subprocess.run([*command, str(logs[-1])])
