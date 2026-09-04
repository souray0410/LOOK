#!/usr/bin/env python3
"""Retire only the stopped e6d740a884be study after saving aggregate evidence."""
import argparse
from pathlib import Path
import json
from look_core.cli import add_runtime_arguments, resolve_runtime_arguments
from look_core.joint_cleanup import build_cleanup, execute_cleanup
from look_core.state import atomic_write_json

def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(p)
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    paths = resolve_runtime_arguments(args)
    report = build_cleanup(paths.runs_root, paths.cache_root, log=args.log)
    destination = paths.runs_root / 'maintenance' / 'joint_protocol_cleanup_preview.json'
    atomic_write_json(report, destination)
    if args.execute:
        destination = execute_cleanup(report, paths.runs_root, paths.cache_root)
    print(json.dumps(dict(audit=str(destination), status=report['status'], paths=report['paths']), indent=2))

if __name__ == '__main__':
    main()
