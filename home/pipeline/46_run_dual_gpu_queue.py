#!/usr/bin/env python3
"""Run only explicit independent LOOK cases, one per physical GPU 0/1."""
import argparse
from pathlib import Path
from look_core.dual_queue import supervise, worker


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', required=True, type=Path)
    p.add_argument('--plan', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--lock-path', type=Path)
    p.add_argument('--worker')
    a = p.parse_args()
    if a.worker:
        worker(a.project_root, a.plan, a.output, a.worker)
    else:
        if not a.lock_path: p.error('--lock-path is required; use the shared bounded_gpu.lock')
        supervise(a.project_root, a.plan, a.output, a.lock_path)


if __name__ == '__main__': main()
