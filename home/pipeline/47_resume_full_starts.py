#!/usr/bin/env python3
"""Verify 61 completed cases, then run the authorized 182-case remainder on two GPUs."""
import argparse
from pathlib import Path
from look_core.resumed_starts import run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('project-root','config','output','acceptance','lock-path'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args()
    run(a.project_root,a.config,a.output,a.acceptance,a.lock_path,prepare_only=a.prepare_only)


if __name__=='__main__':main()
