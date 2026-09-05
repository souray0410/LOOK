#!/usr/bin/env python3
"""Refresh a validation-only advisor snapshot independently of GPU queues."""
import argparse
import json
from pathlib import Path
from look_core.start_study import read_json,file_record
from look_core.method_report import write_method_report
from look_core.state import atomic_write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method-summary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use a fresh snapshot directory; historical reports are immutable')
    summary=read_json(a.method_summary)
    if summary.get('test_access') is not False:raise ValueError('Only sealed development reporting is allowed')
    manifest=write_method_report(summary,a.output)
    atomic_write_json(summary,a.output/'method_summary_snapshot.json')
    manifest['method_summary_snapshot']=file_record(a.output/'method_summary_snapshot.json')
    atomic_write_json(manifest,a.output/'report_manifest.json')
    atomic_write_json(dict(status='complete',manifest=file_record(a.output/'report_manifest.json'),
        pdf=file_record(a.output/'advisor_report.pdf')),a.output/'report_complete.json')
    print(json.dumps(dict(output=str(a.output),manifest=manifest),indent=2))


if __name__=='__main__':main()
