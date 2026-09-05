#!/usr/bin/env python3
"""Read-only compact suffix continuation status. Does not restart or notify."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[2])
p.add_argument('--summary',type=Path)
a=p.parse_args()
config=json.loads((a.project_root/'project.json').read_text())
root=Path(config['deployment_defaults']['data_root'])/'runs/start_study'
files=[a.summary] if a.summary else sorted(root.glob('*/summary.json'))
for path in files:
    s=json.loads(path.read_text())
    parent=json.loads(Path(s['identity']['parent_study']).read_text())
    cases=list(s.get('completed_cases',{}).values())
    print(json.dumps(dict(summary=str(path),status=s['status'],updated_at_utc=s.get('updated_at_utc'),
        active_case=s.get('active_case'),completed_new=sum(c['reuse_stage'] is None for c in cases),expected_new=213,
        referenced_cases=sum(c['reuse_stage'] is not None for c in cases),expected_referenced=30,
        selected_contexts=len(s.get('selected_contexts',{})),test_access=s['test_access'],
        parent_status=parent['status'],parent_stage=parent.get('stage'),parent_completed=len(parent.get('completed_stages',{})),
        error=s.get('error')),indent=2))
if not files: print('No suffix study summary exists yet.')
