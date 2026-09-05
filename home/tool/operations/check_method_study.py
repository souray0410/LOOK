#!/usr/bin/env python3
"""Read-only method queue status; process status must be checked separately."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('summary',type=Path);a=p.parse_args()
    s=json.loads(a.summary.read_text())
    print(json.dumps({k:s.get(k) for k in ('status','updated_at_utc','active_case','error','test_access')},indent=2))
    print(f"completed={len(s['completed_cases'])}/9")
    for key in ('parent_summary','suffix_summary'):
        v=json.loads(Path(s[key]).read_text());print(key,v['status'],len(v.get('completed_stages',v.get('completed_cases',{}))))

if __name__=='__main__':main()
