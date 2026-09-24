"""Immutable policy gate executed before the accepted v20 finalizer payload."""
import argparse, json, subprocess
from pathlib import Path
from look.runtime.v5_role_policy_transition import read_json, verify_monitor_policy

def main():
    p=argparse.ArgumentParser(); p.add_argument('--binding',required=True)
    p.add_argument('--policy',required=True); p.add_argument('--payload',required=True,nargs='+')
    a=p.parse_args(); binding=read_json(a.binding)
    digest=verify_monitor_policy(a.policy,binding)
    if binding['old_finalizer']['dependency']!='afterany:52429877(unfulfilled)':
        raise ValueError('Predecessor identity changed')
    print(json.dumps({'state':'policy_accepted','policy_sha256':digest,'test_access':False},sort_keys=True),flush=True)
    subprocess.run(a.payload,check=True)
if __name__=='__main__': main()
