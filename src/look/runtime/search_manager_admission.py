"""Read-only admission using the exact pinned finite manager's AST guards."""
import argparse
import ast
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace


def assigned(node,name):
    return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)


def section(nodes,start,stop):
    first=next(i for i,n in enumerate(nodes) if assigned(n,start))
    last=next(i for i,n in enumerate(nodes[first:],first) if assigned(n,stop))
    return nodes[first:last]


def guards(source):
    tree=ast.parse(source)
    outer=next(n for n in tree.body if isinstance(n,ast.With))
    resource=section(outer.body,'info','claims')
    attempt=next(n for n in outer.body if isinstance(n,ast.Try))
    phases=next(n for n in attempt.body if isinstance(n,ast.For))
    formal=next(n for n in phases.body if isinstance(n,ast.If) and ast.unparse(n.test)=="phase == 'formal'")
    # Exact original receipt validation, then the original measured-cost assertion.
    cost_index=next(i for i,n in enumerate(formal.body) if isinstance(n,ast.Assert)
                    and 'gpu_peak_reserved_bytes' in ast.unparse(n))
    receipt=formal.body[:cost_index]
    cost=[formal.body[cost_index]]
    steps=section(phases.body,'current','cmd')
    if not resource or not receipt or not steps:raise ValueError('Original admission contract missing')
    return receipt,resource+cost+steps


def execute_nodes(nodes,scope,source):
    tree=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]))
    exec(compile(tree,source,'exec'),scope)


def check(binding_path,job):
    binding=json.loads(Path(binding_path).read_text());path=Path(binding['manager_program'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=binding['pins'][str(path)]:raise ValueError('Manager source changed')
    config_path=Path(binding['config'])
    if hashlib.sha256(config_path.read_bytes()).hexdigest()!=binding['pins'][str(config_path)]:raise ValueError('Manager configuration changed')
    c=json.loads(config_path.read_text());sys.path[:0]=c['env']['PYTHONPATH'].split(':')
    from look.runtime.state import stable_hash
    from look.studies.search_case import check_files
    spec=json.loads(Path(c['task']['spec']).read_text())
    def query(*args,**kwargs):
        kwargs.setdefault('timeout',20)
        return subprocess.check_output(*args,**kwargs)
    scope=dict(C=c,S=spec,profile=Path(c['task']['run_dir'])/'resource_profile',job=str(job),
        threads=spec.get('worker_threads',2),memory_gib=c.get('memory_gib',24),env=dict(c['env']),
        subprocess=SimpleNamespace(check_output=query),datetime=datetime,time=time,json=json,stable_hash=stable_hash,check_files=check_files)
    receipt,capacity=guards(path.read_text())
    execute_nodes(receipt,scope,str(path))
    try:execute_nodes(capacity,scope,str(path))
    except (AssertionError,subprocess.SubprocessError,OSError) as error:
        print(json.dumps(dict(state='capacity_or_liveness_not_admitted',error=repr(error),source=str(path))));return 76
    print(json.dumps(dict(state='admitted_by_original_manager_guards',source=str(path))));return 0


def main():
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--job',required=True);a=p.parse_args()
    raise SystemExit(check(a.binding,a.job))


if __name__=='__main__':main()
