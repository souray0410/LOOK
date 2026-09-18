"""Check current display terminology; no training/data/legacy-archive mutations."""
import argparse,json,re
from pathlib import Path

def check(root, rules_path=None):
    root=Path(root)
    rules=json.loads(Path(rules_path or (root/'standards/terminology.json' if (root/'standards/terminology.json').exists() else Path(__file__).with_name('terminology.json'))).read_text())
    paths=[root/'README.md']
    for base in (root/'docs/reports/current',root/'src'):
        if base.exists():paths.extend(p for p in base.rglob('*') if p.suffix in ('.md','.py','.svg','.html'))
    errors=[]
    for p in paths:
        if not p.is_file():continue
        s=p.read_text()
        for token in rules['forbidden']:
            if token in s:errors.append(str(p.relative_to(root))+': '+token)
    lock=root/'framework.lock.json'
    if lock.exists() and (root/'README.md').exists():
        named=re.search(r'固定提交 `([0-9a-f]{40})`',(root/'README.md').read_text())
        if named and named.group(1)!=json.loads(lock.read_text())['upstream_commit']:
            errors.append('README framework pin differs from framework.lock.json')
    if errors:raise ValueError('Terminology/publication check failed: '+ '; '.join(errors))
    return {'checked_files':len(paths),'known_aliases_absent':True,'scope':'current entry/reports/source text, not archived snapshots or scientific acceptance'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--rules',type=Path)
    a=p.parse_args();print(json.dumps(check(a.root,a.rules),ensure_ascii=False))
