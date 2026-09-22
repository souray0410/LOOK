"""Install the exact current training/operations package from an authorized checkout.

The public application CPU tests do not acquire the private companion. Production
operators supply its checkout explicitly; there is no old-layout fallback.
"""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def tree_digest(root):
    files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(root.rglob('*.py')) if '__pycache__' not in p.parts}
    return hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def check(source,lock):
    source=Path(source).resolve()
    if not (source/'src/mhd_models/__init__.py').is_file():raise ValueError('Current mhd_models package required')
    if tree_digest(source/'src/mhd_models')!=lock['package_tree_sha256']:
        raise ValueError('Training package source differs from the exact lock')
    if (source/'.git').exists():
        head=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
        if head!=lock['commit']:raise ValueError('Training package commit differs')
    return source


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);p.add_argument('--check',action='store_true');a=p.parse_args()
    lock=json.loads((ROOT/'mhd_models.lock.json').read_text());source=check(a.source,lock)
    if not a.check:
        if sys.prefix==sys.base_prefix:raise ValueError('Use a dedicated virtual environment')
        subprocess.run([sys.executable,'-m','pip','install','--no-deps',str(source)],check=True)
        command='import pathlib,mhd_models;print(pathlib.Path(mhd_models.__file__).resolve().parent)'
        installed=Path(subprocess.check_output([sys.executable,'-c',command],text=True).strip())
        if tree_digest(installed)!=lock['package_tree_sha256']:raise ValueError('Installed package differs')
    print(json.dumps(dict(state='source_verified' if a.check else 'installed_verified',version=lock['version'],commit=lock['commit'],runtime_acceptance=False)))


if __name__=='__main__':main()
