"""Avoid repeatedly streaming immutable checkpoints for monitoring only.

Acceptance first verifies every SHA. Cached monitoring is invalidated by receipt,
configuration or any evidence file's inode/size/mtime change. Formal consumption
and final analysis still rehash their inputs; this cache never grants a new run.
"""
from pathlib import Path
import json
from look.runtime.state import stable_hash,file_sha256,atomic_write_json


def monitored_acceptance(root,spec,verifier,cache_root):
    root=Path(root);cache_root=Path(cache_root);cache_root.mkdir(parents=True,exist_ok=True)
    receipt=root/'accepted.json';value=json.loads(receipt.read_text());files=value.get('files',{})
    stamps={}
    for name in files:
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()):raise ValueError('Evidence escapes run')
        st=p.stat();stamps[name]=[st.st_ino,st.st_size,st.st_mtime_ns]
    key=stable_hash(dict(run=str(root.resolve()),spec=spec));path=cache_root/(key+'.json')
    signature=stable_hash(dict(receipt=file_sha256(receipt),stamps=stamps))
    if path.exists() and json.loads(path.read_text()).get('signature')==signature:return value
    checked=verifier(root,spec)
    atomic_write_json(dict(signature=signature),path)
    return checked
