"""Finish the existing immutable LOOK cache chain with its full matched suite."""
import argparse
import json
from pathlib import Path
import shutil
import textwrap

from scripts.build_v5_monitor_runtime_package import build, write_json
from look.runtime.v5_monitor_runtime_pin import file_sha256


def replace_once(text, old, new):
    if text.count(old)!=1:raise ValueError(f'Unexpected source template: {old[:90]}')
    return text.replace(old,new)


def prepare(source, staging, destination, look_source):
    source,staging,destination,look_source=map(lambda p:Path(p).resolve(),
        (source,staging,destination,look_source))
    if staging.exists() or destination.exists():raise FileExistsError('Immutable destination already exists')
    shutil.copytree(source,staging,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    old='/ibex/user/mengh/LOOK/look_v24_mhd_candidate_d6a7ae3_20260924/candidate/look_cataract_middle_v5_feature_chain_sharded_v24'
    for path in staging.iterdir():
        if path.is_file() and path.suffix in ('.py','.sh','.sbatch'):
            path.write_text(path.read_text().replace(old,str(destination)).replace(str(source),str(destination)))
    shutil.rmtree(staging/'source/look')
    shutil.copytree(look_source,staging/'source/look',ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc'))
    chain_path=staging/'chain_manager.py';chain=chain_path.read_text()
    start=chain.index('  policy=read(POL)\n');end=chain.index('  if look>=',start)
    block=chain[start:end]
    helper='def capacity():\n'+textwrap.dedent(block).replace('\n','\n ').rstrip()+'\n'
    # Restore one-space body indentation after dedenting the original with-block.
    helper='def capacity():\n'+textwrap.indent(textwrap.dedent(block),' ')+' return policy,allgpu,look\n'
    chain=chain[:start]+'  policy,allgpu,look=capacity()\n'+chain[end:]
    chain=chain.replace('def main(pred,claim):',helper+'def main(pred,claim):')
    chain=replace_once(chain,'new=min(total-done,by_time,by_disk,512)','new=min(total-done,by_time,by_disk)')
    chain=chain.replace("str(PKG/'stage_monitor.sh'),'pca'","str(PKG/'monitor.sh'),'--stage','pca'")
    chain=replace_once(chain,'def main(pred,claim):','def _main(pred,claim):')
    guard="""def main(pred,claim):
 import fcntl
 with (PKG/f'transition-{pred}.lock').open('a') as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
  prior=PKG/f'transition-{pred}.json'
  if prior.exists() and read(prior).get('state') in ('pca_submitted','successor_submitted'):return
  return _main(pred,claim)
"""
    chain=chain.replace("if __name__=='__main__':",guard+"if __name__=='__main__':")
    chain_path.write_text(chain)
    decision=staging/'run_decision.sbatch'
    text=decision.read_text().replace('look_v5_mid_dec1','look_v5_matched_suite')
    text=text.replace('look.studies.v5_project_sharded_first_decision','look.studies.v5_project_sharded_matched_suite')
    text=text.replace('look_cataract_middle_v5_first_prefix_v1','look_cataract_middle_v5_matched_suite_v1')
    text=replace_once(text,'test -n "${LOOK_ROTATION_CLAIM:-}" || exit 64',
        'test -n "${LOOK_ROTATION_CLAIM:-}" || exit 64\nexport LOOK_LEASE_END_EPOCH=$(( $(date +%s) + 171900 ))')
    decision.write_text(text)
    for path in staging.glob('*.py'):compile(path.read_text(),str(path),'exec')
    verifier=look_source/'src/look/runtime/v5_monitor_runtime_pin.py'
    receipt=build(staging,destination,verifier)
    bootstrap=destination/'monitor.sh';text=bootstrap.read_text()
    text=replace_once(text,'exec python3 "$B/chain_manager.py" "$@"',
        'export PYTHONPATH="$B:$PYTHONPATH"\nif [ "${1:-}" = "--stage" ]; then\n shift\n exec python3 -m look.runtime.v5_matched_stage "$@"\nfi\nexec python3 "$B/chain_manager.py" "$@"')
    bootstrap.write_text(text)
    receipt.update(monitor_sha256=file_sha256(bootstrap),scope='original_full_matched_suite',
        source_package=str(source),look_source=str(look_source),
        remaining_materialization_budget='measured_time_and_storage_no_512_cap',
        gpu_predecessor_preserved='52429877')
    write_json(destination.parent/'package_receipt.json',receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('source','staging','destination','look-source'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.source,args.staging,args.destination,args.look_source),sort_keys=True))
