"""Run the existing dispatcher with one pinned allocation-owner entrypoint."""
import argparse
import hashlib
import json
from pathlib import Path


def allocation_command(original,config_path,name,python,memory_gib,wrapper,owner_config):
    command=original(config_path,name,python,memory_gib)
    expected=[python,'-m','look.runtime.project_dispatch','--config',str(config_path),'--allocation-owner']
    if command[-6:]!=expected:raise ValueError('Original allocation command contract changed')
    return command[:-6]+[python,wrapper,'--config',owner_config,'--allocation-owner']


def main():
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--check',action='store_true');a=p.parse_args()
    binding=json.loads(Path(a.binding).read_text())
    for path,digest in binding['pins'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Dispatcher pin changed: '+path)
    from look.runtime import project_dispatch as dispatcher
    if str(Path(dispatcher.__file__).resolve())!=binding['original_dispatcher']:raise ValueError('Wrong original dispatcher imported')
    original=dispatcher.allocation_command
    dispatcher.allocation_command=lambda path,name,python,memory_gib=128:allocation_command(
        original,path,name,python,memory_gib,binding['wrapper_program'],binding['owner_config'])
    if a.check:
        print(json.dumps(dict(state='prepared_not_activated',command=dispatcher.allocation_command(
            binding['dispatcher_config'],'look_validation_no_submission',binding['python'],128))));return
    dispatcher.daemon(binding['dispatcher_config'])


if __name__=='__main__':main()
