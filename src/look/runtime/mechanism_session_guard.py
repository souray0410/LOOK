"""Keep a superseded CPU session leader paused until its salloc children finish.

tmux resumes ordinary SIGSTOP automatically. Linux ptrace interrupt instead
parks only the inspected CPU dispatcher; it neither changes code nor signals
its process group. Never use this utility on a GPU worker/allocation owner.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import time
from look.runtime.mechanism_handover import process,validate_manager
from look.runtime.state import atomic_write_json

SEIZE=0x4206
INTERRUPT=0x4207
DETACH=17


def trace(request,pid):
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.ptrace(ctypes.c_uint(request),ctypes.c_int(pid),ctypes.c_void_p(),ctypes.c_void_p())==-1:
        error=ctypes.get_errno();raise OSError(error,os.strerror(error))


def park(pid):
    trace(SEIZE,pid)
    try:
        trace(INTERRUPT,pid)
        child,status=os.waitpid(pid,0x40000000)
        if child!=pid or not os.WIFSTOPPED(status) or process(pid)['state']!='t':
            raise RuntimeError('CPU dispatcher did not enter tracing stop')
    except BaseException:
        try:trace(DETACH,pid)
        except OSError:pass
        raise


def live_children(pid):
    found=[]
    for p in Path(f'/proc/{pid}/task/{pid}/children').read_text().split():
        try:
            if process(int(p))['state']!='Z':found.append(int(p))
        except FileNotFoundError:pass
    return found


def guard(pid,config,output,module="look.runtime.project_dispatch"):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    original=process(pid)
    if module not in ('look.runtime.project_dispatch','radon_bridge.runtime.project_dispatch'):raise ValueError('Invalid dispatcher module')
    if module not in original['args']:raise ValueError('Wrong dispatcher module')
    checked=dict(original,args=[v.replace(module,'look.runtime.project_dispatch') for v in original['args']])
    validate_manager(checked,config)
    # The tracer has no training duty. Unexpected termination resumes only this
    # old CPU process; monitor must then reject competing dispatcher admission.
    park(pid)
    def deferred(signum,frame):
        atomic_write_json(dict(state='exit_deferred_until_owners_finish',signal=signum,time=time.time()),out/'exit_request.json')
    signal.signal(signal.SIGTERM,deferred);signal.signal(signal.SIGINT,deferred)
    receipt=dict(state='parked',previous=original,guardian=process(os.getpid()),time=time.time(),
                 signal_scope='CPU_dispatcher_only',unchanged_children=live_children(pid))
    atomic_write_json(receipt,out/'accepted.json')
    while True:
        try:
            current=process(pid)
            if current['start']!=original['start'] or current['state']!='t':
                raise RuntimeError('Parked CPU identity or state changed')
            children=live_children(pid)
            atomic_write_json(dict(state='holding_session',children=children,time=time.time()),out/'status.json')
            if not children:
                os.kill(pid,signal.SIGKILL)
                os.waitpid(pid,0x40000000)
                atomic_write_json(dict(state='retired_after_all_children_exit',time=time.time()),out/'status.json')
                return
        except FileNotFoundError:
            atomic_write_json(dict(state='previous_exited',time=time.time()),out/'status.json');return
        time.sleep(5)


def main():
    p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--module',default='look.runtime.project_dispatch')
    a=p.parse_args();guard(a.pid,a.config,a.output,a.module)

if __name__=='__main__':main()
