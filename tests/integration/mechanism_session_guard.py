"""Linux-only CPU test: parked parent, live child, no Slurm or training signals."""
import subprocess,sys,time,tempfile
from pathlib import Path
from look.runtime.mechanism_session_guard import park,trace,DETACH,live_children
from look.runtime.mechanism_handover import process
with tempfile.TemporaryDirectory() as folder:
    root=Path(folder);pfile=root/'parent';cfile=root/'child'
    child="import time;from pathlib import Path;p=Path("+repr(str(cfile))+");\nfor i in range(40):p.write_text(str(i));time.sleep(.05)"
    parent="import subprocess,sys,time;from pathlib import Path;subprocess.Popen([sys.executable,'-c',"+repr(child)+"]);p=Path("+repr(str(pfile))+");\nfor i in range(100):p.write_text(str(i));time.sleep(.05)"
    p=subprocess.Popen([sys.executable,'-c',parent])
    try:
        deadline=time.time()+5
        while not cfile.exists() and time.time()<deadline:time.sleep(.05)
        park(p.pid);before=pfile.read_text();c0=int(cfile.read_text())
        time.sleep(.3)
        assert process(p.pid)['state']=='t' and pfile.read_text()==before
        assert int(cfile.read_text())>c0 and live_children(p.pid)
        trace(DETACH,p.pid);time.sleep(.1)
        assert int(pfile.read_text())>int(before)
        print('accepted: parent parked; child progressed; exact parent resumed')
    finally:
        p.terminate();p.wait()
        deadline=time.time()+3
        while cfile.exists() and cfile.read_text()!='39' and time.time()<deadline:time.sleep(.05)
