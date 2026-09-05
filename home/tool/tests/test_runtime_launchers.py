"""Exercise the real launch command through a simulated pre-existing tmux server."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class RuntimeLaunchTests(unittest.TestCase):
    def launch(self, launcher, overrides=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / 'python'
            python.write_text(f'#!{sys.executable}\n' + '''import json,os,resource,sys
if sys.argv[1] == '-c': print(os.environ['PROBE_ROOT'])
else: print(json.dumps({'env':dict(os.environ),'nofile':resource.getrlimit(resource.RLIMIT_NOFILE)[0],'args':sys.argv[1:]}))
''')
            python.chmod(0o755)
            tmux = root / 'tmux'
            tmux.write_text(f'#!{sys.executable}\n' + '''import os,resource,subprocess,sys
from pathlib import Path
stamp=Path(os.environ['PROBE_ROOT'])/'started'
if sys.argv[1]=='has-session': sys.exit(0 if stamp.exists() else 1)
assert sys.argv[1]=='new-session'
# Model an older server that kept unsafe transport settings and a low limit.
env=dict(os.environ,NCCL_P2P_DISABLE='0',NCCL_CUMEM_ENABLE='1',NCCL_CUMEM_HOST_ENABLE='1')
soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE,(1024,hard))
subprocess.run(['bash','-c',sys.argv[-1]],env=env,check=True)
stamp.touch()
''')
            tmux.chmod(0o755)
            env = dict(os.environ, PROBE_ROOT=str(root), PATH=str(root)+os.pathsep+os.environ['PATH'])
            for key in ('NCCL_P2P_DISABLE','NCCL_SHM_DISABLE','NCCL_CUMEM_ENABLE','NCCL_CUMEM_HOST_ENABLE','LOOK_NOFILE_LIMIT'):
                env.pop(key, None)
            env.update(overrides or {})
            script = Path(__file__).resolve().parents[1] / 'operations' / launcher
            result = subprocess.run(['bash',str(script),'--project-root',str(root),'--python',str(python),'--gpus','0,1'],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return json.loads(next((root/'logs').glob('*.log')).read_text())

    def test_defaults_override_stale_tmux_environment(self):
        for launcher in ('start_unified_study.sh','start_suffix_study.sh'):
            with self.subTest(launcher=launcher):
                result = self.launch(launcher)
                self.assertEqual(result['nofile'],65536)
                for key,value in {'NCCL_P2P_DISABLE':'1','NCCL_SHM_DISABLE':'0','NCCL_CUMEM_ENABLE':'0','NCCL_CUMEM_HOST_ENABLE':'0','TORCH_FR_BUFFER_SIZE':'2000','TORCH_NCCL_DUMP_ON_TIMEOUT':'1'}.items():
                    self.assertEqual(result['env'][key],value)

    def test_explicit_deployment_override_is_preserved(self):
        for launcher in ('start_unified_study.sh','start_suffix_study.sh'):
            with self.subTest(launcher=launcher):
                result=self.launch(launcher,{'NCCL_CUMEM_ENABLE':'1','NCCL_CUMEM_HOST_ENABLE':'1','LOOK_NOFILE_LIMIT':'32768'})
                self.assertEqual(result['nofile'],32768)
                self.assertEqual(result['env']['NCCL_CUMEM_ENABLE'],'1')
                self.assertEqual(result['env']['NCCL_CUMEM_HOST_ENABLE'],'1')


if __name__ == '__main__':
    unittest.main()
