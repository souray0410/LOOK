"""One read-only publication pass; invoked by existing maintenance, no scheduler."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deployment', type=Path, required=True)
    root = parser.parse_args().deployment.resolve()
    receipt = json.loads((root/'deployment.json').read_text())
    manifest = root/'source_manifest.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != receipt['source_manifest_sha256']:
        raise ValueError('Publication source manifest changed')
    source = (root/'source').resolve()
    for name, digest in json.loads(manifest.read_text()).items():
        path = (source/name).resolve()
        if not path.is_relative_to(source) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Publication source changed')
    command = json.loads((root/'command.json').read_text())
    result = subprocess.run(command['args'], cwd=command['cwd'], env=command['env'],
                            capture_output=True, text=True)
    logs = root/'attempts'; logs.mkdir(exist_ok=True)
    stamp = str(time.time_ns())
    (logs/(stamp+'.log')).write_text(result.stdout+result.stderr)
    status = dict(time=time.time(), returncode=result.returncode, log='attempts/'+stamp+'.log',
                  state='published_or_unchanged' if result.returncode == 0 else 'needs_review')
    temporary = root/'last_publication.tmp'
    temporary.write_text(json.dumps(status, indent=2)+'\n')
    temporary.replace(root/'last_publication.json')
    print(json.dumps(status))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
