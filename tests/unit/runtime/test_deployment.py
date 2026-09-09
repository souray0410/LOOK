import json
import subprocess
import sys
from pathlib import Path

def test_arbitrary_machine_profile_from_other_cwd(tmp_path):
    root = Path(__file__).resolve().parents[3]
    project = json.loads((root / "project.json").read_text())
    profile = tmp_path / "custom.json"
    profile.write_text(json.dumps({"schema": "research_deployment_v1", "project": project["project_name"], "study": project["release_id"], "code_root": "portable/home", "data_root": "portable/data"}))
    result = json.loads(subprocess.check_output([sys.executable, str(root / "scripts/manage.py"), "preflight", "--machine", "custom", "--profile", str(profile)], cwd=tmp_path))
    assert result["paths"]["code"].startswith(str(root / "portable/home"))
    assert result["paths"]["run"].startswith(str(root / "portable/data"))
    assert not result["training_started"]
    assert not (root / "portable").exists()
