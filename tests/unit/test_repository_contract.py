import importlib.util
from pathlib import Path

def test_repository_contract():
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location("research_manage",root/"scripts/manage.py")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert m.check()["structure"]


def test_application_paths_use_new_layout():
    from look_core.paths import ProjectPaths
    from look_core.defaults import DeploymentDefaults
    root=Path(__file__).resolve().parents[2]
    paths=ProjectPaths.load(project_root=root,environ={})
    assert paths.tool_root==root/"src"
    assert paths.pipeline_root==root/"scripts"
    assert DeploymentDefaults.load(root).ukb_label_root.name=="labels"
