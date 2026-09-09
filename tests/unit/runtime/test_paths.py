import json, shutil
from pathlib import Path
import pytest
from look.runtime.paths import ProjectPaths, load_project_config
from look.runtime.defaults import DeploymentDefaults

@pytest.fixture
def project(tmp_path):
    source=Path(__file__).resolve().parents[3]
    root=tmp_path/"project with spaces";root.mkdir()
    shutil.copy2(source/"project.json",root/"project.json")
    shutil.copytree(source/"configs",root/"configs")
    return root

def test_paths_are_independent_of_cwd_and_data_override_cascades(project,tmp_path,monkeypatch):
    first=ProjectPaths.load(project_root=project,data_root="external/data",environ={})
    other=tmp_path/"elsewhere";other.mkdir();monkeypatch.chdir(other)
    second=ProjectPaths.load(project_root=project,data_root="external/data",environ={})
    assert first==second
    assert first.data_root==project/"external/data"
    assert first.dataset_root.is_relative_to(first.data_root)
    assert first.labels_csv.is_relative_to(first.dataset_root)
    assert first.preprocess_cache_root.is_relative_to(first.data_root)
    assert first.tool_root==project/"src"
    assert not first.data_root.exists()

def test_external_profile_and_environment_override(project,tmp_path):
    shared=tmp_path/"external storage"
    profile={"project":"LOOK","data_root":str(shared)}
    (project/"configs/deployment/demo.json").write_text(json.dumps(profile))
    selected=ProjectPaths.load(project_root=project,environ={"SOURAY_MACHINE":"demo"})
    assert selected.data_root==shared/"LOOK"
    overridden=ProjectPaths.load(project_root=project,environ={"SOURAY_MACHINE":"demo","LOOK_DATA_ROOT":"my_data","LOOK_COHORT_ROOT":"my_cohort"})
    assert overridden.data_root==project/"my_data"
    assert overridden.labels_csv==project/"my_cohort/primary/reference_labels.csv"

def test_defaults_and_paths_agree(project,monkeypatch):
    monkeypatch.delenv("SOURAY_MACHINE",raising=False)
    monkeypatch.setenv("LOOK_DATA_ROOT","portable")
    defaults=DeploymentDefaults.load(project)
    assert defaults.project_root==project
    assert defaults.dataset_root==ProjectPaths.load(project).dataset_root
    assert defaults.ukb_label_root==project/"UKBiobank/ophthalmology/raw/labels"

def test_profile_traversal_is_rejected(project):
    with pytest.raises(ValueError,match="SOURAY_MACHINE"):
        load_project_config(project,{"SOURAY_MACHINE":"../other"})
