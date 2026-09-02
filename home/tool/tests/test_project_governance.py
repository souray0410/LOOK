from __future__ import annotations

import json
import hashlib
import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_required_project_contract_is_declared() -> None:
    config = json.loads((PROJECT_ROOT / "project.json").read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "project_name",
        "release_id",
        "deployment_defaults",
        "management_standard",
        "directories",
        "compute",
    }
    assert required <= config.keys()
    standard = (PROJECT_ROOT / config["management_standard"]["document"]).resolve()
    assert standard.is_file()
    assert "Acceptance Gates" in standard.read_text(encoding="utf-8")
    assert config["compute"] == {
        "default_gpu_devices": [0, 1],
        "maximum_visible_gpus": 2,
        "execution": "sequential_ddp",
    }


def test_pipeline_numbering_is_continuous_and_unique() -> None:
    entries = [path for path in (PROJECT_ROOT / "pipeline").iterdir() if path.is_file()]
    numbered = []
    for path in entries:
        match = re.fullmatch(r"(\d+)_[A-Za-z0-9_]+\.(?:py|sh|ipynb)", path.name)
        assert match, f"Unnumbered or invalid pipeline entry: {path.name}"
        numbered.append(int(match.group(1)))
    assert sorted(numbered) == list(range(1, len(numbered) + 1))
    assert len(numbered) == len(set(numbered))


def test_source_manifest_excludes_runtime_metadata() -> None:
    manifest = json.loads((PROJECT_ROOT / "file_manifest.json").read_text(encoding="utf-8"))
    paths = [entry["path"] for entry in manifest["files"]]
    forbidden_parts = {".git", ".venv", "__pycache__", ".pytest_cache"}
    assert not any(forbidden_parts.intersection(Path(path).parts) for path in paths)
    assert not any(path.startswith(("dataset/", "cache/", "runs/")) for path in paths)


def test_release_has_no_deprecated_package_or_compatibility_paths() -> None:
    forbidden = ("mhd_" + "framework", "mhd_" + "toolkit", "MHD_Project" + "-main")
    for path in PROJECT_ROOT.rglob("*"):
        if (
            not path.is_file()
            or {"__pycache__", ".venv", ".pytest_cache"}.intersection(path.parts)
            or path.suffix in {".pdf", ".png", ".pptx", ".pyc"}
            or path.name == "file_manifest.json"
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert not any(token in text or token in str(path) for token in forbidden)
        assert "sys.path" + ".insert" not in text and "sys.path" + ".append" not in text


def test_package_names_are_canonical() -> None:
    import MHD_Project
    import look_core

    assert MHD_Project.__name__ == "MHD_Project"
    assert look_core.__name__ == "look_core"


def test_mhd_core_file_is_the_reviewed_unchanged_version() -> None:
    framework = PROJECT_ROOT / "tool/MHD_Project/MHD_Framework_V4.py"
    assert hashlib.sha256(framework.read_bytes()).hexdigest() == (
        "32223d6aa83c7d92ee7f06391bfa3f31b27c5d69de7f421f36b620a3d21dd403"
    )


def test_release_uses_only_v4_graph_training_contract() -> None:
    package = PROJECT_ROOT / "tool/MHD_Project"
    assert not (package / "MHD_Framework_V3.py").exists()
    assert not (package / "MHD_Utils_V3.py").exists()
    assert not (package / "MHD_Compatibility_V3_to_V4.py").exists()
    training = (PROJECT_ROOT / "tool/look_core/train.py").read_text(encoding="utf-8")
    assert "graph._backward(" in training
    assert "levels=list(backward_levels)" in training
    assert "scaler.scale(loss).backward()" not in training


def test_shell_entrypoints_parse_with_bash() -> None:
    scripts = sorted((PROJECT_ROOT / "pipeline").glob("*.sh"))
    assert scripts
    for script in scripts:
        subprocess.run(["/bin/bash", "-n", str(script)], check=True)


def test_vscode_and_kernel_configuration_are_canonical() -> None:
    settings = json.loads((PROJECT_ROOT / ".vscode/settings.json").read_text(encoding="utf-8"))
    assert settings["python.defaultInterpreterPath"] == (
        "/home/mengh/LOOK/2026_08_30_11_20_47/tool/environment/.venv/bin/python"
    )
    assert settings["python.terminal.activateEnvironment"] is True
    assert not (PROJECT_ROOT / "tool/configuration/vscode/settings.json").exists()

    setup = (PROJECT_ROOT / "pipeline/3_create_environment.sh").read_text(encoding="utf-8")
    for option in ("--python", "--venv-path", "--kernel-name", "--kernel-display-name"):
        assert option in setup
    assert 'KERNEL_NAME="look"' in setup
