from __future__ import annotations

import json
import hashlib
import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_release_contains_general_project_standard() -> None:
    standard = PROJECT_ROOT.parent / "GENERAL_PROJECT_STANDARD.md"
    text = standard.read_text(encoding="utf-8")
    assert "General Reproducible Research Project Standard" in text
    assert "Artifact And Resume Contract" in text
    sync_script = (PROJECT_ROOT / "pipeline/2_sync_project_to_remote.sh").read_text(
        encoding="utf-8"
    )
    assert "GENERAL_PROJECT_STANDARD.md" in sync_script


def test_required_project_contract_is_declared() -> None:
    config = json.loads((PROJECT_ROOT / "project.json").read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "project_name",
        "release_id",
        "selected_task_profile",
        "task_selection_manifest",
        "deployment_defaults",
        "management_standard",
        "directories",
        "compute",
    }
    assert required <= config.keys()
    assert config["selected_task_profile"] == "glaucoma_all_evidence"
    defaults = config["deployment_defaults"]
    assert "/task_scout/glaucoma_all_evidence/primary/" in defaults["labels_csv"]
    assert "/task_scout/glaucoma_all_evidence/natural/" in defaults["natural_labels_csv"]
    standard = (PROJECT_ROOT / config["management_standard"]["document"]).resolve()
    assert standard.is_file()
    assert "Acceptance Gates" in standard.read_text(encoding="utf-8")
    assert config["compute"] == {
        "default_gpu_devices": [0, 1],
        "maximum_visible_gpus": 2,
        "execution": "sequential_ddp",
    }


def test_pipeline_numbering_preserves_shared_steps_and_new_iteration_range() -> None:
    entries = [path for path in (PROJECT_ROOT / "pipeline").iterdir() if path.is_file()]
    numbered = []
    for path in entries:
        match = re.fullmatch(r"(\d+)_[A-Za-z0-9_]+\.(?:py|sh|ipynb)", path.name)
        assert match, f"Unnumbered or invalid pipeline entry: {path.name}"
        numbered.append(int(match.group(1)))
    assert sorted(numbered) == [*range(1, 12), *range(21, 33), 38, 39]
    assert len(numbered) == len(set(numbered))
    history = PROJECT_ROOT / "pipeline/history/2026_09_03_08_30_00"
    assert sorted(int(path.name.split("_", 1)[0]) for path in history.glob("[0-9]*_*")) == list(range(12, 21))


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
            or path == PROJECT_ROOT / "references/260810_source.txt"
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        relative = str(path.relative_to(PROJECT_ROOT))
        assert not any(token in text or token in relative for token in forbidden)
        assert "sys.path" + ".insert" not in text and "sys.path" + ".append" not in text


def test_package_names_are_canonical() -> None:
    import MHD_Project
    import look_core

    assert MHD_Project.__name__ == "MHD_Project"
    assert look_core.__name__ == "look_core"


def test_mhd_core_file_is_the_reviewed_compute_only_version() -> None:
    framework = PROJECT_ROOT / "tool/MHD_Project/MHD_Framework_V4.py"
    assert hashlib.sha256(framework.read_bytes()).hexdigest() == (
        "6f499046ff3068b76c01a69e25d11398f2d2c0ff3ebb18f2faf825ff0199e6ba"
    )
    assert "generate_mermaid" not in framework.read_text(encoding="utf-8")


def test_release_uses_only_v4_graph_training_contract() -> None:
    package = PROJECT_ROOT / "tool/MHD_Project"
    assert not (package / "MHD_Framework_V3.py").exists()
    assert not (package / "MHD_Utils_V3.py").exists()
    assert not (package / "MHD_Compatibility_V3_to_V4.py").exists()
    training = (PROJECT_ROOT / "tool/look_core/train.py").read_text(encoding="utf-8")
    assert "MHD_Trainer(" in training
    assert "register_" + "epoch_node" not in training
    assert "criteria=selection_criterion(config.primary_metric)" in training
    assert "criteria_node" not in training
    assert "criteria_levels" not in training
    assert "graph._backward(" not in training
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


def test_notebook_reference_is_verbatim_and_hash_checked():
    root = PROJECT_ROOT / "references"
    manifest = json.loads((root / "260810_manifest.json").read_text())
    assert hashlib.sha256((root / "260810_source.txt").read_bytes()).hexdigest() == manifest["sha256"]
