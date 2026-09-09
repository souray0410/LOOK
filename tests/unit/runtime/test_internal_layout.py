import importlib.util
import shutil
from pathlib import Path
import pytest

def test_internal_layout_and_flat_module_rejection(tmp_path):
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("internal_layout", root / "scripts/check_layout.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.check(root)["canonical_imports"]
    for name in ("src", "tests"):
        shutil.copytree(root / name, tmp_path / name, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(root / "project.json", tmp_path / "project.json")
    package = next(p for p in (tmp_path / "src").iterdir() if p.is_dir() and (p / "__init__.py").exists())
    (package / "misplaced.py").write_text("x = 1\n")
    with pytest.raises(ValueError, match="Package root"):
        module.check(tmp_path)
