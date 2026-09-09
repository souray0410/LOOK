"""Catch hidden imports of retired launchers as well as core modules."""
import importlib
import json
from pathlib import Path
import pkgutil

def test_all_public_modules_import_without_legacy_scripts():
    root = Path(__file__).resolve().parents[3]
    name = json.loads((root / "project.json").read_text())["package_name"]
    package = importlib.import_module(name)
    for item in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        if not item.name.endswith(".__main__"):
            importlib.import_module(item.name)
