"""Enforce the shared internal architecture without importing scientific dependencies."""
import ast
import json
import re
from pathlib import Path

ROLES = {"data", "models", "methods", "training", "evaluation", "analysis", "runtime", "studies"}
PRIMARY = {"data/dataset.py", "models/graph.py", "methods/operator.py", "training/trainer.py", "evaluation/evaluator.py", "evaluation/metrics.py", "runtime/paths.py", "runtime/cli.py", "studies/protocol.py"}

def check(root):
    root = Path(root)
    package = json.loads((root / "project.json").read_text())["package_name"]
    source = root / "src" / package
    dirs = {p.name for p in source.iterdir() if p.is_dir() and p.name != "__pycache__"}
    if dirs != ROLES:
        raise ValueError(f"Source roles differ: {dirs ^ ROLES}")
    if {p.name for p in source.glob("*.py")} != {"__init__.py", "__main__.py"}:
        raise ValueError("Package root must only contain __init__.py and __main__.py")
    for relative in PRIMARY:
        if not (source / relative).is_file():
            raise ValueError("Missing canonical module: " + relative)
    for role in ROLES:
        if not any(p.name != "__init__.py" for p in (source / role).glob("*.py")):
            raise ValueError("Empty source role: " + role)
    test_dirs = {p.name for p in (root / "tests/unit").iterdir() if p.is_dir() and p.name != "__pycache__"}
    if test_dirs != ROLES:
        raise ValueError("Unit test roles differ")
    modules = {".".join(p.relative_to(root / "src").with_suffix("").parts).removesuffix(".__init__") for p in source.rglob("*.py")}
    count = 0
    for path in source.rglob("*.py"):
        if not re.fullmatch(r"[a-z_][a-z0-9_]*", path.stem):
            raise ValueError("Non snake_case module: " + str(path))
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    raise ValueError("Use canonical absolute internal imports: " + str(path))
                imported = [node.module or ""]
            elif isinstance(node, ast.Import):
                imported = [a.name for a in node.names]
            else:
                continue
            for name in imported:
                if name == "look_core" or name.startswith("look_core."):
                    raise ValueError("Legacy import: " + name)
                if (name == package or name.startswith(package + ".")) and name not in modules:
                    raise ValueError("Unresolved internal module: " + name)
        count += 1
    return {"roles": sorted(ROLES), "python_modules": count, "canonical_imports": True}
