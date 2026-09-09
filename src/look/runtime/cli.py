"""Common checkout management CLI; never starts a research queue implicitly."""
from pathlib import Path
import runpy

def main():
    for root in Path(__file__).resolve().parents:
        if (root / "project.json").is_file():
            runpy.run_path(str(root / "scripts/manage.py"), run_name="__main__")
            return
    raise SystemExit("Management commands require an editable project checkout; install with scripts/bootstrap.sh")

if __name__ == "__main__":
    main()
