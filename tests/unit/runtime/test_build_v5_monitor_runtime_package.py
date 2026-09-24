import json
from pathlib import Path

import pytest

from look.runtime.v5_monitor_runtime_pin import file_sha256, verify_runtime
from tools.build_v5_monitor_runtime_package import build


def test_builder_pins_complete_source_and_detects_later_drift(tmp_path):
    source = tmp_path / "source"; source.mkdir()
    (source / "source/look/src").mkdir(parents=True)
    (source / "source/framework/src").mkdir(parents=True)
    (source / "source/models/src").mkdir(parents=True)
    (source / "source/look/src/runtime.py").write_text("VALUE = 1\n")
    (source / "chain_manager.py").write_text("print('chain')\n")
    (source / "package_manifest.json").write_text(json.dumps({
        "schema": "old", "transformed_files": {"monitor.sh": "old", "chain_manager.py": "old"},
        "test_access": False,
    }))
    (source / "monitor.sh").write_text("old\n")
    verifier = Path(__file__).resolve().parents[3] / "src/look/runtime/v5_monitor_runtime_pin.py"
    destination = tmp_path / "built"
    receipt = build(source, destination, verifier)
    assert receipt["monitor_sha256"] == file_sha256(destination / "monitor.sh")
    binding = destination / "runtime_binding.json"
    verify_runtime(binding, receipt["binding_sha256"])
    (destination / "source/look/src/runtime.py").write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="Runtime file changed"):
        verify_runtime(binding, receipt["binding_sha256"])
