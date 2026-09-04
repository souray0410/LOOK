import json
import os
from pathlib import Path
import runpy
import pytest

API=runpy.run_path(str(Path(__file__).resolve().parents[2]/'tool/operations/retire_auroc_release.py'))
inspect_tree=API['inspect_tree']


def test_retirement_counts_only_explicit_old_subtree(tmp_path):
    old=tmp_path/'old';target=old/'runs/backbones';target.mkdir(parents=True)
    (target/'best.pt').write_bytes(b'1234')
    assert inspect_tree(target,old)['bytes']==4
    with pytest.raises(ValueError,match='Outside'):inspect_tree(tmp_path/'elsewhere',old)


@pytest.mark.parametrize('name',['dataset','preprocessed_pairs','generators','primary_test','natural_test'])
def test_retirement_protects_data_and_test(tmp_path,name):
    old=tmp_path/'old';target=old/name;target.mkdir(parents=True)
    with pytest.raises(ValueError,match='Protected'):inspect_tree(target,old)


def test_retirement_rejects_symlink_and_live_lock(tmp_path):
    old=tmp_path/'old';target=old/'runs/backbones';target.mkdir(parents=True)
    link=target/'escape';link.symlink_to(tmp_path)
    with pytest.raises(ValueError,match='Symlink'):inspect_tree(target,old)
    link.unlink();(target/'stage.lock').write_text(json.dumps({'pid':os.getpid()}))
    with pytest.raises(RuntimeError,match='Live lock'):inspect_tree(target,old)


def test_retirement_rejects_test_prediction(tmp_path):
    old=tmp_path/'old';target=old/'runs/experiments';target.mkdir(parents=True)
    (target/'test_result.json').write_text('{}')
    with pytest.raises(ValueError,match='Test output'):inspect_tree(target,old)
