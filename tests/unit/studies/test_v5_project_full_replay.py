import pytest
from look.studies import v5_project_full_replay as subject

def test_rejects_nonpositive_measured_budget(tmp_path):
 with pytest.raises(ValueError,match='Measured GPU budget'):
  subject.execute(source_run=tmp_path,checkpoint=tmp_path/'x',output=tmp_path/'o',inputs_factory=None,device='cuda:0',gpu_budget_bytes=0)

def test_contract_pins_formal_sources():
 assert subject.FRAMEWORK=='1287681c08846e11364c81653048435482e772a7'
 assert subject.MODELS=='cc16e74a8cfc705d69b3d31efe2daeec9404471f'

def test_claim_fails_closed_without_scheduler_environment(monkeypatch):
 monkeypatch.delenv('LOOK_ROTATION_CLAIM',raising=False)
 with pytest.raises(KeyError): subject.validate_claim()
