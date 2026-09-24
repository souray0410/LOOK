import os,pathlib,pytest
from look.runtime.rotation_gate import gpu_count,account_usage,existing_lock,validate_claim,validate_completed_claim

def test_counts_multigpu_and_rejects_unknown_gres():
 assert gpu_count('gpu:a100:4')==4
 assert gpu_count('gpu:a100:2(IDX:0-1),gpu:v100:1')==3
 with pytest.raises(ValueError):gpu_count('gpu:a100')

def test_usage_requires_unique_owner():
 total,counts=account_usage(['1|gpu:a100:4'],{'1':{'LOOK'}},['LOOK'])
 assert total==4 and counts['LOOK']==4
 with pytest.raises(ValueError,match='ambiguous'):account_usage(['1|gpu:a100:1'],{'1':{'LOOK','RB'}},['LOOK','RB'])
 with pytest.raises(ValueError,match='absent'):account_usage(['2|gpu:a100:1'],{},['LOOK'])

def test_lock_must_exist_and_not_symlink(tmp_path):
 with pytest.raises(FileNotFoundError):
  with existing_lock(tmp_path/'missing'):pass
 target=tmp_path/'lock';target.write_text('');link=tmp_path/'link';link.symlink_to(target)
 with pytest.raises(OSError):
  with existing_lock(link):pass
 with existing_lock(target):pass

def test_claim_binds_job_task_owner_and_cursor():
 c={'schema':'look_rotation_claim_v1','state':'submitted','job_id':'7','task':'t','owner':'u','start_batch':3}
 validate_claim(c,job='7',task='t',owner='u',start_batch=3)
 for key,value in [('job','8'),('task','x'),('owner','v'),('start_batch',4)]:
  args={'job':'7','task':'t','owner':'u','start_batch':3};args[key]=value
  with pytest.raises(ValueError):validate_claim(c,**args)


def test_completed_claim_binds_receipt_and_terminal():
 c={"schema":"look_rotation_claim_v1","state":"completed","job_id":"7","task":"t","owner":"u","start_batch":3,"terminal":{"state":"COMPLETED","exit_code":"0:0"},"receipt_sha256":"abc"}
 validate_completed_claim(c,job="7",task="t",owner="u",start_batch=3,receipt_sha256="abc")
 c["receipt_sha256"]="changed"
 with pytest.raises(ValueError,match="receipt"):validate_completed_claim(c,job="7",task="t",owner="u",start_batch=3,receipt_sha256="abc")


def test_signed_non_project_counts_global_not_project():
 total,counts=account_usage(["9|gpu:a100:2"],{},["LOOK"],non_project={"9"})
 assert total==2 and counts["LOOK"]==0
