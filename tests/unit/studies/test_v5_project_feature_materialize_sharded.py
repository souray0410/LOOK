import inspect
from look.studies import v5_project_feature_materialize_sharded as subject

def test_sharded_cache_commits_directory_after_all_sites():
 source=inspect.getsource(subject.materialize)
 assert "partial.rename(final)" in source
 assert "for ordinal,site in enumerate(sites)" in source
 assert source.index("write_json_atomic(row,partial/'receipt.json')") < source.index("partial.rename(final)")

def test_sharded_identity_is_order_sensitive():
 assert subject.participant_sha(['a','b'])!=subject.participant_sha(['b','a'])


def test_lossless_codec_preserves_tensor_bits_and_downstream_moments(tmp_path):
 import torch
 from look.methods.operator import downsample_flatten
 x=torch.randn(5,3,8,8).transpose(2,3)
 x[0,0,0,0]=-0.0
 p=tmp_path/'site.pt.gz';subject.save_lossless_tensor(x,p);y=subject.load_lossless_tensor(p)
 assert x.dtype==y.dtype and x.shape==y.shape
 assert torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8))
 a,_=downsample_flatten(x,2);b,_=downsample_flatten(y,2)
 assert torch.equal(a,b) and torch.equal(a.T@a,b.T@b)


def test_lossless_codec_rejects_corruption(tmp_path):
 import pytest,torch
 p=tmp_path/'site.pt.gz';subject.save_lossless_tensor(torch.arange(1000),p)
 p.write_bytes(p.read_bytes()[:-12])
 with pytest.raises((EOFError,OSError,RuntimeError)):subject.load_lossless_tensor(p)
