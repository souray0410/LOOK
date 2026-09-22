import json
import pytest
import torch
from look.methods import operator, joint
from look.methods.operator import FullFeaturePCA
from look.runtime.state import file_sha256, stable_hash
from look.studies.v5_pca_migration import migrate


def fixture(tmp):
    identity=dict(case='case', host_best_sha256='old', max_rank=2,
                  pca_code_sha256=file_sha256(operator.__file__),
                  joint_code_sha256=file_sha256(joint.__file__))
    bank_id=stable_hash(identity)[:16]; source=tmp/'old'/bank_id;source.mkdir(parents=True)
    p=source/'x_x1.pt'
    basis=FullFeaturePCA('x',1,(3,),(3,),torch.zeros(3),torch.ones(3),torch.zeros(3),
        torch.eye(3)[:2],torch.tensor([.7,.2]),10,2.,100,bank_id+'/x_x1')
    basis.save(p)
    manifest=source/'bank_manifest.json'
    manifest.write_text(json.dumps(dict(bank_id=bank_id,identity=identity,entries=[
        dict(node='x',factor=1,source_id=basis.source_id,path=str(p),sha256=file_sha256(p))])))
    checkpoint=tmp/'new.pt';torch.save({'model':{}},checkpoint)
    proof=tmp/'proof.json'
    proof.write_text(json.dumps(dict(schema='look_node_feature_replay_v1',state='accepted',
        test_access=False,all_site_values_bitwise_equal=True,framework_api='V5',
        original_bank_sha256=file_sha256(manifest),source_checkpoint_sha256='old',
        target_checkpoint_sha256=file_sha256(checkpoint),sites=['x'],counts=dict(train=5,development=3),
        imported_look_source_sha256={'look.methods.operator':file_sha256(operator.__file__),
                                     'look.methods.joint':file_sha256(joint.__file__)})))
    return manifest,tmp/'new',dict(feature_receipt=proof,feature_receipt_sha256=file_sha256(proof),
        target_checkpoint=checkpoint,target_checkpoint_sha256=file_sha256(checkpoint)),basis


def test_preserves_values_and_rebinds_all_references(tmp_path):
    source,out,kw,basis=fixture(tmp_path)
    dest=migrate(source,out,**kw)
    assert migrate(source,out,**kw)==dest
    bank=json.loads((dest/'bank_manifest.json').read_text())
    entry=bank['entries'][0];loaded=FullFeaturePCA.load(entry['path'])
    assert loaded.source_id!=basis.source_id
    assert loaded.sample_count==basis.sample_count
    x=torch.randn(4,3)
    assert torch.equal(x@loaded.components.T,x@basis.components.T)
    assert entry['sha256']==file_sha256(entry['path'])
    assert json.loads((dest/'migration.json').read_text())['contract']['production_dispatch_authorized'] is False


@pytest.mark.parametrize('change', [dict(all_site_values_bitwise_equal=False),
    dict(sites=[]),dict(source_checkpoint_sha256='wrong'),dict(test_access=True)])
def test_incomplete_proof_is_rejected(tmp_path,change):
    source,out,kw,_=fixture(tmp_path)
    p=kw['feature_receipt'];d=json.loads(p.read_text());d.update(change);p.write_text(json.dumps(d))
    kw['feature_receipt_sha256']=file_sha256(p)
    with pytest.raises(ValueError):migrate(source,out,**kw)
    assert not out.exists()


def test_tampered_target_not_reused(tmp_path):
    source,out,kw,_=fixture(tmp_path);dest=migrate(source,out,**kw)
    (dest/'x_x1.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='converted bank changed'):migrate(source,out,**kw)


def test_selected_family_rebinds_without_changing_operator(tmp_path):
    from dataclasses import asdict
    from look.methods.operator import LOOKArtifact
    from look.methods.affine_family import FamilyArtifact, FamilyMap
    from look.methods.linear_operator import fingerprint
    from look.studies.family_search_case import load_bank
    from look.studies.v5_pca_migration import migrate_selected_family
    source,out,kw,basis=fixture(tmp_path);pca=migrate(source,out,**kw)
    base=LOOKArtifact('x','oct_missing','zero',1,2,(3,),(3,),basis.mean,basis.std,
        basis.pca_mean,basis.components,torch.zeros(2,2),torch.zeros(2),.1,0.,1.,
        pca_source_id=basis.source_id)
    mapping=FamilyMap(torch.zeros(3),torch.ones(3),torch.eye(3),torch.empty(0),
                      'pca_free_mean',.1,2,10,{})
    original=FamilyArtifact(base,mapping);records=[original.record()]
    bank=tmp_path/'bank.pt'
    torch.save(dict(identity=dict(identity='case',bases={'x':fingerprint(asdict(basis))}),
                    bank=records,sha256=fingerprint(records)),bank)
    target=tmp_path/'selected'/'bank.pt'
    args=dict(source_sha256=file_sha256(bank),source_pca_manifest=source,
              converted_pca_directory=pca)
    migrate_selected_family(bank,target,**args)
    assert migrate_selected_family(bank,target,**args)==target
    loaded,=load_bank(target.parent)
    assert loaded.base.pca_source_id!=original.base.pca_source_id
    x=torch.randn(8,3)
    assert torch.equal(original.apply_feature(x),loaded.apply_feature(x))
    torch.save({},bank)
    with pytest.raises(ValueError,match='Original selected bank changed'):
        migrate_selected_family(bank,target,**args)
