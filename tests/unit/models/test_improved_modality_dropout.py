import torch
from torch.nn import functional as F

from look.models.improved_modality_dropout import (
    EmptyToken, ImprovedDropoutFusion, SigmoidContrastiveLoss,
    simultaneous_modality_dropout_loss, multimodal_contrastive_loss,
)


def test_empty_token_is_learnable_and_zero_initialized():
    token=EmptyToken(5)
    assert token.token.requires_grad
    torch.testing.assert_close(token(3),torch.zeros(3,5),rtol=0,atol=0)


def test_missing_state_ignores_declared_absent_feature():
    torch.manual_seed(3)
    model=ImprovedDropoutFusion(4,4,2,dropout=0,classifier_dropout=0)
    oct_feature=torch.randn(3,4);cfp_feature=torch.randn(3,4)
    a=model.fused_feature(oct_feature,cfp_feature,state="oct_missing")
    b=model.fused_feature(torch.full_like(oct_feature,99),cfp_feature,state="oct_missing")
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    c=model.fused_feature(oct_feature,cfp_feature,state="cfp_missing")
    d=model.fused_feature(oct_feature,torch.full_like(cfp_feature,99),state="cfp_missing")
    torch.testing.assert_close(c,d,rtol=0,atol=0)


def test_simultaneous_target_loss_is_complete_plus_two_missing_losses():
    torch.manual_seed(5)
    model=ImprovedDropoutFusion(4,4,2,dropout=0,classifier_dropout=0)
    oct_feature=torch.randn(4,4);cfp_feature=torch.randn(4,4);labels=torch.tensor([0,1,1,0])
    total,logits=simultaneous_modality_dropout_loss(model,oct_feature,cfp_feature,labels,missing_weight=1.)
    expected=sum(F.cross_entropy(logits[s],labels) for s in ("complete","oct_missing","cfp_missing"))
    torch.testing.assert_close(total,expected,rtol=0,atol=0)


def test_sigmoid_contrastive_matches_pinned_author_bce_primitive():
    torch.manual_seed(7)
    q=F.normalize(torch.randn(4,6),dim=1);k=F.normalize(torch.randn(4,6),dim=1);labels=torch.tensor([0,0,1,1])
    criterion=SigmoidContrastiveLoss()
    got=criterion(q,k,labels)
    targets=(labels[:,None]==labels[None,:]).float()
    logits=(q@k.T)*torch.exp(criterion.log_scale)+criterion.bias
    expected=F.binary_cross_entropy_with_logits(logits,targets)
    torch.testing.assert_close(got,expected,rtol=0,atol=0)


def test_multimodal_contrastive_is_three_pairs():
    torch.manual_seed(11);labels=torch.tensor([0,1,0])
    a,b,f=[F.normalize(torch.randn(3,5),dim=1) for _ in range(3)]
    criterion=SigmoidContrastiveLoss()
    got=multimodal_contrastive_loss(a,b,f,labels,criterion)
    expected=criterion(a,b,labels)+criterion(a,f,labels)+criterion(b,f,labels)
    torch.testing.assert_close(got,expected,rtol=0,atol=0)


def test_provenance_is_source_pinned_and_project_adaptation_is_explicit():
    model=ImprovedDropoutFusion(8,8,2)
    assert model.provenance["author_commit"]=="8040d96b2dec48cf8fc7d13b45e15af0d07952ed"
    assert model.provenance["author_license"]=="MIT"
    assert "not author image+tabular" in model.provenance["adaptation"]
