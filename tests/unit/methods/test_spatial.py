import dataclasses
import numpy as np
import pytest
import torch
from look.methods.spatial import reduce_spatial,METHODS
from look.methods.operator import LOOKArtifact,apply_artifact,downsample_flatten,FullFeaturePCA


@pytest.mark.parametrize('method',METHODS)
def test_linear_and_gradient(method):
    torch.manual_seed(8);x=torch.randn(2,3,7,9,dtype=torch.float64,requires_grad=True);y=torch.randn_like(x)
    factor=1 if method=='direct' else 3
    torch.testing.assert_close(reduce_spatial(2*x-3*y,factor,method),2*reduce_spatial(x,factor,method)-3*reduce_spatial(y,factor,method))
    z=reduce_spatial(x,factor,method);z.square().sum().backward();assert torch.isfinite(x.grad).all()
    if method=='average_pool':
        # First adaptive bin is rows [0:4], columns [0:3] for 7x9 -> 2x3.
        torch.testing.assert_close(z[:,:,0,0],x[:,:,:4,:3].mean((-2,-1)))


@pytest.mark.parametrize('method',METHODS)
def test_artifact_fit_inference_identity_and_reload(method,tmp_path):
    factor=1 if method=='direct' else 2;x=torch.arange(36,dtype=torch.float32).reshape(1,1,6,6)
    flat,shape=downsample_flatten(x,factor,method);d=flat.shape[1]
    a=LOOKArtifact(node_name='site',missing_pattern='oct_missing',filling_strategy='normalized_mean',factor=factor,latent_dim=d,
        feature_shape=(1,6,6),downsample_shape=shape,mean=torch.zeros(d),std=torch.ones(d),pca_mean=torch.zeros(d),
        components=torch.eye(d),weight=torch.eye(d),bias=torch.zeros(d),ridge_lambda=1.,train_r2=0.,train_mse=0.,spatial_method=method)
    expected=flat.reshape(1,*shape)
    if factor!=1:expected=torch.nn.functional.interpolate(expected,size=(6,6),mode='bilinear',align_corners=False)
    torch.testing.assert_close(apply_artifact(x,a),x+expected)
    a.save(tmp_path/'a.pt');torch.testing.assert_close(apply_artifact(x,LOOKArtifact.load(tmp_path/'a.pt')),x+expected)
    old=dataclasses.asdict(a);old.pop('spatial_method');torch.save(old,tmp_path/'old.pt')
    assert LOOKArtifact.load(tmp_path/'old.pt').spatial_method=='interpolate'


def test_direct_rejects_hidden_compression():
    with pytest.raises(ValueError):reduce_spatial(torch.ones(2,3,8,8),4,'direct')


@pytest.mark.parametrize('method',METHODS)
def test_real_mhd_writeback_matches_explicit_input_correction(method):
    from look.models.graph import build_resnet50_mhd_graph
    from look.methods.operator import forward_with_look
    from look.methods.joint import read_site
    torch.manual_seed(18)
    graph=build_resnet50_mhd_graph('layer3',batch_size=1,pretrained=False,device='cpu').eval()
    ids=[(n.id,n.name) for n in graph.nodes]
    x=torch.randn(1,2,3,32,32);y=torch.randn_like(x)
    with torch.no_grad():
        joined=forward_with_look(graph,x,y,stop_node='joint_input').clone()
        factor=1 if method=='direct' else 4;flat,shape=downsample_flatten(joined,factor,method);d=flat.shape[1]
        a=LOOKArtifact(node_name='joint_input',missing_pattern='oct_missing',filling_strategy='normalized_mean',factor=factor,latent_dim=2,
            feature_shape=tuple(joined.shape[1:]),downsample_shape=shape,mean=torch.zeros(d),std=torch.ones(d),pca_mean=torch.zeros(d),
            components=torch.randn(2,d)/d**.5,weight=torch.eye(2)*.1,bias=torch.ones(2)*.03,
            ridge_lambda=1.,train_r2=0.,train_mse=0.,spatial_method=method)
        expected=apply_artifact(joined,a)
        manual=forward_with_look(graph,expected[:,:3].reshape_as(x),expected[:,3:].reshape_as(y),stop_node='joint_stem').clone()
        actual=forward_with_look(graph,x,y,[a],stop_node='joint_stem').clone()
    torch.testing.assert_close(actual,manual,rtol=0,atol=0)
    assert ids==[(n.id,n.name) for n in graph.nodes]
