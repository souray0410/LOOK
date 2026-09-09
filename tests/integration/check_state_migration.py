import hashlib,json,sys
from pathlib import Path
import torch
from look_core.graph import build_resnet50_mhd_graph,reset_and_forward

def digest_tensors(values):
 h=hashlib.sha256()
 for k,v in sorted(values.items()):
  h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

torch.set_num_threads(2);torch.manual_seed(98123)
mode,out=sys.argv[1],Path(sys.argv[2]);out.mkdir(exist_ok=True)
graph=build_resnet50_mhd_graph('layer3',batch_size=1,pretrained=False,device='cpu');graph.eval()
if mode=='before':torch.save(graph.state_dict(),out/'synthetic_state_dict.pt')
else:graph.load_state_dict(torch.load(out/'synthetic_state_dict.pt',map_location='cpu',weights_only=True),strict=True)
cfp=torch.randn(1,2,3,224,224,requires_grad=True);oct=torch.randn(1,2,3,224,224,requires_grad=True)
logits=reset_and_forward(graph,cfp,oct);loss=logits.square().sum();loss.backward()
result={'state_before':digest_tensors(graph.state_dict()),'output':logits.detach().tolist(),'gradients':digest_tensors({k:p.grad for k,p in graph.named_parameters() if p.grad is not None}),'input_gradients':digest_tensors({'cfp':cfp.grad,'oct':oct.grad}),'nodes':sorted((n.id,n.name) for n in graph.nodes)}
torch.optim.SGD(graph.parameters(),lr=1e-5).step();result['state_after_update']=digest_tensors(graph.state_dict())
(out/(mode+'.json')).write_text(json.dumps(result,indent=2)+'\n')
if mode=='after':
 assert result==json.loads((out/'before.json').read_text()),'Migration numerical mismatch'
 print('PASS: strict state_dict reload, outputs, all parameter/input gradients, node IDs, optimizer update')
