"""CPU-only source-contract audit for the pinned EyeMoSt+ author snapshot."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

EXPECTED_COMMIT="7f146ac7839567952c22a14777a1f44fb5e3d048"
EXPECTED_FILES={
    "MedIA’24/model.py":"5bca3b39e072741e7173cba165deae76e7fa0131fdda1abec0f0fadbf551003c",
    "MedIA’24/train3_trans.py":"4b80ed32b695283552e2fc28b957cd1316dfd44a834eb22fb0760767ff59dcff",
    "README.md":"8ab7007f7c7f394df6777d8c836a1487388a1e4eb23dc05a9bfce542a96482ff",
}
AUTHOR_URL="https://github.com/Cocofeat/EyeMoSt"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_snapshot(source_root):
    source_root=Path(source_root)
    commit=subprocess.check_output(["git","-C",str(source_root),"rev-parse","HEAD"],text=True).strip()
    if commit!=EXPECTED_COMMIT:
        raise ValueError("EyeMoSt+ commit changed")
    files={}
    for rel,digest in EXPECTED_FILES.items():
        actual=sha256(source_root/rel)
        if actual!=digest:
            raise ValueError(f"EyeMoSt+ source hash changed: {rel}")
        files[rel]=actual
    return dict(url=AUTHOR_URL,commit=commit,files=files)


def line_number(path,needle,start=None,end=None):
    lines=Path(path).read_text().splitlines()
    lo=1 if start is None else start;hi=len(lines) if end is None else end
    hits=[i for i,line in enumerate(lines,1) if lo<=i<=hi and needle in line]
    if len(hits)!=1:
        raise ValueError(f"Expected one source line for {needle!r} in {lo}:{hi}, got {hits}")
    return hits[0]


def _class_and_methods(path,class_name):
    tree=ast.parse(Path(path).read_text())
    cls=next((n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name),None)
    if cls is None: raise ValueError(f"Class not found: {class_name}")
    methods={n.name:n for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    return cls,methods


def source_line_contract(source_root):
    root=Path(source_root);model=root/"MedIA’24/model.py";train=root/"MedIA’24/train3_trans.py"
    cls,methods=_class_and_methods(model,"EyeMost_Plus")
    init=methods["__init__"];combine=methods["ST_Combin"];infer=methods["infer"];forward=methods["forward"]
    scoped=lambda node,needle: line_number(model,needle,node.lineno,node.end_lineno)
    return {
        "model":{
            "class_eye_most_plus":cls.lineno,
            "cfp_2d_backbone":scoped(init,"self.res2net_2DNet = Medical_base_2DNet"),
            "oct_3d_backbone":scoped(init,"self.resnet_3DNet = Medical_base_3DNet"),
            "combine_definition":combine.lineno,
            "combine_two_modalities":scoped(combine,"for m in range(2):"),
            "infer_definition":infer.lineno,
            "infer_backbone_output":scoped(infer,"backbone_output = self.Classifiers[m_num](input[m_num])"),
            "infer_returns_distribution_params":scoped(infer,"return gamma, v, alpha, beta"),
            "forward_signature":forward.lineno,
            "forward_infer_before_labels":scoped(forward,"gamma, v, alpha, beta = self.infer(X)"),
            "forward_one_hot_label":scoped(forward,"one_hot_y = torch.zeros(y.size(0), self.classes).cuda().scatter_"),
            "forward_combine":scoped(forward,"loc_a, scale_2_a, df_a = self.ST_Combin(gamma,v,alpha,beta)"),
            "forward_cml_label":scoped(forward,"cml_loss = calculate_cml_loss(gamma, v, alpha, beta, y, loc_a, scale_2_a, df_a,modality_num)"),
            "student_t_output":scoped(forward,"dist = torch.distributions.studentT.StudentT"),
            "test_return":scoped(forward,"return dist, loc_u, loss, av_epis, gamma, v, alpha, beta"),
        },
        "train":{
            "test_model_call":line_number(train,"evidences, evidence_a, _, u_a, gamma, v, alpha, beta = model(data, target, epoch)"),
            "softmax_score":line_number(train,"probability_fu_m = F.softmax(evidence_a)"),
            "argmax_score":line_number(train,"correct_pred, predicted = torch.max(evidence_a.data, 1)"),
            "probability_score":line_number(train,"probability = torch.softmax(evidence_a, dim=1)"),
            "nll_brier_call":line_number(train,"nll, brier = calc_nll_brier(probability, evidence_a, target, one_hot_label)"),
        }
    }


def _author_forward_function(model_path):
    tree=ast.parse(Path(model_path).read_text())
    method=None
    for node in tree.body:
        if isinstance(node,ast.ClassDef) and node.name=="EyeMost_Plus":
            for child in node.body:
                if isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef)) and child.name=="forward":
                    method=copy.deepcopy(child);break
    if method is None:
        raise ValueError("EyeMost_Plus.forward not found")
    method.name="_author_forward"
    method.decorator_list=[]
    mod=ast.Module(body=[method],type_ignores=[]);ast.fix_missing_locations(mod)
    return compile(mod,str(model_path),"exec")


def synthetic_label_independence(source_root):
    import torch
    code=_author_forward_function(Path(source_root)/"MedIA’24/model.py")
    def ev_loss(step,one_hot,*args,**kwargs):
        return one_hot[:,0].sum()*0.13
    def st_loss(step,one_hot,*args,**kwargs):
        return one_hot[:,1].sum()*0.17
    def cml_loss(gamma,v,alpha,beta,y,*args,**kwargs):
        return y.float().sum()*0.19
    ns=dict(torch=torch,
        calculate_evidential_loss_constraints=ev_loss,
        calculate_evidential_st_loss_constraints=st_loss,
        calculate_cml_loss=cml_loss)
    exec(code,ns);forward=ns["_author_forward"]
    class Stub:
        classes=2;lambda_epochs=5;mode="test";ev_st_u_min=0.05
        def infer(self,X):
            base=X[0].to(torch.float64)
            gamma={0:base+0.1,1:X[1].to(torch.float64)+0.2}
            v={0:torch.ones_like(base)*2,1:torch.ones_like(base)*3}
            alpha={0:torch.ones_like(base)*2.5,1:torch.ones_like(base)*3.5}
            beta={0:torch.ones_like(base)*1.2,1:torch.ones_like(base)*1.4}
            return gamma,v,alpha,beta
        def ST_Combin(self,gamma,v,alpha,beta):
            loc=(gamma[0]+gamma[1])/2
            scale=torch.ones_like(loc)*1.3
            df=torch.ones_like(loc)*6
            return loc,scale,df
    stub=Stub();X=[torch.tensor([[.2,.8],[.7,.3]]),torch.tensor([[.4,.6],[.1,.9]])]
    y0=torch.tensor([0,0]);y1=torch.tensor([1,1])
    original_cuda=torch.Tensor.cuda
    try:
        torch.Tensor.cuda=lambda self,*args,**kwargs:self
        a=forward(stub,X,y0,3);b=forward(stub,X,y1,3)
    finally:
        torch.Tensor.cuda=original_cuda
    # returned: dist, loc_u, loss, av_epis, gamma, v, alpha, beta
    unchanged=(torch.equal(a[1],b[1]) and torch.equal(a[0].loc,b[0].loc)
        and torch.equal(a[0].scale,b[0].scale) and torch.equal(a[0].df,b[0].df)
        and all(torch.equal(a[i][k],b[i][k]) for i in (4,5,6,7) for k in a[i]))
    if not unchanged:
        raise ValueError("Author forward prediction objects changed under label permutation")
    if float(a[2])==float(b[2]):
        raise ValueError("Synthetic loss did not respond to label permutation")
    # Label-free route uses the exact same infer/combine outputs; no dummy y.
    gamma,v,alpha,beta=stub.infer(X)
    loc,scale,df=stub.ST_Combin(gamma,v,alpha,beta)
    loc_u=loc+stub.ev_st_u_min*torch.ones(loc.shape).to(loc.device)
    dist=torch.distributions.studentT.StudentT(df=df,loc=loc_u,scale=scale)
    if not (torch.equal(dist.loc,a[0].loc) and torch.equal(loc_u,a[1])):
        raise ValueError("Label-free infer/combine route does not match forward prediction object")
    return dict(
        prediction_objects_unchanged_under_label_permutation=True,
        loss_changed_under_label_permutation=True,
        forward_requires_label_argument=True,
        label_free_route_matches_forward_prediction=True,
        label_free_route=["infer(X)","ST_Combin(gamma,v,alpha,beta)","loc_u=loc+ev_st_u_min","StudentT(df,loc_u,scale)"],
        labels_used_for_loss_not_infer=True)


def readiness_contract(source_root,native_import_result=None):
    identity=verify_snapshot(source_root);lines=source_line_contract(source_root)
    synthetic=synthetic_label_independence(source_root)
    return dict(
        schema="look_eyemost_plus_readiness_v1",method="EyeMoSt+",author=identity,
        source_lines=lines,synthetic_cpu=synthetic,native_import=native_import_result,
        current_vs_author=dict(
            author_inputs="CFP 2D backbone + OCT 3D backbone",
            current_look_inputs="CFP 2D 224x224 + OCT 2D middle slice 224x224",
            missing_modality="Author EyeMoSt+ combine path is coded for two modalities; train condition is normal/noise, not LOOK missing-modality semantics.",
            output_semantics="Author test code treats returned loc_u/evidence_a as class scores: argmax(loc_u), softmax(loc_u), and calc_nll_brier(probability, loc_u,...). loc_u is a Student-t location parameter, not an ordinary classifier logit by definition.",
        ),
        hook_contract=dict(
            safest_candidate="Per-modality backbone_output before gamma/v/alpha/beta transforms.",
            reason="A LOOK writeback there can feed all four original evidential heads while preserving Student-t fusion and confidence-loss machinery.",
            existing_hooks_complete=False,
            gap="Current LOOK standard 2D host hooks do not expose EyeMoSt+ 3D OCT backbone or its four evidential parameter heads. A dedicated MHD adapter is required.",
            rejected_shortcut="Do not correct loc_u alone and ignore scale/df; do not treat gamma/v/alpha/beta as unconstrained logits; do not delete author evidential/confidence losses."
        ),
        score_adapter=dict(
            author_convention="Use fused loc_u as classification score, then softmax/argmax exactly as train3_trans.py.",
            look_metric_compatibility="Current binary logit_metrics is mathematically compatible with softmax(loc_u) if the next contract explicitly names loc_u as author classification score. Preserve scale/df and uncertainty outputs separately.",
            decision_required=False,
            reason="This mapping is taken from author test code, not invented by LOOK."
        ),
        decision_required=[
            "Choose and lock a faithful 2D OCT encoder replacement/output dimension for the author's 3D OCT branch, including initialization.",
            "Choose and lock how LOOK missing-OCT/missing-CFP conditions are represented for an author combine path coded for two modalities; author normal/noise condition is not equivalent.",
            "Lock full A training/optimizer/stopping budget for the adapted EyeMoSt+ host before A vs A+LOOK; this package does not inherit a new training rule automatically.",
            "Decide whether adapted A must preserve both per-modality Student-t heads and CML/evidential losses exactly after the 3D->2D encoder change."
        ],
        reusable_without_new_scientific_choice=[
            "Existing train/dev cohort audit and sealed test rule.",
            "Participant ordering and missing-scenario naming once the missing-input adapter is explicitly defined.",
            "Macro-F1/AUROC/NLL/Brier reporting and participant-level paired evaluation.",
            "LOOK positive-forward-tree machinery after a dedicated adapter exposes legal feature nodes."
        ],
        readiness="interface_contract_evidence_ready_but_training_not_authorized",
        training_enabled=False,test_access=False)


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument("--source-root",required=True);ap.add_argument("--output",required=True)
    ap.add_argument("--native-import-result")
    args=ap.parse_args()
    native=json.loads(args.native_import_result) if args.native_import_result else None
    Path(args.output).write_text(json.dumps(readiness_contract(args.source_root,native),indent=2)+"\n")


if __name__=="__main__":
    main()
