import ast
from pathlib import Path

import pytest

from look.analysis import eyemost_readiness as e


def minimal_source(tmp_path):
    root=tmp_path
    folder=root/'MedIA’24';folder.mkdir()
    model=folder/'model.py'
    model.write_text("""class EyeMost_Plus:
    def forward(self, X, y, global_step):
        loss = 0
        gamma, v, alpha, beta = self.infer(X)
        one_hot_y = torch.zeros(y.size(0), self.classes).cuda().scatter_(1, y.unsqueeze(1), 1)
        annealing_coef = min(0.5, global_step / self.lambda_epochs)
        modality_num = len(X)
        for m_num in range(len(X)):
            loss += calculate_evidential_loss_constraints(global_step, one_hot_y, gamma[m_num], v[m_num], alpha[m_num], beta[m_num], lambda_coef=annealing_coef)
        loc_a, scale_2_a, df_a = self.ST_Combin(gamma,v,alpha,beta)
        loss += calculate_evidential_st_loss_constraints(global_step, one_hot_y, loc_a, scale_2_a, df_a, annealing_coef)
        cml_loss = calculate_cml_loss(gamma, v, alpha, beta, y, loc_a, scale_2_a, df_a,modality_num)
        loss += 10 * cml_loss
        df_v = df_a
        loc_u = loc_a
        loc_u_min = self.ev_st_u_min * torch.ones(loc_u.shape).to(loc_u.device)
        loc_u = loc_u + loc_u_min
        scale_sigma = scale_2_a
        dist = torch.distributions.studentT.StudentT(df=df_v, loc=loc_u, scale=scale_sigma)
        av_epis = scale_sigma * (1+2/(df_v-2))
        if self.mode == "test":
            return dist, loc_u, loss, av_epis, gamma, v, alpha, beta
        else:
            return dist, loc_u, loss, av_epis
""")
    return root


def test_extracted_author_forward_label_permutation_changes_loss_not_prediction(tmp_path):
    root=minimal_source(tmp_path)
    result=e.synthetic_label_independence(root)
    assert result['prediction_objects_unchanged_under_label_permutation'] is True
    assert result['loss_changed_under_label_permutation'] is True
    assert result['label_free_route_matches_forward_prediction'] is True


def test_author_forward_ast_requires_label_but_infer_precedes_label_use(tmp_path):
    root=minimal_source(tmp_path);tree=ast.parse((root/'MedIA’24/model.py').read_text())
    forward=next(c for n in tree.body if isinstance(n,ast.ClassDef) for c in n.body if isinstance(c,ast.FunctionDef))
    assert [a.arg for a in forward.args.args]==['self','X','y','global_step']
    calls=[node for node in ast.walk(forward) if isinstance(node,ast.Call)]
    names=[ast.unparse(node.func) for node in calls]
    assert 'self.infer' in names
