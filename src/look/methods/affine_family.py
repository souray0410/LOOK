"""Train-only affine correction family in the same standardized feature space.

PLS-SVD here means cross-covariance directions followed by ridge regression;
it is deliberately not labelled iterative PLSRegression.
"""
from dataclasses import asdict, dataclass
import math
import torch
from look.methods.linear_vector import AffineResidual, solve
from look.methods.linear_operator import LinearVectorArtifact
from look.methods.operator import LOOKArtifact

VERSION = 'look_affine_family_v1'
REFERENCE_ARMS = ('shared_pca_ridge', 'rrr_shared_intercept', 'residual_rrr')
NEW_ARMS = ('pca_free_mean', 'residual_ridge', 'pls_svd_ridge', 'diagonal_ridge', 'orthogonal_alignment')
ARMS = REFERENCE_ARMS + NEW_ARMS


@dataclass
class FamilyMap(AffineResidual):
    def predict(self, x):
        if x.ndim != 2 or x.shape[1] != len(self.input_mean):
            raise ValueError('Affine input shape changed')
        def on(t): return t.to(device=x.device, dtype=x.dtype)
        z = x - on(self.input_mean)
        if self.left.ndim == 1:
            y = z * on(self.left)
        elif self.right.numel() == 0:
            y = z @ on(self.left)
        else:
            y = (z @ on(self.left)) @ on(self.right)
        return y + on(self.output_mean)


class FamilyArtifact(LinearVectorArtifact):
    def record(self):
        return dict(schema=VERSION, base=asdict(self.base), mapping=asdict(self.mapping))

    @classmethod
    def from_record(cls, record):
        if record['schema'] != VERSION:
            raise ValueError('Unknown family artifact schema')
        return cls(LOOKArtifact(**record['base']), FamilyMap(**record['mapping']))


def fit_map(s, rank, ridge_lambda, *, arm, basis):
    d = len(s.mean_x)
    if arm not in ARMS or s.count < 2:
        raise ValueError('Unknown arm or insufficient fitting participants')
    if not math.isfinite(ridge_lambda) or ridge_lambda <= 0:
        raise ValueError('Positive finite regularization required')
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= min(d, s.count-1):
        raise ValueError('Infeasible reference rank; no silent truncation')
    if not all(torch.isfinite(t).all() for t in (s.mean_x, s.mean_y, s.cxx, s.cxy, s.syy)):
        raise ValueError('Nonfinite statistics')
    if arm in REFERENCE_ARMS or arm == 'pca_free_mean':
        kw = {'basis': basis} if arm in ('shared_pca_ridge', 'pca_free_mean') else (
            {'intercept_basis': basis} if arm == 'rrr_shared_intercept' else {})
        m = solve(s, rank, ridge_lambda, **kw)
        m = FamilyMap(**asdict(m))
        if arm == 'pca_free_mean':
            m.output_mean = s.mean_y.clone()
        m.method = arm
        w = m.left @ m.right
    else:
        eye = torch.eye(d, dtype=torch.float64)
        g = (s.cxx + s.cxx.T) * .5
        right = torch.empty(0, dtype=torch.float64)
        if arm == 'residual_ridge':
            left = torch.linalg.solve(g + ridge_lambda * eye, s.cxy)
            w = left
        elif arm == 'diagonal_ridge':
            left = s.cxy.diag() / (g.diag() + ridge_lambda)
            w = torch.diag(left)
        elif arm == 'orthogonal_alignment':
            # Y = X + D; Xc.T Yc = Cxx + CxD. Q can include reflections.
            u, _, vh = torch.linalg.svd(g + s.cxy, full_matrices=False)
            left = u @ vh - eye
            w = left
        else:
            u, _, _ = torch.linalg.svd(s.cxy, full_matrices=False)
            left = u[:, :rank]
            right = torch.linalg.solve(left.T @ g @ left + ridge_lambda * torch.eye(rank, dtype=g.dtype),
                                       left.T @ s.cxy)
            w = left @ right
        m = FamilyMap(s.mean_x.clone(), s.mean_y.clone(), left, right, arm,
                      float(ridge_lambda), rank, s.count, {})
    sse = (s.syy - 2 * (w * s.cxy).sum() + (w * (s.cxx @ w)).sum()
           + s.count * (s.mean_y - m.output_mean).square().sum()).clamp_min(0)
    norm = w.square().sum()
    if not torch.isfinite(sse) or not torch.isfinite(norm) or not torch.isfinite(w).all():
        raise ValueError('Nonfinite family fit')
    m.diagnostics.update(train_standardized_mse=float(sse)/(s.count*d),
        sse=float(sse), regularized_objective=float(sse + ridge_lambda*norm) if arm != 'orthogonal_alignment' else None,
        matrix_frobenius=float(norm.sqrt()), feature_dimension=d, reference_rank=rank,
        stored_factor_scalars=m.left.numel()+m.right.numel(), intercept_scalars=d,
        rank_budget_kind='slope_rank_upper_bound' if arm in (*REFERENCE_ARMS, 'pca_free_mean', 'pls_svd_ridge') else 'not_rank_matched',
        ridge_applied=arm != 'orthogonal_alignment',
        coordinate_system='standardized_reduced_features_not_raw_feature_metric',
        objective='orthogonal_complete_feature_alignment' if arm == 'orthogonal_alignment' else 'residual_ridge_with_declared_constraint',
        note='Same reference setting is a conditional mechanism control, not independently optimized best performance.')
    return m
