"""Closed-form affine residual maps; no neural optimizer or host dependency.

Row convention: delta = (x - input_mean) @ left @ right + output_mean.
Statistics describe standardized spatial features, not PCA coordinates.
"""
from dataclasses import dataclass
import math
import torch


@dataclass
class ResidualMoments:
    count: int
    mean_x: torch.Tensor
    mean_y: torch.Tensor
    cxx: torch.Tensor
    cxy: torch.Tensor
    syy: torch.Tensor

    @classmethod
    def empty(cls, dimension):
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ValueError('Positive feature dimension required')
        z = torch.zeros(dimension, dtype=torch.float64)
        return cls(0, z.clone(), z.clone(), torch.zeros(dimension, dimension, dtype=torch.float64),
                   torch.zeros(dimension, dimension, dtype=torch.float64), torch.zeros((), dtype=torch.float64))

    def update(self, missing, residual):
        x = missing.detach().to(device='cpu', dtype=torch.float64)
        y = residual.detach().to(device='cpu', dtype=torch.float64)
        if x.ndim != 2 or y.shape != x.shape or x.shape[1] != len(self.mean_x) or len(x) == 0:
            raise ValueError('Nonempty matching two-dimensional feature batches required')
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError('Nonfinite paired features')
        n = len(x); mx = x.mean(0); my = y.mean(0)
        xc = x - mx; yc = y - my
        dx = mx - self.mean_x; dy = my - self.mean_y
        total = self.count + n; scale = self.count * n / total
        # Chan merging avoids subtracting large raw second moments.
        self.cxx += xc.T @ xc + scale * torch.outer(dx, dx)
        self.cxy += xc.T @ yc + scale * torch.outer(dx, dy)
        self.syy += yc.square().sum() + scale * dy.square().sum()
        self.mean_x += dx * n / total; self.mean_y += dy * n / total
        self.count = total


@dataclass
class AffineResidual:
    input_mean: torch.Tensor
    output_mean: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    method: str
    ridge_lambda: float
    rank_budget: int
    samples: int
    diagnostics: dict

    def predict(self, x):
        if x.ndim != 2 or x.shape[1] != len(self.input_mean):
            raise ValueError('Residual feature dimension changed')
        def on(t): return t.to(device=x.device, dtype=x.dtype)
        return ((x - on(self.input_mean)) @ on(self.left)) @ on(self.right) + on(self.output_mean)


def solve(moments, rank, ridge_lambda, *, basis=None, intercept_basis=None):
    """Minimize SSE + lambda * ||A||_F^2, with an unpenalized intercept.

    basis=None: rank(A)<=rank; an exact regularized reduced-rank solution.
    basis=[rank,D]: A=Q.T W Q and the intercept lies in Q's row space.
    All arms use the SAME standardized-coordinate objective and penalty.
    intercept_basis constrains only the RRR intercept for the matched-mean control.
    """
    if basis is not None and intercept_basis is not None:
        raise ValueError('PCA arm already constrains its intercept')
    s = moments; d = len(s.mean_x)
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= min(d, s.count - 1):
        raise ValueError('Rank budget infeasible; never silently truncate')
    if not math.isfinite(ridge_lambda) or ridge_lambda <= 0:
        raise ValueError('Strictly positive finite ridge penalty required')
    if not all(torch.isfinite(t).all() for t in (s.mean_x,s.mean_y,s.cxx,s.cxy,s.syy)):
        raise ValueError('Nonfinite sufficient statistics')
    spectrum = None
    if basis is not None:
        q = basis.detach().to(device='cpu', dtype=torch.float64)
        if q.shape != (rank,d) or not torch.isfinite(q).all():
            raise ValueError('Shared basis shape or values invalid')
        if not torch.allclose(q @ q.T, torch.eye(rank,dtype=q.dtype), rtol=2e-5,atol=2e-5):
            raise ValueError('Shared PCA rows must be orthonormal')
        w = torch.linalg.solve(q @ s.cxx @ q.T + ridge_lambda * torch.eye(rank,dtype=q.dtype),
                               q @ s.cxy @ q.T)
        left = q.T @ w; right = q; output_mean = (s.mean_y @ q.T) @ q
        method = 'shared_pca_ridge'
    else:
        # G = L L.T; ||L.T A - L^-1 Cxy||_F^2 has a truncated-SVD minimizer.
        g = (s.cxx + s.cxx.T) * .5 + ridge_lambda * torch.eye(d,dtype=s.cxx.dtype)
        chol = torch.linalg.cholesky(g)
        whitened = torch.linalg.solve_triangular(chol,s.cxy,upper=False)
        u, values, vh = torch.linalg.svd(whitened,full_matrices=False)
        from look.methods.rank_budget import retained_curve
        gains=values.square()
        spectrum=retained_curve(gains[:rank].tolist(),float(gains.sum()),quantity='regularized_centered_residual_objective_gain')
        left = torch.linalg.solve_triangular(chol.T,u[:,:rank]*values[:rank],upper=True)
        right = vh[:rank]; output_mean = s.mean_y.clone(); method = 'residual_rrr'
        if intercept_basis is not None:
            q = intercept_basis.detach().to(device='cpu', dtype=torch.float64)
            if q.shape != (rank,d) or not torch.isfinite(q).all() or not torch.allclose(
                q @ q.T,torch.eye(rank,dtype=q.dtype),rtol=2e-5,atol=2e-5):
                raise ValueError('Invalid matched intercept basis')
            output_mean = (s.mean_y @ q.T) @ q
            method = 'rrr_shared_intercept'
    cross = torch.sum(left * (s.cxy @ right.T))
    fitted = torch.trace((left.T @ s.cxx @ left) @ (right @ right.T))
    mean_error = s.count * (s.mean_y-output_mean).square().sum()
    sse = (s.syy - 2*cross + fitted + mean_error).clamp_min(0)
    norm2 = torch.trace((left.T @ left) @ (right @ right.T)).clamp_min(0)
    if not all(torch.isfinite(t).all() for t in (left,right,output_mean,sse,norm2)):
        raise ValueError('Nonfinite affine fit')
    return AffineResidual(s.mean_x.clone(),output_mean,left,right,method,float(ridge_lambda),rank,s.count,
        dict(train_standardized_mse=float(sse)/(s.count*d),sse=float(sse),
             regularized_objective=float(sse+ridge_lambda*norm2),matrix_frobenius=float(norm2.sqrt()),
             stored_factor_scalars=left.numel()+right.numel(),intercept_scalars=d,
             retained_quantity=spectrum,rank_budget_kind='slope_matrix_rank_upper_bound',
             note='Rank budget matches; free subspace and intercept capacity do not.'))


def estimated_workspace_bytes(dimension, rank, batch=32):
    """Conservative dense FP64 envelope; measured admission is still required."""
    if min(dimension,rank,batch)<1: raise ValueError('Invalid resource dimensions')
    return 8 * (16*dimension*dimension + 8*dimension*rank + 6*batch*dimension)
