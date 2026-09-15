"""Exact compact spectra of a square low-rank residual map, in stated coordinates.

This diagnostic never refits, selects a rank or predicts a clinical outcome. It
does not interpret fitted directions as physical/causal factors. It avoids a
dense D-by-D allocation when the stored factor rank is much smaller than D.
"""
import numpy as np


def analyze(left, right, input_mean, output_mean, *, coordinate_system):
    if coordinate_system != 'standardized_reduced_features':
        raise ValueError('Explicit standardized reduced-feature coordinates required')
    u, v, mx, c = [np.array(x, dtype=np.float64, copy=True) for x in (left,right,input_mean,output_mean)]
    if u.ndim != 2 or v.ndim != 2 or not 1 <= u.shape[1] <= u.shape[0]:
        raise ValueError('Expected D-by-q, q<=D factors')
    d, q = u.shape
    if v.shape != (q,d) or mx.shape != (d,) or c.shape != (d,):
        raise ValueError('Inconsistent affine factor shapes')
    if not all(np.isfinite(x).all() for x in (u,v,mx,c)):
        raise ValueError('Nonfinite affine factor')
    # A = Qu (Ru Rv.T) Qv.T; all omitted singular values are zero.
    _, ru = np.linalg.qr(u, mode='reduced')
    _, rv = np.linalg.qr(v.T, mode='reduced')
    singular = np.linalg.svd(ru @ rv.T, compute_uv=False)
    # span(U,V.T) contains both the column and row spaces. On its orthogonal
    # complement A is zero, so I+A is exactly identity. Rank-deficient input
    # columns may yield a larger QR span; the extra identity directions are safe.
    basis, _ = np.linalg.qr(np.concatenate((u,v.T),axis=1),mode='reduced')
    p = basis.shape[1]
    applied = np.linalg.svd(np.eye(p)+(basis.T@u)@(v@basis),compute_uv=False)
    applied = np.sort(np.concatenate((applied,np.ones(d-p))))[::-1]
    eigen = np.linalg.eigvals(v@u)
    eigen = np.concatenate((eigen,np.zeros(d-q)))
    bias = c-(mx@u)@v
    norm = float(singular[0]); energy = float(np.square(singular).sum())
    return dict(coordinate_system=coordinate_system,feature_dimension=d,rank_budget=q,
        correction_singular_values=singular.tolist(),omitted_correction_zero_count=d-q,
        correction_eigenvalues=[dict(real=float(z.real),imag=float(z.imag)) for z in eigen],
        applied_singular_values=applied.tolist(),applied_identity_complement_dimension=d-p,
        correction_spectral_norm=norm,correction_frobenius_norm=float(np.sqrt(energy)),
        stable_rank=energy/(norm*norm) if norm else None,
        singular_energy_fraction=None if not energy else (np.cumsum(singular**2)/energy).tolist(),
        energy_fraction_defined=bool(energy),
        mean_correction_l2=float(np.linalg.norm(c)),affine_bias_l2=float(np.linalg.norm(bias)),
        applied_spectral_norm=float(applied[0]),applied_minimum_singular_value=float(applied[-1]),
        note='Spectra describe matrices, not feature-distribution energy or classification benefit; bias is separate.')


def compare(first, second):
    """Compare slope actions without requiring equal stored factorizations."""
    def unpack(mapping):
        return [np.asarray(mapping[k],dtype=np.float64) for k in ('left','right','input_mean','output_mean')]
    u,v,mx,c=unpack(first);a,b,my,e=unpack(second)
    if u.shape[0]!=a.shape[0] or not np.array_equal(mx,my):
        raise ValueError('Different coordinates/input centers; do not assert matched slopes')
    # Difference factors support different but algebraically equivalent gauges.
    f=np.concatenate((u,-a),axis=1);g=np.concatenate((v,b),axis=0)
    _,rf=np.linalg.qr(f,mode='reduced');_,rg=np.linalg.qr(g.T,mode='reduced')
    return dict(slope_difference_frobenius=float(np.linalg.norm(rf@rg.T)),
                mean_difference_l2=float(np.linalg.norm(c-e)))
