"""Display names only; immutable method identifiers retain their original meaning."""
METHOD_LABELS = {
    'shared_pca_ridge': 'PCA-constrained residual ridge',
    'rrr_shared_intercept': 'Low-rank residual regression (PCA-constrained mean)',
    'residual_rrr': 'Low-rank residual regression (free mean)',
}
METHOD_EXPLANATION = (
    'All three maps predict complete-minus-missing residuals and add them to the original feature. '
    'PCA constrains both the slope subspace and centered-input mean correction. '
    'The two RRR arms share the same fitted slope at matched data/rank/lambda; '
    'only the mean correction is projected onto PCA or left free. '
    'Neither RRR arm directly predicts the complete feature. '
    'Fixed maps are affine and fitted by linear algebra, without neural-network optimization.'
)


def method_label(method):
    return METHOD_LABELS.get(method, method)
