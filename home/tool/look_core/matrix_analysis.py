from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .look import LOOKArtifact
from .reproducibility import write_json_atomic


def _plot_matrix_bank(records: Sequence[Dict[str, object]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{record['node_name']}\nd={record['latent_dim']}" for record in records]
    x = np.arange(len(records))
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    effective = [record["correction_weight"]["effective_rank"] / record["latent_dim"] for record in records]
    stable = [record["correction_weight"]["stable_rank"] / record["latent_dim"] for record in records]
    axes[0, 0].plot(x, effective, marker="o", label="effective rank / d")
    axes[0, 0].plot(x, stable, marker="s", label="stable rank / d")
    axes[0, 0].set_ylabel("Relative rank")
    axes[0, 0].set_ylim(bottom=0)
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(x, [record["correction_weight"]["spectral_norm"] for record in records], marker="o", label="W")
    axes[0, 1].plot(x, [record["applied_latent_map"]["spectral_norm"] for record in records], marker="s", label="I + alpha W")
    axes[0, 1].set_ylabel("Spectral norm")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].bar(x, [record["off_diagonal_energy_fraction"] for record in records])
    axes[1, 0].set_ylabel("Off-diagonal energy fraction")
    axes[1, 0].set_ylim(0, 1)

    axes[1, 1].bar(x, [record["nonnormality"] for record in records])
    axes[1, 1].set_ylabel("Normalized non-normality")
    for axis in axes.flat:
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Selected LOOK correction matrices")
    figure.savefig(output_dir / "matrix_summary.png", dpi=180)
    figure.savefig(output_dir / "matrix_summary.pdf")
    plt.close(figure)

    for index, record in enumerate(records, start=1):
        spectra = np.load(record["spectra_file"])
        singular = spectra["singular_values"]
        eigenvalues = spectra["eigenvalues"]
        applied_eigenvalues = spectra["applied_eigenvalues"]
        figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        normalized = singular / singular[0] if len(singular) and singular[0] else singular
        axes[0].plot(np.arange(1, len(singular) + 1), normalized, marker=".")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Singular-value index")
        axes[0].set_ylabel("Normalized singular value")
        axes[0].grid(alpha=0.25)
        axes[1].scatter(eigenvalues.real, eigenvalues.imag, s=18, alpha=0.7, label="W")
        axes[1].scatter(
            applied_eigenvalues.real,
            applied_eigenvalues.imag,
            s=18,
            alpha=0.7,
            label="I + alpha W",
        )
        axes[1].axhline(0, color="black", linewidth=0.6)
        axes[1].axvline(0, color="black", linewidth=0.6)
        axes[1].set_xlabel("Real")
        axes[1].set_ylabel("Imaginary")
        axes[1].set_aspect("equal", adjustable="datalim")
        axes[1].legend(frameon=False)
        figure.suptitle(f"{record['node_name']} | {record['missing_pattern']}")
        figure.savefig(
            output_dir / "matrix_spectra" / f"{index:02d}_{record['node_name']}.png",
            dpi=180,
        )
        plt.close(figure)


def _energy_rank(singular_values: np.ndarray, threshold: float) -> int:
    energy = np.square(singular_values)
    if not np.any(energy):
        return 0
    return int(np.searchsorted(np.cumsum(energy) / energy.sum(), threshold) + 1)


def _effective_rank(singular_values: np.ndarray) -> float:
    total = singular_values.sum()
    if total <= 0:
        return 0.0
    probabilities = singular_values / total
    probabilities = probabilities[probabilities > 0]
    return float(np.exp(-(probabilities * np.log(probabilities)).sum()))


def _matrix_scalars(matrix: np.ndarray) -> Dict[str, object]:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    spectral_norm = float(singular_values[0]) if len(singular_values) else 0.0
    frobenius_norm = float(np.linalg.norm(matrix, ord="fro"))
    tolerance = np.finfo(np.float64).eps * max(matrix.shape) * spectral_norm
    nonzero = singular_values[singular_values > tolerance]
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if len(singular_values) and singular_values[-1] > tolerance
        else None
    )
    return {
        "numerical_rank": int(len(nonzero)),
        "effective_rank": _effective_rank(singular_values),
        "stable_rank": float(frobenius_norm**2 / spectral_norm**2) if spectral_norm else 0.0,
        "energy_rank_90": _energy_rank(singular_values, 0.90),
        "energy_rank_95": _energy_rank(singular_values, 0.95),
        "energy_rank_99": _energy_rank(singular_values, 0.99),
        "spectral_norm": spectral_norm,
        "frobenius_norm": frobenius_norm,
        "nuclear_norm": float(singular_values.sum()),
        "condition_number": condition_number,
    }


def analyze_correction_matrix(
    artifact: LOOKArtifact,
    spectra_path: Path | None = None,
    top_k: int = 10,
) -> Dict[str, object]:
    """Describe W in delta-z = zW+b and the applied map I+alpha*W."""
    weight = artifact.weight.detach().cpu().double().numpy()
    bias = artifact.bias.detach().cpu().double().numpy()
    dimension = weight.shape[0]
    if weight.shape != (dimension, dimension):
        raise ValueError(f"LOOK correction weight must be square, got {weight.shape}")

    left, singular_values, right_h = np.linalg.svd(weight, full_matrices=False)
    eigenvalues, right_eigenvectors = np.linalg.eig(weight)
    left_eigenvalues, left_eigenvectors = np.linalg.eig(weight.T)
    eigen_order = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[eigen_order]
    right_eigenvectors = right_eigenvectors[:, eigen_order]
    corrected_map = np.eye(dimension) + artifact.alpha * weight
    corrected_singular_values = np.linalg.svd(corrected_map, compute_uv=False)
    corrected_eigenvalues = 1.0 + artifact.alpha * eigenvalues

    weight_frobenius = float(np.linalg.norm(weight, ord="fro"))
    diagonal_energy = float(np.square(np.diag(weight)).sum())
    total_energy = float(np.square(weight).sum())
    commutator = weight.T @ weight - weight @ weight.T
    top = min(top_k, dimension)
    record: Dict[str, object] = {
        "node_name": artifact.node_name,
        "missing_pattern": artifact.missing_pattern,
        "filling_strategy": artifact.filling_strategy,
        "factor": artifact.factor,
        "latent_dim": artifact.latent_dim,
        "alpha": artifact.alpha,
        "ridge_lambda": artifact.ridge_lambda,
        "train_r2": artifact.train_r2,
        "train_mse": artifact.train_mse,
        "matrix_semantics": "delta_z = z_missing @ W + b; z_corrected = z_missing + alpha * delta_z",
        "correction_weight": _matrix_scalars(weight),
        "applied_latent_map": _matrix_scalars(corrected_map),
        "spectral_radius": float(np.abs(eigenvalues).max()) if dimension else 0.0,
        "applied_spectral_radius": float(np.abs(corrected_eigenvalues).max()) if dimension else 0.0,
        "bias_l2": float(np.linalg.norm(bias)),
        "applied_bias_l2": float(abs(artifact.alpha) * np.linalg.norm(bias)),
        "diagonal_energy_fraction": diagonal_energy / total_energy if total_energy else 0.0,
        "off_diagonal_energy_fraction": 1.0 - diagonal_energy / total_energy if total_energy else 0.0,
        "symmetry_error": (
            float(np.linalg.norm(weight - weight.T, ord="fro") / weight_frobenius)
            if weight_frobenius
            else 0.0
        ),
        "nonnormality": (
            float(np.linalg.norm(commutator, ord="fro") / weight_frobenius**2)
            if weight_frobenius
            else 0.0
        ),
        "positive_real_eigenvalues": int(np.sum(eigenvalues.real > 0)),
        "negative_real_eigenvalues": int(np.sum(eigenvalues.real < 0)),
        "complex_eigenvalues": int(np.sum(np.abs(eigenvalues.imag) > 1e-10)),
        "top_singular_values": singular_values[:top].tolist(),
        "top_eigenvalues": [
            {"real": float(value.real), "imag": float(value.imag), "magnitude": float(abs(value))}
            for value in eigenvalues[:top]
        ],
    }
    if spectra_path is not None:
        spectra_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            spectra_path,
            weight=weight,
            bias=bias,
            singular_values=singular_values,
            left_singular_vectors=left,
            right_singular_vectors=right_h.T,
            eigenvalues=eigenvalues,
            right_eigenvectors=right_eigenvectors,
            left_eigenvalues=left_eigenvalues,
            left_eigenvectors=left_eigenvectors,
            applied_singular_values=corrected_singular_values,
            applied_eigenvalues=corrected_eigenvalues,
        )
        record["spectra_file"] = str(spectra_path)
    return record


def _flat_row(record: Dict[str, object]) -> Dict[str, object]:
    row = {
        key: value
        for key, value in record.items()
        if not isinstance(value, (dict, list))
    }
    for prefix in ("correction_weight", "applied_latent_map"):
        for key, value in record[prefix].items():
            row[f"{prefix}_{key}"] = value
    return row


def analyze_look_bank(
    artifacts: Sequence[LOOKArtifact], output_dir: Path
) -> List[Dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, artifact in enumerate(artifacts, start=1):
        spectra_path = output_dir / "matrix_spectra" / f"{index:02d}_{artifact.node_name}.npz"
        records.append(analyze_correction_matrix(artifact, spectra_path=spectra_path))
    write_json_atomic(
        {
            "interpretation_note": (
                "Eigenvectors describe latent PCA-coordinate directions. For the row-vector map zW, "
                "left eigenvectors are input directions. Because W may be non-normal, singular vectors "
                "and singular values are the primary stable amplification descriptors."
            ),
            "matrices": records,
        },
        output_dir / "matrix_analysis.json",
    )
    rows = [_flat_row(record) for record in records]
    if rows:
        with (output_dir / "matrix_analysis.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        _plot_matrix_bank(records, output_dir)
    return records


def aggregate_matrix_analyses(output_root: Path, output_csv: Path) -> int:
    rows = []
    pattern = "**/experiments/*/look/*/matrix_analysis.json"
    for analysis_path in sorted(output_root.glob(pattern)):
        experiment_dir = analysis_path.parents[2]
        result_path = experiment_dir / "experiment_result.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        for record in analysis["matrices"]:
            row = {
                "experiment_id": result["experiment_id"],
                "fusion_position": result["selection"]["fusion_position"],
                "seed": result["selection"]["seed"],
                "filling_strategy": result["selection"]["filling_strategy"],
                **_flat_row(record),
            }
            rows.append(row)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    else:
        output_csv.write_text(
            "experiment_id,fusion_position,seed,filling_strategy,node_name,missing_pattern\n",
            encoding="utf-8",
        )
    return len(rows)
