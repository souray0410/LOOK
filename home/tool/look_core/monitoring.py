from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .state import atomic_write_json


DEFAULT_CLASS_NAMES = ("normal", "glaucoma")
PER_CLASS_METRICS = (
    "f1_per_class",
    "precision_per_class",
    "sensitivity_per_class",
    "specificity_per_class",
)


class TrainingMonitor:
    """Rank-zero live JSONL/CSV/figure monitor for notebook and CLI runs."""

    def __init__(
        self, run_dir: Path, class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES
    ) -> None:
        self.run_dir = Path(run_dir)
        self.class_names = tuple(class_names)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.run_dir / "monitor_history.jsonl"
        self.started = time.monotonic()

    def append(self, record: dict[str, Any]) -> None:
        payload = {**record, "elapsed_seconds": time.monotonic() - self.started}
        with self.jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        print(self._console_summary(payload), flush=True)
        self._materialize(payload.get("event", "epoch"))

    def finalize(self, event: str = "epoch") -> None:
        epoch_count = self._materialize(event)
        if epoch_count:
            atomic_write_json(
                {"status": "complete", "event": event, "epochs": epoch_count},
                self.run_dir / "monitor_complete.json",
            )

    @staticmethod
    def _console_summary(payload: dict[str, Any]) -> str:
        validation = payload.get("validation", {})
        values = [
            ("event", payload.get("event")),
            ("epoch", payload.get("epoch")),
            ("train_loss", payload.get("train_loss")),
            ("train_acc", payload.get("train_batch_accuracy")),
            ("val_loss", validation.get("cross_entropy")),
            ("val_acc", validation.get("accuracy")),
            ("val_bal_acc", validation.get("balanced_accuracy")),
            ("val_macro_f1", validation.get("macro_f1")),
            ("val_weighted_f1", validation.get("weighted_f1")),
            ("val_macro_auroc", validation.get("macro_auroc_ovr")),
            ("val_ece", validation.get("ece_15")),
            ("val_brier", validation.get("multiclass_brier")),
            ("val_kappa", validation.get("cohen_kappa")),
            ("generator_loss", payload.get("generator_loss")),
            ("discriminator_loss", payload.get("discriminator_loss")),
            ("validation_l1", payload.get("validation_l1")),
            ("elapsed_seconds", payload.get("elapsed_seconds")),
        ]
        return " | ".join(
            f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in values
            if value is not None
        )

    def _materialize(self, event: str) -> int:
        if not self.jsonl.is_file():
            return 0
        rows = [
            json.loads(line)
            for line in self.jsonl.read_text(encoding="utf-8").splitlines()
            if line
        ]
        epoch_rows = [row for row in rows if row.get("event") == event]
        if not epoch_rows:
            return 0
        frame = pd.DataFrame([self._flatten(row) for row in epoch_rows])
        self._write_csv(frame, self.run_dir / "training_curves.csv")
        if event == "gan_epoch":
            self._plot_gan(frame)
        else:
            self._plot_classifier(frame)
            self._plot_per_class(frame)
            self._plot_confusion_matrix(epoch_rows[-1].get("validation", {}))
        return len(epoch_rows)

    def _flatten(self, row: dict[str, Any]) -> dict[str, Any]:
        item = {
            key: value
            for key, value in row.items()
            if key not in {"validation", "graph_monitor"}
        }
        validation = row.get("validation", {})
        for key, value in validation.items():
            if isinstance(value, (int, float)):
                item[f"validation_{key}"] = value
            elif key in PER_CLASS_METRICS and isinstance(value, list):
                for index, metric_value in enumerate(value):
                    class_name = (
                        self.class_names[index]
                        if index < len(self.class_names)
                        else f"class_{index}"
                    )
                    item[f"validation_{key}_{class_name}"] = metric_value
        item.update(
            {f"graph_{key}": value for key, value in row.get("graph_monitor", {}).items()}
        )
        if "validation_cross_entropy" in item and "train_loss" in item:
            item["generalization_loss_gap"] = item["validation_cross_entropy"] - item["train_loss"]
        if "validation_accuracy" in item and "train_batch_accuracy" in item:
            item["generalization_accuracy_gap"] = item["train_batch_accuracy"] - item["validation_accuracy"]
        return item

    @staticmethod
    def _write_csv(frame: pd.DataFrame, path: Path) -> None:
        temporary = path.with_name(path.name + ".partial")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)

    def _save_figure(self, figure, stem: str) -> None:
        for suffix, options in (("png", {"dpi": 180}), ("pdf", {})):
            path = self.run_dir / f"{stem}.{suffix}"
            temporary = path.with_name(path.name + ".partial")
            figure.savefig(temporary, format=suffix, **options)
            temporary.replace(path)
        plt.close(figure)

    def _plot_classifier(self, frame: pd.DataFrame) -> None:
        figure, axes = plt.subplots(2, 2, figsize=(12, 8))
        epoch = frame["epoch"]
        self._plot_columns(axes[0, 0], epoch, frame, (
            ("train_loss", "Train loss"),
            ("validation_cross_entropy", "Validation loss"),
        ), "Cross-entropy", "Optimization and generalization")
        self._plot_columns(axes[0, 1], epoch, frame, (
            ("train_batch_accuracy", "Train accuracy"),
            ("validation_accuracy", "Validation accuracy"),
            ("validation_balanced_accuracy", "Validation balanced accuracy"),
        ), "Score", "Accuracy", ylim=(0, 1))
        self._plot_columns(axes[1, 0], epoch, frame, (
            ("validation_macro_f1", "Macro-F1"),
            ("validation_weighted_f1", "Weighted-F1"),
            ("validation_macro_auroc_ovr", "Macro-AUROC"),
        ), "Score", "Validation discrimination", ylim=(0, 1))
        self._plot_columns(axes[1, 1], epoch, frame, (
            ("validation_ece_15", "ECE"),
            ("validation_multiclass_brier", "Brier"),
            ("validation_cohen_kappa", "Cohen kappa"),
        ), "Value", "Calibration and agreement")
        figure.tight_layout()
        self._save_figure(figure, "training_curves")

    def _plot_per_class(self, frame: pd.DataFrame) -> None:
        figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
        metric_labels = (
            ("f1_per_class", "F1"),
            ("precision_per_class", "Precision"),
            ("sensitivity_per_class", "Sensitivity"),
            ("specificity_per_class", "Specificity"),
        )
        for axis, (metric, label) in zip(axes.flat, metric_labels):
            for class_name in self.class_names:
                column = f"validation_{metric}_{class_name}"
                if column in frame:
                    axis.plot(
                        frame["epoch"], frame[column], marker="o", markersize=3,
                        label=class_name.replace("_", " "),
                    )
            axis.set(xlabel="Epoch", ylabel=label, title=f"Validation {label}", ylim=(0, 1))
            axis.grid(alpha=0.25)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
            figure.subplots_adjust(bottom=0.15)
        figure.tight_layout(rect=(0, 0.10, 1, 1))
        self._save_figure(figure, "validation_per_class")

    def _plot_confusion_matrix(self, validation: dict[str, Any]) -> None:
        matrix = np.asarray(validation.get("confusion_matrix", []), dtype=np.float64)
        if matrix.shape != (len(self.class_names), len(self.class_names)):
            return
        denominators = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(
            matrix, denominators, out=np.zeros_like(matrix), where=denominators != 0
        )
        figure, axes = plt.subplots(1, 2, figsize=(13, 5.4))
        labels = [name.replace("_", " ") for name in self.class_names]
        for axis, values, title, format_value in (
            (axes[0], matrix, "Validation confusion matrix: counts", lambda value: f"{int(value)}"),
            (axes[1], normalized, "Validation confusion matrix: row-normalized", lambda value: f"{value:.2f}"),
        ):
            image = axis.imshow(values, cmap="Blues", vmin=0)
            axis.set(
                xlabel="Predicted class", ylabel="True class", title=title,
                xticks=range(len(labels)), yticks=range(len(labels)),
                xticklabels=labels, yticklabels=labels,
            )
            axis.tick_params(axis="x", rotation=35)
            threshold = float(values.max()) / 2 if values.size else 0
            for row in range(values.shape[0]):
                for column in range(values.shape[1]):
                    value = values[row, column]
                    axis.text(
                        column, row, format_value(value), ha="center", va="center",
                        color="white" if value > threshold else "black", fontsize=8,
                    )
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.tight_layout()
        self._save_figure(figure, "validation_confusion_matrix")

    def _plot_gan(self, frame: pd.DataFrame) -> None:
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        self._plot_columns(axes[0], frame["epoch"], frame, (
            ("generator_loss", "Generator"),
            ("discriminator_loss", "Discriminator"),
        ), "Loss", "cGAN optimization")
        self._plot_columns(axes[1], frame["epoch"], frame, (
            ("validation_l1", "Validation L1"),
        ), "L1", "Held-out reconstruction")
        figure.tight_layout()
        self._save_figure(figure, "training_curves")

    @staticmethod
    def _plot_columns(axis, epoch, frame, columns, ylabel, title, ylim=None) -> None:
        for column, label in columns:
            if column in frame:
                axis.plot(epoch, frame[column], marker="o", markersize=3, label=label)
        axis.set(xlabel="Epoch", ylabel=ylabel, title=title)
        if ylim is not None:
            axis.set_ylim(*ylim)
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend()
