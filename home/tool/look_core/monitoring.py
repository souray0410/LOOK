from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .state import atomic_write_json


class TrainingMonitor:
    """Rank-zero JSONL/CSV/figure monitor for resumable notebook and CLI runs."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.run_dir / "monitor_history.jsonl"
        self.started = time.monotonic()

    def append(self, record: dict[str, Any]) -> None:
        payload = {**record, "elapsed_seconds": time.monotonic() - self.started}
        with self.jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        print(
            " | ".join(
                f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
                for key, value in payload.items()
                if key not in {"validation", "graph_monitor", "gpu_memory_mb"}
            ),
            flush=True,
        )

    def finalize(self, event: str = "epoch") -> None:
        if not self.jsonl.is_file():
            return
        rows = [json.loads(line) for line in self.jsonl.read_text(encoding="utf-8").splitlines() if line]
        epoch_rows = [row for row in rows if row.get("event") == event]
        if not epoch_rows:
            return
        flattened = []
        for row in epoch_rows:
            item = {
                key: value
                for key, value in row.items()
                if key not in {"validation", "graph_monitor"}
            }
            item.update({f"validation_{key}": value for key, value in row.get("validation", {}).items() if isinstance(value, (int, float))})
            item.update(
                {f"graph_{key}": value for key, value in row.get("graph_monitor", {}).items()}
            )
            flattened.append(item)
        frame = pd.DataFrame(flattened)
        frame.to_csv(self.run_dir / "training_curves.csv", index=False)
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        if event == "gan_epoch":
            axes[0].plot(frame["epoch"], frame["generator_loss"], label="Generator")
            axes[0].plot(frame["epoch"], frame["discriminator_loss"], label="Discriminator")
            axes[0].set(xlabel="Epoch", ylabel="Loss", title="cGAN optimization")
            axes[1].plot(frame["epoch"], frame["validation_l1"], label="Validation L1")
            axes[1].set(xlabel="Epoch", ylabel="L1", title="Held-out reconstruction")
        else:
            axes[0].plot(frame["epoch"], frame["train_loss"], label="Train loss")
            if "validation_cross_entropy" in frame:
                axes[0].plot(frame["epoch"], frame["validation_cross_entropy"], label="Validation loss")
            axes[0].set(xlabel="Epoch", ylabel="Cross-entropy", title="Optimization")
            for column, label in (
                ("validation_macro_f1", "Macro-F1"),
                ("validation_balanced_accuracy", "Balanced accuracy"),
                ("validation_macro_auroc_ovr", "Macro-AUROC"),
            ):
                if column in frame:
                    axes[1].plot(frame["epoch"], frame[column], label=label)
            axes[1].set(xlabel="Epoch", ylabel="Score", title="Validation performance", ylim=(0, 1))
        axes[0].legend()
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(self.run_dir / "training_curves.png", dpi=180)
        figure.savefig(self.run_dir / "training_curves.pdf")
        plt.close(figure)
        atomic_write_json(
            {"status": "complete", "event": event, "epochs": len(epoch_rows)},
            self.run_dir / "monitor_complete.json",
        )
