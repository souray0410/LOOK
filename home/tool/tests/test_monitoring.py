import pandas as pd

from look_core.monitoring import TrainingMonitor


def _record(epoch: int):
    return {
        "event": "epoch",
        "epoch": epoch,
        "train_loss": 0.8 / epoch,
        "train_batch_accuracy": 0.7,
        "validation": {
            "cross_entropy": 0.9 / epoch,
            "accuracy": 0.6,
            "balanced_accuracy": 0.5,
            "macro_f1": 0.4,
            "weighted_f1": 0.55,
            "macro_auroc_ovr": 0.75,
            "ece_15": 0.1,
            "multiclass_brier": 0.2,
            "cohen_kappa": 0.3,
            "f1_per_class": [0.8, 0.3],
            "precision_per_class": [0.8, 0.3],
            "sensitivity_per_class": [0.8, 0.3],
            "specificity_per_class": [0.8, 0.9],
            "confusion_matrix": [[8, 2], [3, 7]],
        },
        "graph_monitor": {"node/loss_mean": 0.2},
    }


def test_monitor_materializes_validation_outputs_after_each_epoch(tmp_path, capsys):
    monitor = TrainingMonitor(tmp_path)
    monitor.append(_record(1))

    assert (tmp_path / "training_curves.csv").is_file()
    assert (tmp_path / "training_curves.png").is_file()
    assert (tmp_path / "training_curves.pdf").is_file()
    assert (tmp_path / "validation_per_class.png").is_file()
    assert (tmp_path / "validation_per_class.pdf").is_file()
    assert (tmp_path / "validation_confusion_matrix.png").is_file()
    assert (tmp_path / "validation_confusion_matrix.pdf").is_file()
    assert not (tmp_path / "monitor_complete.json").exists()

    frame = pd.read_csv(tmp_path / "training_curves.csv")
    assert frame.loc[0, "validation_macro_f1"] == 0.4
    assert frame.loc[0, "validation_sensitivity_per_class_glaucoma"] == 0.3
    assert "generalization_loss_gap" in frame
    assert "val_macro_f1=0.4000" in capsys.readouterr().out

    monitor.append(_record(2))
    assert len(pd.read_csv(tmp_path / "training_curves.csv")) == 2
    monitor.finalize()
    assert (tmp_path / "monitor_complete.json").is_file()
