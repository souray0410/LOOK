from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .config import ExperimentConfig, ExperimentSelection
from .data import UKBPairedEyeDataset, make_loader, validate_reference_table
from .evaluate import evaluate_missing
from .filling import NormalizedMeanFiller, PairedCGANFiller
from .gan import load_generator, make_gan_loaders, train_paired_cgan_direction
from .graph import build_resnet50_mhd_graph, graph_summary
from .look import greedy_fit_look, load_selected_bank
from .matrix_analysis import analyze_look_bank
from .metrics import (
    classification_metrics,
    holm_adjust,
    paired_participant_bootstrap,
    participant_cluster_bootstrap,
)
from .reproducibility import (
    environment_manifest,
    implementation_sha256,
    seed_everything,
    sha256,
    write_json_atomic,
)
from .state import PipelineState
from .train import predict, train_complete_model


@dataclass(frozen=True)
class PipelineOptions:
    train_if_missing: bool = True
    fit_look: bool = True
    evaluate_random_missing: bool = True
    allow_test: bool = False
    frozen_test_confirmation: str = ""
    resume: bool = True
    restart: bool = False
    check_all_image_paths: bool = False
    smoke_limit: Optional[int] = None
    bootstrap_iterations: int = 2000

    def validate(self, selection: ExperimentSelection) -> None:
        if self.resume and self.restart:
            raise ValueError("resume and restart are mutually exclusive")
        if self.allow_test and self.frozen_test_confirmation != "CONFIGURATION_FROZEN":
            raise ValueError("Set frozen_test_confirmation='CONFIGURATION_FROZEN' to access test data")
        if self.allow_test and self.smoke_limit is not None:
            raise ValueError("Smoke subsets cannot produce final test results")
        if self.bootstrap_iterations < 100:
            raise ValueError("bootstrap_iterations is too small for inferential reporting")


class ExperimentRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        selection: ExperimentSelection,
        options: PipelineOptions,
        device: torch.device,
    ) -> None:
        config.validate()
        selection.validate(config)
        options.validate(selection)
        self.config = config
        self.selection = selection
        self.options = options
        self.device = device
        self.project_root = Path(__file__).resolve().parents[2]
        self.data_hash = sha256(config.labels_csv)
        self.implementation_hash = implementation_sha256(self.project_root)
        self.experiment_id = self._experiment_id()
        self.experiment_dir = Path(config.output_root) / "experiments" / self.experiment_id
        self.prediction_dir = self.experiment_dir / "predictions"

    def _experiment_id(self) -> str:
        scientific_config = self.config.as_dict()
        scientific_config.pop("data_root", None)
        scientific_config.pop("output_root", None)
        scientific_config.pop("cache_root", None)
        identity = {
            "config": scientific_config,
            "selection": self.selection.as_dict(),
            "fit_look": self.options.fit_look,
            "evaluate_random_missing": self.options.evaluate_random_missing,
            "smoke_limit": self.options.smoke_limit,
            "bootstrap_iterations": self.options.bootstrap_iterations,
            "labels_sha256": self.data_hash,
            "implementation_sha256": self.implementation_hash,
        }
        fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:12]
        return f"{self.selection.experiment_id}__{fingerprint}"

    def run(self) -> Dict[str, object]:
        result_path = self.experiment_dir / "experiment_result.json"
        if self.options.resume and result_path.is_file():
            completed = json.loads(result_path.read_text(encoding="utf-8"))
            if completed.get("status") == "complete" and completed.get("experiment_id") == self.experiment_id:
                return completed
        state_root = Path(self.config.cache_root) / "pipeline_state"
        state_config = {
            "experiment_id": self.experiment_id,
            "config": self.config.as_dict(),
            "selection": self.selection.as_dict(),
            "options": asdict(self.options),
        }
        with PipelineState(state_root, f"experiment__{self.experiment_id}", state_config, [self.config.labels_csv]) as state:
            result = self._run_impl()
            state.complete([result_path])
            return result

    def _run_impl(self) -> Dict[str, object]:
        seed_everything(self.selection.seed)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        started = datetime.now(timezone.utc).isoformat()
        audit = validate_reference_table(
            self.config.labels_csv,
            self.config.data_root,
            check_paths=self.options.check_all_image_paths,
        )
        manifest = environment_manifest(
            self.config.as_dict(), self.config.labels_csv, self.project_root
        )
        manifest.update(
            {
                "selection": self.selection.as_dict(),
                "options": asdict(self.options),
                "experiment_id": self.experiment_id,
                "data_audit": audit,
                "backbone_id": self._backbone_id(),
                "implementation_sha256": self.implementation_hash,
                "started_at_utc": started,
            }
        )
        write_json_atomic(manifest, self.experiment_dir / "run_manifest.json")

        loaders, datasets = self._build_loaders()
        checkpoint_candidate = (
            Path(self.config.output_root) / "backbones" / self._backbone_id() / "best.pt"
        )
        graph = build_resnet50_mhd_graph(
            self.selection.fusion_position,
            self.config.num_classes,
            self.config.micro_batch_size,
            self.config.image_size,
            self.device,
            pretrained=not (checkpoint_candidate.exists() and self.options.resume),
        )
        structure = graph_summary(graph)
        write_json_atomic(structure, self.experiment_dir / "graph_summary.json")
        checkpoint_path = self._train_or_resume(graph, loaders["train"], loaders["validation"])

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if checkpoint["architecture_id"] != self.selection.architecture_id:
            raise RuntimeError("Checkpoint architecture does not match the selected experiment")
        if checkpoint.get("backbone_training") != "complete_modalities_only":
            raise RuntimeError("The classifier checkpoint was not trained on complete modalities only")
        if checkpoint.get("labels_sha256") != self.data_hash:
            raise RuntimeError("The classifier checkpoint was trained from a different label table")
        graph.load_state_dict(checkpoint["graph_state_dict"])
        graph.eval()
        for parameter in graph.parameters():
            parameter.requires_grad_(False)

        filler, filling_summary = self._prepare_filler(datasets)
        validation_results = self._evaluate_split(
            graph, loaders["validation"], "validation", filler, {}
        )
        look_banks = {}
        look_summary = {"enabled": False, "banks": {}}
        if self.options.fit_look:
            look_banks, look_summary = self._fit_or_load_look(
                graph, loaders["look_train"], loaders["validation"], filler
            )
            validation_results.update(
                self._evaluate_split(
                    graph, loaders["validation"], "validation", filler, look_banks, look_only=True
                )
            )

        test_results: Dict[str, object] = {}
        statistics: Dict[str, object] = {"status": "test_sealed"}
        if self.options.allow_test:
            test_results = self._evaluate_split(
                graph, loaders["test"], "test", filler, look_banks
            )
            statistics = self._statistics(test_results)

        result = {
            "experiment_id": self.experiment_id,
            "status": "complete",
            "started_at_utc": started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection": self.selection.as_dict(),
            "options": asdict(self.options),
            "data_audit": audit,
            "graph": structure,
            "checkpoint": {
                "backbone_id": self._backbone_id(),
                "path": str(checkpoint_path),
                "epoch": checkpoint["epoch"],
                "validation_primary_score": checkpoint["score"],
                "sha256": sha256(checkpoint_path),
                "frozen_after_loading": True,
            },
            "filling": filling_summary,
            "look": look_summary,
            "validation": {name: item["metrics"] for name, item in validation_results.items()},
            "test": {name: item["metrics"] for name, item in test_results.items()},
            "statistics": statistics,
            "prediction_directory": str(self.prediction_dir),
        }
        write_json_atomic(result, self.experiment_dir / "experiment_result.json")
        self._register_completed_result(result)
        return result

    def _build_loaders(self):
        kwargs = {
            "labels_csv": self.config.labels_csv,
            "data_root": self.config.data_root,
            "image_size": self.config.image_size,
            "limit": self.options.smoke_limit,
        }
        datasets = {
            "train": UKBPairedEyeDataset(split="train", augment=True, **kwargs),
            "look_train": UKBPairedEyeDataset(split="train", augment=False, **kwargs),
            "validation": UKBPairedEyeDataset(split="validation", augment=False, **kwargs),
        }
        if self.options.allow_test:
            datasets["test"] = UKBPairedEyeDataset(split="test", augment=False, **kwargs)
        loaders = {
            name: make_loader(
                dataset,
                self.config.micro_batch_size,
                self.config.num_workers,
                train=name == "train",
                seed=self.selection.seed,
                sampler_power=self.config.sampler_power,
            )
            for name, dataset in datasets.items()
        }
        return loaders, datasets

    def _backbone_id(self) -> str:
        identity = {
            "architecture_id": self.selection.architecture_id,
            "seed": self.selection.seed,
            "num_classes": self.config.num_classes,
            "image_size": self.config.image_size,
            "epochs": self.config.epochs,
            "patience": self.config.patience,
            "effective_batch_size": self.config.effective_batch_size,
            "micro_batch_size": self.config.micro_batch_size,
            "num_workers": self.config.num_workers,
            "pretrained_lr": self.config.pretrained_lr,
            "new_layer_lr": self.config.new_layer_lr,
            "weight_decay": self.config.weight_decay,
            "warmup_epochs": self.config.warmup_epochs,
            "sampler_power": self.config.sampler_power,
            "amp": self.config.amp,
            "smoke_limit": self.options.smoke_limit,
            "labels_sha256": self.data_hash,
            "implementation_sha256": self.implementation_hash,
        }
        fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:12]
        return f"{self.selection.architecture_id}__complete_modalities__seed{self.selection.seed}__{fingerprint}"

    def _train_or_resume(self, graph, train_loader, validation_loader) -> Path:
        run_dir = Path(self.config.output_root) / "backbones" / self._backbone_id()
        checkpoint_path = run_dir / "best.pt"
        completion = run_dir / "training_complete.json"
        if checkpoint_path.exists() and completion.exists() and self.options.resume:
            return checkpoint_path
        if not self.options.train_if_missing:
            raise FileNotFoundError(checkpoint_path)
        write_json_atomic(
            {
                "config": self.config.as_dict(),
                "architecture_id": self.selection.architecture_id,
                "seed": self.selection.seed,
                "backbone_training": "complete_modalities_only",
                "labels_sha256": self.data_hash,
                "implementation_sha256": self.implementation_hash,
            },
            run_dir / "run_config.json",
        )
        graph, _ = train_complete_model(
            graph,
            train_loader,
            validation_loader,
            self.config,
            run_dir,
            self.device,
        )
        return checkpoint_path

    def _prepare_filler(self, datasets):
        if self.selection.filling_strategy == "normalized_mean":
            return NormalizedMeanFiller(), {
                "strategy": "normalized_mean",
                "definition": "zero after ImageNet normalization",
                "classifier_frozen": True,
            }
        gan_root = Path(self.config.output_root) / "generators" / self._gan_id()
        generators = {}
        directions = {
            "cfp_to_oct": ("cfp", "oct"),
            "oct_to_cfp": ("oct", "cfp"),
        }
        direction_summary = {}
        split_summary = None
        for direction_index, (name, (source, target)) in enumerate(directions.items()):
            direction_seed = self.selection.seed + 10_000 + direction_index
            seed_everything(direction_seed)
            train_loader, validation_loader, current_split = make_gan_loaders(
                datasets["train"],
                datasets["look_train"],
                self.config,
                self.selection.seed,
                loader_seed=direction_seed,
            )
            split_summary = current_split
            output_dir = gan_root / name
            checkpoint_path = output_dir / "best_generator.pt"
            completion = output_dir / "training_complete.json"
            if checkpoint_path.exists() and completion.exists() and self.options.resume:
                generator = load_generator(checkpoint_path, self.device)
            else:
                generator, _ = train_paired_cgan_direction(
                    source,
                    target,
                    train_loader,
                    validation_loader,
                    self.config,
                    output_dir,
                    self.device,
                )
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            generators[name] = generator
            direction_summary[name] = {
                "checkpoint": str(checkpoint_path),
                "epoch": checkpoint["epoch"],
                "validation_l1": checkpoint["validation_l1"],
                "checkpoint_sha256": sha256(checkpoint_path),
                "direction_seed": direction_seed,
            }
        write_json_atomic(split_summary, gan_root / "internal_split_manifest.json")
        filler = PairedCGANFiller(
            generators["cfp_to_oct"], generators["oct_to_cfp"], self.device
        )
        return filler, {
            "strategy": "paired_cgan",
            "classifier_frozen": True,
            "joint_training": False,
            "internal_training_split": split_summary,
            "generator_id": self._gan_id(),
            "directions": direction_summary,
            "alignment_limitation": "CFP and OCT are paired cross-view images, not pixel-registered.",
        }

    def _gan_id(self) -> str:
        identity = {
            "seed": self.selection.seed,
            "image_size": self.config.image_size,
            "validation_fraction": self.config.gan_validation_fraction,
            "epochs": self.config.gan_epochs,
            "patience": self.config.gan_patience,
            "batch_size": self.config.gan_batch_size,
            "num_workers": self.config.gan_num_workers,
            "learning_rate": self.config.gan_learning_rate,
            "beta1": self.config.gan_beta1,
            "lambda_l1": self.config.gan_lambda_l1,
            "base_channels": self.config.gan_base_channels,
            "smoke_limit": self.options.smoke_limit,
            "labels_sha256": self.data_hash,
            "implementation_sha256": self.implementation_hash,
        }
        fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:12]
        return f"paired_cgan__seed{self.selection.seed}__{fingerprint}"

    def _fit_or_load_look(self, graph, train_loader, validation_loader, filler):
        banks, summary = {}, {"enabled": True, "banks": {}}
        correction_nodes = (
            graph.correction_nodes
            if self.config.correction_nodes == ["all_available"]
            else self.config.correction_nodes
        )
        unknown_nodes = set(correction_nodes) - set(graph.correction_nodes)
        if unknown_nodes:
            raise ValueError(f"Unavailable correction nodes: {sorted(unknown_nodes)}")
        for pattern in self.config.missing_patterns:
            output_dir = self.experiment_dir / "look" / pattern
            selected_dir = output_dir / "selected"
            completion = output_dir / "look_complete.json"
            if selected_dir.exists() and completion.exists() and self.options.resume:
                artifacts = load_selected_bank(output_dir)
            else:
                artifacts, _ = greedy_fit_look(
                    graph,
                    train_loader,
                    validation_loader,
                    pattern,
                    correction_nodes,
                    self.config.downsample_factors,
                    self.config.latent_dims,
                    self.config.alpha_grid,
                    self.config.max_pca_rank,
                    self.device,
                    output_dir,
                    filler=filler,
                    primary_metric=self.config.primary_metric,
                )
            if any(artifact.filling_strategy != filler.name for artifact in artifacts):
                raise RuntimeError("LOOK artifact filling strategy does not match this experiment")
            matrix_records = analyze_look_bank(artifacts, output_dir)
            write_json_atomic(
                {"status": "complete", "pattern": pattern, "artifacts": len(artifacts)},
                completion,
            )
            banks[pattern] = artifacts
            summary["banks"][pattern] = {
                "selected": [
                    {
                        "node": artifact.node_name,
                        "factor": artifact.factor,
                        "latent_dim": artifact.latent_dim,
                        "alpha": artifact.alpha,
                        "ridge_lambda": artifact.ridge_lambda,
                        "train_r2": artifact.train_r2,
                    }
                    for artifact in artifacts
                ],
                "matrix_analysis_json": str(output_dir / "matrix_analysis.json"),
                "matrix_analysis_csv": str(output_dir / "matrix_analysis.csv"),
                "matrix_summary_png": str(output_dir / "matrix_summary.png"),
                "matrix_summary_pdf": str(output_dir / "matrix_summary.pdf"),
                "matrix_diagnostics": matrix_records,
            }
        return banks, summary

    def _evaluate_split(
        self, graph, loader, split: str, filler, look_banks, look_only: bool = False
    ):
        results = {}
        if not look_only:
            complete = predict(graph, loader, self.device)
            complete["metrics"] = classification_metrics(
                complete["labels"], complete["probabilities"]
            )
            results["complete"] = complete
            for pattern in self.config.missing_patterns:
                results[f"fill_{pattern}"] = evaluate_missing(
                    graph, loader, self.device, fixed_pattern=pattern, filler=filler
                )
        if look_banks:
            for pattern in self.config.missing_patterns:
                results[f"look_after_fill_{pattern}"] = evaluate_missing(
                    graph,
                    loader,
                    self.device,
                    fixed_pattern=pattern,
                    artifact_banks=look_banks,
                    filler=filler,
                )
            if self.options.evaluate_random_missing:
                for ratio in self.config.missing_ratios:
                    results[f"fill_random_{ratio:.1f}"] = evaluate_missing(
                        graph,
                        loader,
                        self.device,
                        random_ratio=ratio,
                        random_seed=self.selection.seed,
                        filler=filler,
                    )
                    results[f"look_after_fill_random_{ratio:.1f}"] = evaluate_missing(
                        graph,
                        loader,
                        self.device,
                        random_ratio=ratio,
                        random_seed=self.selection.seed,
                        artifact_banks=look_banks,
                        filler=filler,
                    )
        for name, result in results.items():
            self._save_prediction(result, split, name)
        return results

    def _save_prediction(self, result, split: str, name: str) -> None:
        self.prediction_dir.mkdir(parents=True, exist_ok=True)
        destination = self.prediction_dir / f"{split}__{name}.npz"
        temporary = destination.with_suffix(".npz.partial")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                labels=result["labels"],
                probabilities=result["probabilities"],
                participant_ids=result["participant_ids"],
                patterns=result.get("patterns", np.asarray(["complete"] * len(result["labels"]))),
            )
        temporary.replace(destination)

    def _statistics(self, test_results):
        comparisons = []
        for pattern in self.config.missing_patterns:
            look_key, fill_key = f"look_after_fill_{pattern}", f"fill_{pattern}"
            if look_key not in test_results:
                continue
            corrected, filled = test_results[look_key], test_results[fill_key]
            ci = participant_cluster_bootstrap(
                corrected["labels"],
                corrected["probabilities"],
                corrected["participant_ids"],
                iterations=self.options.bootstrap_iterations,
                seed=self.selection.seed,
            )
            paired = paired_participant_bootstrap(
                corrected["labels"],
                corrected["probabilities"],
                filled["probabilities"],
                corrected["participant_ids"],
                iterations=self.options.bootstrap_iterations,
                seed=self.selection.seed,
            )
            comparisons.append({"pattern": pattern, "look_ci": ci, **paired})
        if comparisons:
            adjusted = holm_adjust([item["p_value"] for item in comparisons])
            for item, value in zip(comparisons, adjusted):
                item["holm_p_value"] = float(value)
        return {"primary_endpoint": self.config.primary_metric, "comparisons": comparisons}

    def _register_completed_result(self, result: Dict[str, object]) -> None:
        registry_path = Path(self.config.output_root) / "experiment_registry.json"
        registry = json.loads(registry_path.read_text()) if registry_path.exists() else {"experiments": []}
        registry["experiments"] = [
            item for item in registry["experiments"]
            if item["experiment_id"] != self.experiment_id
        ]
        registry["experiments"].append(
            {
                "experiment_id": self.experiment_id,
                "status": result["status"],
                "completed_at_utc": result["completed_at_utc"],
                "result_file": str(self.experiment_dir / "experiment_result.json"),
            }
        )
        registry["experiments"].sort(key=lambda item: item["experiment_id"])
        write_json_atomic(registry, registry_path)
