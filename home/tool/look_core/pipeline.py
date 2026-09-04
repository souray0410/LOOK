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
from .data import (
    UKBBilateralVisitDataset,
    make_loader,
    validate_reference_table,
)
from .distributed import launch_ddp_stage
from .evaluate import evaluate_missing
from .filling import NormalizedMeanFiller, PairedCGANFiller, RawZeroFiller
from .gan import load_generator, make_gan_loaders, train_paired_cgan_direction
from .graph import build_resnet50_mhd_graph, graph_summary
from .look import greedy_fit_look, load_selected_bank, prepare_complete_pca_bank, validate_global_factor_bank
from .matrix_analysis import analyze_look_bank
from .metrics import (
    classification_metrics,
    holm_adjust,
    paired_participant_bootstrap,
    participant_cluster_bootstrap,
)
from .quality_audit import evidence_stratified_validation
from .reproducibility import (
    backbone_implementation_sha256,
    environment_manifest,
    implementation_sha256,
    seed_everything,
    sha256,
    write_json_atomic,
)
from .state import PipelineState, quarantine
from .train import predict


@dataclass(frozen=True)
class PipelineOptions:
    train_if_missing: bool = True
    fit_look: bool = True
    evaluate_random_missing: bool = True
    evaluate_missing_baselines: bool = True
    phase: str = "validation"
    frozen_manifest: Optional[str] = None
    look_load_only: bool = False
    resume: bool = True
    restart: bool = False
    check_all_image_paths: bool = False
    smoke_limit: Optional[int] = None
    bootstrap_iterations: int = 2000
    gpu_devices: tuple[int, ...] = (0,)

    def validate(self, selection: ExperimentSelection) -> None:
        if self.resume and self.restart:
            raise ValueError("resume and restart are mutually exclusive")
        if self.phase not in {"validation", "test"}:
            raise ValueError("phase must be validation or test")
        if self.phase == "test" and not self.frozen_manifest:
            raise ValueError("Test access requires a frozen configuration manifest")
        if self.phase == "test" and self.smoke_limit is not None:
            raise ValueError("Smoke subsets cannot produce final test results")
        if self.bootstrap_iterations < 100:
            raise ValueError("bootstrap_iterations is too small for inferential reporting")
        if not self.gpu_devices or len(set(self.gpu_devices)) != len(self.gpu_devices):
            raise ValueError("gpu_devices must be a non-empty unique sequence")


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
        self.natural_data_hash = sha256(config.natural_labels_csv)
        self.implementation_hash = implementation_sha256(self.project_root)
        self.backbone_implementation_hash = backbone_implementation_sha256(
            self.project_root
        )
        self.experiment_id = self._experiment_id()
        self.experiment_dir = Path(config.output_root) / "experiments" / self.experiment_id
        self.prediction_dir = self.experiment_dir / "predictions"

    def _experiment_id(self) -> str:
        scientific_config = self.config.as_dict()
        for path_key in (
            "image_root",
            "labels_csv",
            "natural_labels_csv",
            "preprocess_cache_root",
            "output_root",
            "cache_root",
        ):
            scientific_config.pop(path_key, None)
        identity = {
            "config": scientific_config,
            "selection": self.selection.as_dict(),
            "fit_look": self.options.fit_look,
            "evaluate_random_missing": self.options.evaluate_random_missing,
            "evaluate_missing_baselines": self.options.evaluate_missing_baselines,
            "smoke_limit": self.options.smoke_limit,
            "bootstrap_iterations": self.options.bootstrap_iterations,
            "labels_sha256": self.data_hash,
            "natural_labels_sha256": self.natural_data_hash,
            "implementation_sha256": self.implementation_hash,
        }
        fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:12]
        return f"{self.selection.experiment_id}__{fingerprint}"

    def run(self) -> Dict[str, object]:
        result_path = self.experiment_dir / f"{self.options.phase}_result.json"
        if self.options.restart and result_path.is_file():
            quarantine(
                result_path,
                Path(self.config.cache_root) / "quarantine",
                f"explicit {self.options.phase} restart",
            )
        if self.options.resume and result_path.is_file():
            completed = self._read_valid_completed_result(result_path)
            if completed is not None:
                return completed
        state_root = Path(self.config.cache_root) / "pipeline_state"
        state_config = {
            "experiment_id": self.experiment_id,
            "config": self.config.as_dict(),
            "selection": self.selection.as_dict(),
            "options": asdict(self.options),
        }
        with PipelineState(
            state_root,
            f"experiment__{self.experiment_id}__{self.options.phase}",
            state_config,
            [self.config.labels_csv, self.config.natural_labels_csv],
        ) as state:
            result = self._run_impl(result_path)
            state.complete([result_path])
            return result

    def _read_valid_completed_result(self, result_path: Path) -> Optional[Dict[str, object]]:
        try:
            completed = json.loads(result_path.read_text(encoding="utf-8"))
            checkpoint = completed["checkpoint"]
            checkpoint_path = Path(str(checkpoint["path"]))
            valid = (
                completed.get("status") == "complete"
                and completed.get("experiment_id") == self.experiment_id
                and checkpoint_path.is_file()
                and sha256(checkpoint_path) == checkpoint.get("sha256")
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            valid = False
        if valid:
            return completed
        quarantine(
            result_path,
            Path(self.config.cache_root) / "quarantine",
            "invalid or incomplete experiment result",
        )
        return None

    def _run_impl(self, result_path: Path) -> Dict[str, object]:
        seed_everything(self.selection.seed)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        started = datetime.now(timezone.utc).isoformat()
        audit = validate_reference_table(
            self.config.labels_csv,
            self.config.image_root,
            check_paths=self.options.check_all_image_paths,
        )
        natural_audit = validate_reference_table(
            self.config.natural_labels_csv,
            self.config.image_root,
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
                "natural_data_audit": natural_audit,
                "natural_labels_sha256": self.natural_data_hash,
                "backbone_id": self._backbone_id(),
                "implementation_sha256": self.implementation_hash,
                "started_at_utc": started,
            }
        )
        write_json_atomic(manifest, self.experiment_dir / f"{self.options.phase}_manifest.json")

        loaders, datasets = self._build_loaders()
        summary_graph = build_resnet50_mhd_graph(
            self.selection.fusion_position,
            self.config.num_classes,
            1,
            self.config.image_size,
            torch.device("cpu"),
            pretrained=False,
            classifier_dropout=self.config.classifier_dropout,
            label_smoothing=self.config.label_smoothing,
        )
        structure = graph_summary(summary_graph)
        del summary_graph
        write_json_atomic(structure, self.experiment_dir / "graph_summary.json")
        checkpoint_path = self._train_or_resume()

        graph, checkpoint = self._load_frozen_graph(checkpoint_path)
        pca_bank = None
        if self.options.fit_look and not self.options.look_load_only:
            pca_bank = self._prepare_shared_pca(graph, loaders['look_train'], checkpoint_path)

        filler = None
        filling_summary = {
            "strategy": "not_evaluated",
            "reason": "complete_modality_classifier_search",
        }
        if self.options.fit_look or self.options.evaluate_missing_baselines:
            filler, filling_summary = self._prepare_filler(datasets)
        validation_results = {}
        look_banks = {}
        look_summary = {"enabled": False, "banks": {}}
        if self.options.fit_look:
            look_banks, look_summary = self._fit_or_load_look(
                graph, loaders["look_train"], loaders["validation"], filler, pca_bank
            )
        if self.options.phase == "validation":
            validation_results = self._evaluate_split(
                graph, loaders["validation"], "validation", filler, {}
            )
            if look_banks:
                validation_results.update(
                    self._evaluate_split(
                        graph, loaders["validation"], "validation", filler, look_banks, look_only=True
                    )
                )

        validation_diagnostics: Dict[str, object] = {}
        if self.options.phase == "validation" and "complete" in validation_results:
            complete = validation_results["complete"]
            validation_diagnostics = evidence_stratified_validation(
                self.config.labels_csv,
                complete["labels"],
                complete["probabilities"],
                complete["participant_ids"],
                require_complete_split=self.options.smoke_limit is None,
            )
            write_json_atomic(
                validation_diagnostics,
                self.experiment_dir / "validation_evidence_diagnostics.json",
            )

        test_results: Dict[str, object] = {}
        statistics: Dict[str, object] = {"status": "test_sealed"}
        if self.options.phase == "test":
            primary_test = self._evaluate_split(
                graph, loaders["primary_test"], "primary_test", filler, look_banks
            )
            natural_test = self._evaluate_split(
                graph, loaders["natural_test"], "natural_test", filler, look_banks
            )
            test_results = {
                **{f"primary_{name}": value for name, value in primary_test.items()},
                **{f"natural_{name}": value for name, value in natural_test.items()},
            }
            statistics = {
                "primary": self._statistics(primary_test),
                "natural": self._statistics(natural_test),
                "natural_distribution_is_secondary": True,
            }

        result = {
            "experiment_id": self.experiment_id,
            "phase": self.options.phase,
            "status": "complete",
            "started_at_utc": started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection": self.selection.as_dict(),
            "options": asdict(self.options),
            "data_audit": audit,
            "natural_data_audit": natural_audit,
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
            "validation_diagnostics": validation_diagnostics,
            "test": {name: item["metrics"] for name, item in test_results.items()},
            "statistics": statistics,
            "prediction_directory": str(self.prediction_dir),
        }
        write_json_atomic(result, result_path)
        self._register_completed_result(result)
        return result

    def _build_loaders(self):
        kwargs = {
            "labels_csv": self.config.labels_csv,
            "data_root": self.config.image_root,
            "image_size": self.config.image_size,
            "limit": self.options.smoke_limit,
            "preprocess_cache_root": self.config.preprocess_cache_root,
        }
        datasets = {
            "train": UKBBilateralVisitDataset(split="train", augment=True, **kwargs),
            "look_train": UKBBilateralVisitDataset(split="train", augment=False, **kwargs),
            "validation": UKBBilateralVisitDataset(split="validation", augment=False, **kwargs),
        }
        if self.options.phase == "test":
            datasets["primary_test"] = UKBBilateralVisitDataset(
                split="test", augment=False, **kwargs
            )
            datasets["natural_test"] = UKBBilateralVisitDataset(
                labels_csv=self.config.natural_labels_csv,
                data_root=self.config.image_root,
                split="test",
                image_size=self.config.image_size,
                augment=False,
                limit=self.options.smoke_limit,
                preprocess_cache_root=self.config.preprocess_cache_root,
            )
        loaders = {
            name: make_loader(
                dataset,
                self.config.micro_batch_size,
                self.config.num_workers,
                train=name == "train",
                seed=self.selection.seed,
                sampling_strategy=self.config.sampling_strategy,
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
            "sampling_strategy": self.config.sampling_strategy,
            "loss_name": self.config.loss_name,
            "label_smoothing": self.config.label_smoothing,
            "classifier_dropout": self.config.classifier_dropout,
            "training_strategy": self.config.training_strategy,
            "amp": self.config.amp,
            "world_size": self.config.world_size,
            "smoke_limit": self.options.smoke_limit,
            "labels_sha256": self.data_hash,
            "backbone_implementation_sha256": self.backbone_implementation_hash,
        }
        fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:12]
        return f"{self.selection.architecture_id}__complete_modalities__seed{self.selection.seed}__{fingerprint}"

    def _train_or_resume(self) -> Path:
        run_dir = Path(self.config.output_root) / "backbones" / self._backbone_id()
        checkpoint_path = run_dir / "best.pt"
        completion = run_dir / "training_complete.json"
        if self.options.restart and run_dir.exists():
            quarantine(
                run_dir,
                Path(self.config.cache_root) / "quarantine",
                "explicit classifier restart",
            )
        if checkpoint_path.is_file() and completion.is_file() and self.options.resume:
            try:
                summary = json.loads(completion.read_text(encoding="utf-8"))
                valid = (
                    summary.get("status") == "complete"
                    and summary.get("labels_sha256") == self.data_hash
                    and summary.get("backbone_implementation_sha256")
                    == self.backbone_implementation_hash
                    and summary.get("checkpoint_sha256") == sha256(checkpoint_path)
                    and summary.get("training_strategy") == "end_to_end_finetuning"
                    and summary.get("training_stage") == "complete_modalities"
                )
            except (OSError, ValueError, json.JSONDecodeError):
                valid = False
            if valid:
                return checkpoint_path
            quarantine(
                completion,
                Path(self.config.cache_root) / "quarantine",
                "classifier completion metadata or checkpoint hash mismatch",
            )
        elif completion.is_file():
            quarantine(
                completion,
                Path(self.config.cache_root) / "quarantine",
                "classifier completion exists without portable checkpoint",
            )
        elif checkpoint_path.is_file() and not (run_dir / "last").is_dir():
            quarantine(
                checkpoint_path,
                Path(self.config.cache_root) / "quarantine",
                "portable checkpoint exists without completion or resumable state",
            )
        if not self.options.train_if_missing:
            raise FileNotFoundError(checkpoint_path)
        write_json_atomic(
            {
                "config": self.config.as_dict(),
                "architecture_id": self.selection.architecture_id,
                "seed": self.selection.seed,
                "backbone_training": "complete_modalities_only",
                "training_strategy": self.config.training_strategy,
                "labels_sha256": self.data_hash,
                "backbone_implementation_sha256": self.backbone_implementation_hash,
            },
            run_dir / "run_config.json",
        )
        launch_ddp_stage(
            self.project_root,
            {
                "config": self.config.as_dict(),
                "seed": self.selection.seed,
                "fusion_position": self.selection.fusion_position,
                "smoke_limit": self.options.smoke_limit,
                "run_dir": str(run_dir),
            },
            "classifier",
            self.options.gpu_devices,
            Path(self.config.cache_root) / "partial" / f"classifier__{self._backbone_id()}.json",
        )
        return checkpoint_path

    def _prepare_filler(self, datasets):
        if self.selection.filling_strategy == "raw_zero":
            return RawZeroFiller(), {
                "strategy": "raw_zero",
                "definition": "raw-image zero represented after ImageNet normalization",
                "classifier_frozen": True,
            }
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
                if not self.options.train_if_missing:
                    raise FileNotFoundError(f"Frozen test generator is incomplete: {output_dir}")
                launch_ddp_stage(
                    self.project_root,
                    {
                        "config": self.config.as_dict(),
                        "seed": self.selection.seed,
                        "direction_seed": direction_seed,
                        "source": source,
                        "target": target,
                        "smoke_limit": self.options.smoke_limit,
                        "output_dir": str(output_dir),
                    },
                    "gan",
                    self.options.gpu_devices,
                    Path(self.config.cache_root) / "partial" / f"gan__{self._gan_id()}__{name}.json",
                )
                generator = load_generator(checkpoint_path, self.device)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            generators[name] = generator
            direction_summary[name] = {
                "checkpoint": str(checkpoint_path),
                "epoch": checkpoint["epoch"],
                "validation_l1": checkpoint["validation_l1"],
                "checkpoint_sha256": sha256(checkpoint_path),
                "direction_seed": direction_seed,
            }
        if self.options.phase == "validation":
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
            "effective_batch_size": self.config.gan_effective_batch_size,
            "num_workers": self.config.gan_num_workers,
            "learning_rate": self.config.gan_learning_rate,
            "beta1": self.config.gan_beta1,
            "lambda_l1": self.config.gan_lambda_l1,
            "base_channels": self.config.gan_base_channels,
            "world_size": self.config.world_size,
            "smoke_limit": self.options.smoke_limit,
            "labels_sha256": self.data_hash,
            "implementation_sha256": self.implementation_hash,
        }
        fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:12]
        return f"paired_cgan__seed{self.selection.seed}__{fingerprint}"

    def _load_frozen_graph(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        graph = build_resnet50_mhd_graph(
            self.selection.fusion_position, self.config.num_classes,
            self.config.micro_batch_size, self.config.image_size, self.device,
            pretrained=False, classifier_dropout=self.config.classifier_dropout,
            label_smoothing=self.config.label_smoothing,
        )
        if checkpoint['architecture_id'] != self.selection.architecture_id:
            raise RuntimeError('Checkpoint architecture does not match the selected experiment')
        if (checkpoint.get('backbone_training') != 'complete_modalities_only'
                or checkpoint.get('training_strategy') != 'end_to_end_finetuning'
                or checkpoint.get('training_stage') != 'complete_modalities'
                or checkpoint.get('labels_sha256') != self.data_hash):
            raise RuntimeError('Checkpoint is not the matching complete-modality baseline')
        graph.load_state_dict(checkpoint['graph_state_dict'])
        graph.eval()
        for parameter in graph.parameters():
            parameter.requires_grad_(False)
        return graph, checkpoint

    def _prepare_shared_pca(self, graph, train_loader, checkpoint_path):
        nodes = graph.correction_nodes if self.config.correction_nodes == ['all_available'] else self.config.correction_nodes
        identity = dict(backbone_id=self._backbone_id(), checkpoint_sha256=sha256(checkpoint_path),
            labels_sha256=self.data_hash, feature_implementation_sha256=self.backbone_implementation_hash,
            image_size=self.config.image_size, feature_batch_size=self.config.micro_batch_size,
            split='train', augment=False, order='reference_table', smoke_limit=self.options.smoke_limit)
        return prepare_complete_pca_bank(graph, train_loader, nodes,
            self.config.downsample_factors, self.config.max_pca_rank, self.device,
            Path(self.config.output_root) / 'pca', identity,
            Path(self.config.cache_root) / 'quarantine')

    def prepare_shared_pca(self):
        """Explicit stage for all full-feature PCs before any missing-modality fits."""
        seed_everything(self.selection.seed)
        checkpoint_path = self._train_or_resume()
        graph, _ = self._load_frozen_graph(checkpoint_path)
        dataset = UKBBilateralVisitDataset(labels_csv=self.config.labels_csv,
            data_root=self.config.image_root, split='train', augment=False,
            image_size=self.config.image_size, limit=self.options.smoke_limit,
            preprocess_cache_root=self.config.preprocess_cache_root)
        loader = make_loader(dataset, self.config.micro_batch_size, self.config.num_workers,
            train=False, seed=self.selection.seed, sampling_strategy=self.config.sampling_strategy)
        bank = self._prepare_shared_pca(graph, loader, checkpoint_path)
        return [basis.source_id for basis in bank.values()]

    def _fit_or_load_look(self, graph, train_loader, validation_loader, filler, pca_bank):
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
            factor_selection_path = output_dir / "factor_selection.json"
            if (
                selected_dir.exists()
                and completion.exists()
                and factor_selection_path.exists()
                and self.options.resume
            ):
                artifacts = load_selected_bank(output_dir)
            else:
                if self.options.look_load_only:
                    raise FileNotFoundError(f"Frozen test LOOK bank is incomplete: {output_dir}")
                artifacts, _ = greedy_fit_look(
                    graph,
                    train_loader,
                    validation_loader,
                    pattern,
                    correction_nodes,
                    self.config.downsample_factors,
                    self.config.latent_dims,
                    self.config.max_pca_rank,
                    self.device,
                    output_dir,
                    pca_bank=pca_bank,
                    filler=filler,
                    primary_metric=self.config.primary_metric,
                    resume=self.options.resume,
                )
            if any(artifact.filling_strategy != filler.name for artifact in artifacts):
                raise RuntimeError("LOOK artifact filling strategy does not match this experiment")
            factor_selection = json.loads(
                factor_selection_path.read_text(encoding="utf-8")
            )
            validate_global_factor_bank(
                artifacts,
                correction_nodes,
                int(factor_selection["selected_factor"]),
            )
            if self.options.look_load_only:
                matrix_path = output_dir / "matrix_analysis.json"
                if not matrix_path.is_file():
                    raise FileNotFoundError(f"Frozen matrix analysis is incomplete: {matrix_path}")
                matrix_records = json.loads(
                    matrix_path.read_text(encoding="utf-8")
                )["matrices"]
            else:
                matrix_records = analyze_look_bank(artifacts, output_dir)
                write_json_atomic(
                    {
                        "status": "complete",
                        "pattern": pattern,
                        "artifacts": len(artifacts),
                        "selected_factor": factor_selection["selected_factor"],
                        "selection_file": str(factor_selection_path),
                    },
                    completion,
                )
            banks[pattern] = artifacts
            summary["banks"][pattern] = {
                "selected": [
                    {
                        "node": artifact.node_name,
                        "factor": artifact.factor,
                        "latent_dim": artifact.latent_dim,
                        "pca_source_id": artifact.pca_source_id,
                        "ridge_lambda": artifact.ridge_lambda,
                        "train_r2": artifact.train_r2,
                    }
                    for artifact in artifacts
                ],
                "factor_selection": factor_selection,
                "factor_selection_json": str(factor_selection_path),
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
            if self.options.evaluate_missing_baselines:
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
            primary_ci = participant_cluster_bootstrap(
                corrected["labels"],
                corrected["probabilities"],
                corrected["participant_ids"],
                metric=self.config.primary_metric,
                iterations=self.options.bootstrap_iterations,
                seed=self.selection.seed,
            )
            macro_f1_ci = participant_cluster_bootstrap(
                corrected["labels"],
                corrected["probabilities"],
                corrected["participant_ids"],
                metric="macro_f1",
                iterations=self.options.bootstrap_iterations,
                seed=self.selection.seed,
            )
            paired = paired_participant_bootstrap(
                corrected["labels"],
                corrected["probabilities"],
                filled["probabilities"],
                corrected["participant_ids"],
                metric=self.config.primary_metric,
                iterations=self.options.bootstrap_iterations,
                seed=self.selection.seed,
            )
            paired_macro_f1 = paired_participant_bootstrap(
                corrected["labels"],
                corrected["probabilities"],
                filled["probabilities"],
                corrected["participant_ids"],
                metric="macro_f1",
                iterations=self.options.bootstrap_iterations,
                seed=self.selection.seed,
            )
            comparisons.append({
                "pattern": pattern,
                "look_primary_ci": primary_ci,
                "look_macro_f1_ci": macro_f1_ci,
                "paired_primary": paired,
                "paired_macro_f1": paired_macro_f1,
                "p_value": paired["p_value"],
            })
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
            if not (
                item["experiment_id"] == self.experiment_id
                and item.get("phase", "validation") == self.options.phase
            )
        ]
        registry["experiments"].append(
            {
                "experiment_id": self.experiment_id,
                "phase": self.options.phase,
                "status": result["status"],
                "completed_at_utc": result["completed_at_utc"],
                "result_file": str(self.experiment_dir / f"{self.options.phase}_result.json"),
            }
        )
        registry["experiments"].sort(key=lambda item: item["experiment_id"])
        write_json_atomic(registry, registry_path)
