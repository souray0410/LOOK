from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from look.data.missingness import missingness_plan
from look.runtime.state import atomic_write_json, durable_replace
from look.methods.imputation import MissingModalityFiller, NormalizedMeanFiller
from look.methods.operator import LOOKArtifact, forward_with_look
from look.evaluation.stability import logit_metrics, probabilities_from_logits


def save_prediction_bundle(result: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial.npz")
    np.savez_compressed(
        temporary,
        labels=result["labels"],
        probabilities=result["probabilities"],
        logits=result["logits"],
        scores=result["scores"],
        participant_ids=result["participant_ids"],
        patterns=result["patterns"],
    )

    durable_replace(temporary, path)
    if "missingness" in result:
        atomic_write_json(result["missingness"], path.with_suffix(".missingness.json"))


@torch.no_grad()
def evaluate_missing(
    graph,
    loader,
    device: torch.device,
    fixed_pattern: str | None = None,
    random_ratio: float | None = None,
    random_seed: int = 3407,
    artifact_banks: Mapping[str, Sequence[LOOKArtifact]] | None = None,
    filler: MissingModalityFiller | None = None,
) -> Dict[str, object]:
    if (fixed_pattern is None) == (random_ratio is None):
        raise ValueError("Specify exactly one of fixed_pattern or random_ratio")
    graph.eval()
    artifact_banks = artifact_banks or {}
    filler = filler or NormalizedMeanFiller()
    all_labels, all_logits, all_participants, all_patterns = [], [], [], []
    plan, missingness = ({}, None) if random_ratio is None else missingness_plan(
        loader.dataset.participant_ids, random_ratio, random_seed
    )
    for batch in tqdm(loader, desc="Missing-modality evaluation", leave=False):
        oct_tensor = batch["oct"].to(device, non_blocking=True)
        cfp_tensor = batch["cfp"].to(device, non_blocking=True)
        participants = list(batch["participant_id"])
        patterns = (
            [fixed_pattern] * len(participants)
            if fixed_pattern is not None
            else [plan[str(pid)] for pid in participants]
        )
        batch_logits = torch.empty(
            (len(participants), int(graph.num_classes)), device=device
        )
        for pattern in sorted(set(patterns)):
            indices = [index for index, value in enumerate(patterns) if value == pattern]
            index_tensor = torch.as_tensor(indices, device=device)
            counts = batch.get("counts")
            if counts is None:
                eye_indices = index_tensor
                selected_counts = None
            else:
                offsets = np.cumsum([0, *counts])
                eye_indices = torch.tensor([j for i in indices for j in range(offsets[i], offsets[i+1])], device=device)
                selected_counts = [counts[i] for i in indices]
            pattern_oct = oct_tensor.index_select(0, eye_indices)
            pattern_cfp = cfp_tensor.index_select(0, eye_indices)
            pattern_oct, pattern_cfp = filler.fill(pattern_oct, pattern_cfp, pattern)
            logits = forward_with_look(
                graph,
                pattern_oct,
                pattern_cfp,
                artifacts=artifact_banks.get(pattern, ()), **({'counts': selected_counts} if selected_counts is not None else {}),
            )
            batch_logits[index_tensor] = logits
        all_labels.append(batch["label"].numpy())
        all_logits.append(batch_logits.cpu().numpy())
        all_participants.extend(participants)
        all_patterns.extend(patterns)
    labels = np.concatenate(all_labels)
    logits = np.concatenate(all_logits).astype(np.float64)
    probabilities = probabilities_from_logits(logits)
    return {
        "labels": labels,
        "probabilities": probabilities,
        "logits": logits,
        "scores": logits[:, 1] - logits[:, 0],
        "participant_ids": np.asarray(all_participants),
        "patterns": np.asarray(all_patterns),
        "metrics": logit_metrics(labels, logits),
        **({"missingness": missingness} if missingness is not None else {}),
    }
