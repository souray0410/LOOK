from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from .data import participant_missing_pattern
from .filling import MissingModalityFiller, NormalizedMeanFiller
from .look import LOOKArtifact, forward_with_look
from .metrics import classification_metrics


def save_prediction_bundle(result: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        labels=result["labels"],
        probabilities=result["probabilities"],
        participant_ids=result["participant_ids"],
        patterns=result["patterns"],
    )


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
    all_labels, all_probabilities, all_participants, all_patterns = [], [], [], []
    for batch in tqdm(loader, desc="Missing-modality evaluation", leave=False):
        oct_tensor = batch["oct"].to(device, non_blocking=True)
        cfp_tensor = batch["cfp"].to(device, non_blocking=True)
        participants = list(batch["participant_id"])
        patterns = (
            [fixed_pattern] * len(participants)
            if fixed_pattern is not None
            else [participant_missing_pattern(pid, random_ratio, random_seed) for pid in participants]
        )
        batch_probabilities = torch.empty((len(participants), 5), device=device)
        for pattern in sorted(set(patterns)):
            indices = [index for index, value in enumerate(patterns) if value == pattern]
            index_tensor = torch.as_tensor(indices, device=device)
            pattern_oct = oct_tensor.index_select(0, index_tensor)
            pattern_cfp = cfp_tensor.index_select(0, index_tensor)
            pattern_oct, pattern_cfp = filler.fill(pattern_oct, pattern_cfp, pattern)
            logits = forward_with_look(
                graph,
                pattern_oct,
                pattern_cfp,
                artifacts=artifact_banks.get(pattern, ()),
            )
            batch_probabilities[index_tensor] = torch.softmax(logits, dim=1)
        all_labels.append(batch["label"].numpy())
        all_probabilities.append(batch_probabilities.cpu().numpy())
        all_participants.extend(participants)
        all_patterns.extend(patterns)
    labels = np.concatenate(all_labels)
    probabilities = np.concatenate(all_probabilities)
    return {
        "labels": labels,
        "probabilities": probabilities,
        "participant_ids": np.asarray(all_participants),
        "patterns": np.asarray(all_patterns),
        "metrics": classification_metrics(labels, probabilities),
    }
