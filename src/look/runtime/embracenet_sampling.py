"""Sampling-state checkpoint helpers for the EmbraceNet MHD prototype."""
from __future__ import annotations

from collections.abc import MutableMapping

from look.models.embracenet import capture_embracenet_sampling, restore_embracenet_sampling

PROGRESS_KEY = "embracenet_sampling_state"


def stash_embracenet_sampling(graph, progress: MutableMapping[str, object]) -> None:
    """Attach the method-private RNG state to an existing optimizer-boundary progress record."""
    progress[PROGRESS_KEY] = capture_embracenet_sampling(graph)


def restore_embracenet_sampling_from_progress(graph, progress: MutableMapping[str, object]) -> None:
    if PROGRESS_KEY not in progress:
        raise ValueError("EmbraceNet checkpoint progress lacks method-private sampling state")
    restore_embracenet_sampling(graph, progress[PROGRESS_KEY])
