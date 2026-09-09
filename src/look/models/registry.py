"""Project-owned complete native models; independent of method operators."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelResolution:
    status: str
    request_id: str
    model: Any = None
    bundle: Path | None = None
    request_file: Path | None = None


def resolve_native_model(request, *, shared_models, project_models, device="cpu"):
    """Exact hit -> verified independent local copy; miss -> explicit pending task.

    The study runner must wait for baseline training acceptance before creating
    a method experiment. This function never silently starts from random weights.
    """
    from mhd_framework.models.artifacts import (resolve_or_request,
        materialize_bundle, load_bundle)
    result = resolve_or_request(shared_models, request)
    if result["status"] != "ready":
        return ModelResolution("pending", result["request_id"],
            request_file=Path(result["request_file"]))
    destination = Path(project_models)/result["request_id"]
    materialize_bundle(result["bundle"], destination, request)
    model, manifest = load_bundle(destination, request, device=device)
    return ModelResolution("ready", result["request_id"], model=model, bundle=destination)


def require_native_model(*args, **kwargs):
    result = resolve_native_model(*args, **kwargs)
    if result.status != "ready":
        raise RuntimeError("Native model training is pending: " + str(result.request_file))
    return result.model
