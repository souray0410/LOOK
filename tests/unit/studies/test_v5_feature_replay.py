import json

import pytest
import torch

from look.studies.v5_feature_replay import ReplayJournal, tensor_digest


def record(value=1):
    return {"role": "train", "batch": 0, "participants": "people",
            "sites": {"joint_input": tensor_digest(torch.tensor([[value]], dtype=torch.float32))}}


def test_journal_resumes_only_an_identical_completed_batch(tmp_path):
    journal = ReplayJournal(tmp_path, "reference")
    first = record(); journal.commit(first); journal.commit(first)
    assert journal.read("train", 0) == first
    with pytest.raises(ValueError, match="changed"):
        journal.commit(record(2))


def test_check_journal_fails_closed_on_feature_drift(tmp_path):
    reference = record(); ReplayJournal(tmp_path, "reference").commit(reference)
    with pytest.raises(ValueError, match="differs"):
        ReplayJournal(tmp_path, "check").commit(record(2), expected=reference)


def test_missing_reference_batch_cannot_be_synthesized(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        ReplayJournal(tmp_path, "reference").read("development", 7)


def test_tensor_digest_binds_shape_dtype_and_values():
    value = tensor_digest(torch.arange(6, dtype=torch.float32).reshape(2, 3))
    assert value["shape"] == [2, 3] and value["dtype"] == "torch.float32"
    assert len(value["sha256"]) == 64
