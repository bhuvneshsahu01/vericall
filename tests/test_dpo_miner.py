"""
tests/test_dpo_miner.py
-----------------------
Unit tests verifying DPO preference pair mining and error taxonomy separation.
"""

import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DPO_FILE = REPO_ROOT / "data" / "train_dpo.jsonl"


def test_dpo_dataset_structure_and_error_taxonomy():
    assert DPO_FILE.exists(), "train_dpo.jsonl must exist"

    pairs = [json.loads(line) for line in open(DPO_FILE, "r", encoding="utf-8") if line.strip()]
    assert len(pairs) == 1000, f"Expected 1,000 DPO pairs, found {len(pairs)}"

    valid_error_statuses = {"missing_arg", "hallucinated_arg", "type_error", "wrong_endpoint"}
    status_counts = {}

    for p in pairs:
        assert p["chosen_status"] == "passed", f"Chosen candidate not passed: {p}"
        assert p["rejected_status"] in valid_error_statuses, f"Unexpected rejected status: {p['rejected_status']}"
        assert len(p["prompt"]) > 20
        assert "Available Tools:" in p["prompt"]
        assert len(p["chosen"]) > 5
        assert len(p["rejected"]) > 5
        assert p["chosen"].endswith("<|im_end|>"), f"Chosen must end with <|im_end|>: {p['chosen']}"
        assert p["rejected"].endswith("<|im_end|>"), f"Rejected must end with <|im_end|>: {p['rejected']}" 
        status_counts[p["rejected_status"]] = status_counts.get(p["rejected_status"], 0) + 1

    # Verify target distribution
    assert status_counts["hallucinated_arg"] == 400
    assert status_counts["type_error"] == 300
    assert status_counts["missing_arg"] == 200
    assert status_counts["wrong_endpoint"] == 100
