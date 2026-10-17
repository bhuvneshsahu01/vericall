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
    assert len(pairs) >= 500, f"Expected at least 500 DPO pairs, found {len(pairs)}"

    valid_error_statuses = {"missing_arg", "hallucinated_arg", "type_error", "wrong_endpoint"}

    for p in pairs:
        assert p["chosen_status"] == "success", f"Chosen candidate not successful: {p}"
        assert p["rejected_status"] in valid_error_statuses, f"Unexpected rejected status: {p['rejected_status']}"
        assert len(p["prompt"]) > 5
        assert len(p["chosen"]) > 5
        assert len(p["rejected"]) > 5
