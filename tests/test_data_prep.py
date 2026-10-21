"""
tests/test_data_prep.py
-----------------------
Unit tests verifying dataset integrity, exact sample counts (M=500), zero domain/prompt contamination,
and test manifest SHA-256 validity.
"""

import hashlib
import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_eval_unseen_apis_has_exactly_500_samples():
    path = DATA_DIR / "eval_unseen_apis.jsonl"
    assert path.exists(), "eval_unseen_apis.jsonl must exist"

    records = [json.loads(line) for line in open(path, "r", encoding="utf-8") if line.strip()]
    assert len(records) == 500, f"Expected exactly M=500 unseen API eval samples, found {len(records)}"
    
    # Confirm candidate_tools has 6 schemas in each prompt
    for r in records[:10]:
        assert len(r["candidate_tools"]) == 6
        assert "Available Tools:" in r["prompt"]


def test_eval_bfcl_subset_has_200_samples():
    path = DATA_DIR / "eval_bfcl_subset.jsonl"
    assert path.exists(), "eval_bfcl_subset.jsonl must exist"

    records = [json.loads(line) for line in open(path, "r", encoding="utf-8") if line.strip()]
    assert len(records) == 200, f"Expected 200 BFCL eval samples, found {len(records)}"


def test_zero_domain_overlap_between_train_and_unseen_eval():
    train_path = DATA_DIR / "train_sft.jsonl"
    unseen_path = DATA_DIR / "eval_unseen_apis.jsonl"

    train_domains = set()
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_domains.add(json.loads(line)["domain"])

    unseen_domains = set()
    with open(unseen_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                unseen_domains.add(json.loads(line)["domain"])

    overlap = train_domains.intersection(unseen_domains)
    assert len(overlap) == 0, f"Contamination detected! Overlapping domains: {overlap}"
    assert train_domains == {"finance", "commerce", "crm"}
    assert unseen_domains == {"infra", "travel"}


def test_zero_prompt_overlap_between_train_and_eval_sets():
    train_path = DATA_DIR / "train_sft.jsonl"
    unseen_path = DATA_DIR / "eval_unseen_apis.jsonl"
    bfcl_path = DATA_DIR / "eval_bfcl_subset.jsonl"

    train_prompts = set()
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_prompts.add(json.loads(line)["query"].strip().lower())

    with open(unseen_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line)["query"].strip().lower()
                assert q not in train_prompts, f"Contaminated query found: {q}"

    with open(bfcl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line)["query"].strip().lower()
                assert q not in train_prompts, f"Contaminated BFCL query found: {q}"


def test_manifest_sha256_hashes_match_disk():
    manifest_path = DATA_DIR / "test_manifest.json"
    assert manifest_path.exists(), "test_manifest.json must exist"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for filename, meta in manifest["file_hashes"].items():
        filepath = DATA_DIR / filename
        assert filepath.exists(), f"File {filename} in manifest does not exist"
        computed = compute_file_sha256(filepath)
        assert computed == meta["sha256"], f"Hash mismatch for {filename}: {computed} vs {meta['sha256']}"
