"""
vericall/data/mine_dpo.py
-------------------------
Execution-Verified Near-Miss Preference Miner for DPO.
Samples candidate tool calls per prompt, verifies each against the FastAPI sandbox,
and pairs passed calls (chosen) against near-miss failed calls (rejected) with
the exact target failure distribution:
- 40% hallucinated_arg (extra un-modeled keys)
- 30% type_error (string-instead-of-int, string-instead-of-bool, lowercase enums, bad dates)
- 20% missing_arg (dropped required parameter)
- 10% wrong_endpoint (decoy tool selected from prompt candidates)
"""

import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from vericall.sandbox.validator import ExecutionValidator

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"

EXTRA_KEYS = [
    ("timeout_seconds", 30),
    ("notify_user", True),
    ("debug_mode", False),
    ("execution_priority", "HIGH"),
    ("client_timestamp", "2026-10-21T12:00:00Z"),
    ("user_confirmation", True),
    ("bypass_cache", True),
    ("correlation_id", "corr-9921"),
    ("caller_ip", "192.168.1.100"),
    ("retry_limit", 3),
]


def mutate_hallucinated_arg(func_name: str, args: Dict[str, Any], validator: ExecutionValidator) -> Tuple[Dict[str, Any], Any]:
    """Adds a plausible extra parameter not defined in the schema."""
    for key, val in EXTRA_KEYS:
        if key not in args:
            cand_args = copy.deepcopy(args)
            cand_args[key] = val
            res = validator.validate_call({"name": func_name, "arguments": cand_args}, expected_func=func_name)
            if res.status == "hallucinated_arg":
                return cand_args, res

    # Fallback extra key
    cand_args = copy.deepcopy(args)
    cand_args["__unauthorized_extra_param__"] = True
    res = validator.validate_call({"name": func_name, "arguments": cand_args}, expected_func=func_name)
    return cand_args, res


def mutate_type_error(func_name: str, args: Dict[str, Any], validator: ExecutionValidator) -> Tuple[Dict[str, Any], Any]:
    """Mutates an argument to trigger type_error (strict=True violations, bad regex, lowercase enums)."""
    cand_args = copy.deepcopy(args)

    # 1. Try mutating boolean to string "true"/"false"
    for k, v in cand_args.items():
        if isinstance(v, bool):
            cand_args[k] = "true" if v else "false"
            res = validator.validate_call({"name": func_name, "arguments": cand_args}, expected_func=func_name)
            if res.status == "type_error":
                return cand_args, res

    # 2. Try mutating int to string "100"
    for k, v in cand_args.items():
        if isinstance(v, int) and not isinstance(v, bool):
            cand_args[k] = str(v)
            res = validator.validate_call({"name": func_name, "arguments": cand_args}, expected_func=func_name)
            if res.status == "type_error":
                return cand_args, res

    # 3. Try mutating string enum to lowercase or invalid string
    for k, v in cand_args.items():
        if isinstance(v, str) and v.isupper() and len(v) > 1:
            cand_args[k] = v.lower()  # lowercase enum fails strict Enum
            res = validator.validate_call({"name": func_name, "arguments": cand_args}, expected_func=func_name)
            if res.status == "type_error":
                return cand_args, res

    # 4. Try mutating date format YYYY-MM-DD to DD/MM/YYYY
    for k, v in cand_args.items():
        if isinstance(v, str) and "-" in v and len(v) == 10:
            cand_args[k] = "11/25/2026"
            res = validator.validate_call({"name": func_name, "arguments": cand_args}, expected_func=func_name)
            if res.status == "type_error":
                return cand_args, res

    # 5. Fallback: replace any value with nested dict
    first_k = list(cand_args.keys())[0] if cand_args else "dummy"
    cand_args[first_k] = {"invalid_nested_type": 123}
    res = validator.validate_call({"name": func_name, "arguments": cand_args}, expected_func=func_name)
    return cand_args, res


def mutate_missing_arg(func_name: str, args: Dict[str, Any], validator: ExecutionValidator) -> Tuple[Dict[str, Any], Any]:
    """Drops one required argument."""
    cand_args = copy.deepcopy(args)
    if cand_args:
        # Drop the first key
        k_to_drop = list(cand_args.keys())[0]
        del cand_args[k_to_drop]

    res = validator.validate_call({"name": func_name, "arguments": cand_args}, expected_func=func_name)
    return cand_args, res


def mutate_wrong_endpoint(func_name: str, args: Dict[str, Any], candidate_tools: List[Dict[str, Any]], validator: ExecutionValidator) -> Tuple[str, Dict[str, Any], Any]:
    """Picks one of the decoy endpoints available in the prompt candidates."""
    decoys = [t["name"] for t in candidate_tools if t["name"] != func_name]
    decoy_fn = decoys[0] if decoys else "invalid_tool"
    res = validator.validate_call({"name": decoy_fn, "arguments": args}, expected_func=func_name)
    return decoy_fn, args, res


def mine_dpo_pairs(num_pairs: int = 1000) -> List[Dict[str, Any]]:
    """
    Constructs execution-verified DPO preference pairs (chosen vs rejected)
    with exact target failure distribution:
    - 400 (40%) hallucinated_arg
    - 300 (30%) type_error
    - 200 (20%) missing_arg
    - 100 (10%) wrong_endpoint
    """
    train_file = DATA_DIR / "train_sft.jsonl"
    if not train_file.exists():
        from .prep import run_pipeline
        run_pipeline()

    records = []
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    assert len(records) >= num_pairs, f"Need at least {num_pairs} train records, found {len(records)}"
    validator = ExecutionValidator()
    dpo_pairs = []

    # Target counts
    n_hal = int(num_pairs * 0.40)   # 400
    n_type = int(num_pairs * 0.30)  # 300
    n_miss = int(num_pairs * 0.20)  # 200
    n_wrong = num_pairs - (n_hal + n_type + n_miss)  # 100

    target_types = (
        ["hallucinated_arg"] * n_hal +
        ["type_error"] * n_type +
        ["missing_arg"] * n_miss +
        ["wrong_endpoint"] * n_wrong
    )
    # Shuffle distribution order deterministically
    random.Random(42).shuffle(target_types)

    print(f"[*] Mining {num_pairs} execution-verified DPO pairs...")
    print(f"    Target distribution: 40% hallucinated_arg ({n_hal}), 30% type_error ({n_type}), 20% missing_arg ({n_miss}), 10% wrong_endpoint ({n_wrong})")

    status_counts: Dict[str, int] = {}

    for idx, r in enumerate(records[:num_pairs]):
        target_mutation = target_types[idx]
        prompt = r["prompt"]
        target_fn = r["target_function"]
        valid_args = r["target_arguments"]
        cand_tools = r["candidate_tools"]

        # 1. Chosen candidate (strictly passes sandbox validation with HTTP 200)
        chosen_call = {"name": target_fn, "arguments": valid_args}
        chosen_res = validator.validate_call(chosen_call, expected_func=target_fn)
        assert chosen_res.status == "passed", f"Chosen candidate failed for {target_fn}: {chosen_res.error_message}"

        # 2. Rejected candidate with designated mutation
        if target_mutation == "hallucinated_arg":
            rej_fn = target_fn
            rej_args, rej_res = mutate_hallucinated_arg(target_fn, valid_args, validator)
        elif target_mutation == "type_error":
            rej_fn = target_fn
            rej_args, rej_res = mutate_type_error(target_fn, valid_args, validator)
        elif target_mutation == "missing_arg":
            rej_fn = target_fn
            rej_args, rej_res = mutate_missing_arg(target_fn, valid_args, validator)
        else:  # wrong_endpoint
            rej_fn, rej_args, rej_res = mutate_wrong_endpoint(target_fn, valid_args, cand_tools, validator)

        assert rej_res.status != "passed", f"Rejected candidate passed for {rej_fn}!"
        actual_status = rej_res.status
        status_counts[actual_status] = status_counts.get(actual_status, 0) + 1

        rejected_call = {"name": rej_fn, "arguments": rej_args}

        dpo_pairs.append({
            "id": f"dpo_pair_{idx:04d}",
            "prompt": prompt,
            "chosen": json.dumps(chosen_call),
            "rejected": json.dumps(rejected_call),
            "chosen_status": "passed",
            "rejected_status": actual_status,
            "rejected_error_detail": rej_res.error_message,
        })

    # Save to data/train_dpo.jsonl
    out_file = DATA_DIR / "train_dpo.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"[+] Saved {len(dpo_pairs)} execution-verified DPO pairs to {out_file.name}")
    print(f"    Actual Rejected Breakdown: {status_counts}")
    return dpo_pairs


if __name__ == "__main__":
    mine_dpo_pairs(1000)
