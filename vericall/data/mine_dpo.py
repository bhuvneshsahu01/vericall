"""
vericall/data/mine_dpo.py
-------------------------
Execution-Verified Near-Miss Preference Miner for DPO.
Samples candidate tool calls per prompt, verifies each against the FastAPI sandbox,
and pairs passed calls (chosen) against near-miss failed calls (rejected: type_error,
missing_arg, hallucinated_arg, wrong_endpoint).
"""

import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from vericall.sandbox.validator import ToolCallValidator

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"


def generate_verified_near_miss(validator: ToolCallValidator, func_name: str, valid_args: Dict[str, Any]) -> tuple[str, Dict[str, Any], Any]:
    """
    Synthesizes a realistic near-miss error (wrong_endpoint, type_error, missing_arg, or hallucinated_arg)
    and verifies that it strictly fails sandbox validation.
    """
    for attempt in range(20):
        mutation_type = random.choice(["wrong_endpoint", "type_error", "missing_arg", "hallucinated_arg"])
        cand_func = func_name
        cand_args = copy.deepcopy(valid_args)

        if mutation_type == "wrong_endpoint":
            # Pick a different endpoint from the registry
            other_endpoints = [ep for ep in validator.registry.keys() if ep != func_name]
            cand_func = random.choice(other_endpoints)

        elif mutation_type == "missing_arg":
            # Drop a key
            if cand_args:
                key_to_drop = random.choice(list(cand_args.keys()))
                del cand_args[key_to_drop]

        elif mutation_type == "hallucinated_arg":
            # Add an extra unauthorized parameter
            extra_keys = ["notes", "user_confirmation", "metadata", "api_key", "debug_mode", "timestamp", "priority"]
            cand_args[random.choice(extra_keys)] = True

        else:  # type_error
            if "quantity" in cand_args:
                cand_args["quantity"] = "three"  # string instead of int
            elif "order_type" in cand_args:
                cand_args["order_type"] = "INVALID_TYPE"
            elif "currency" in cand_args:
                cand_args["currency"] = "BITCOIN"
            elif "symbol" in cand_args:
                cand_args["symbol"] = "toolongticker"
            elif "category" in cand_args:
                cand_args["category"] = "unsupported_category"
            elif "in_stock_only" in cand_args:
                cand_args["in_stock_only"] = "maybe"
            else:
                first_k = list(cand_args.keys())[0] if cand_args else "dummy"
                cand_args[first_k] = {"nested_obj": 123}

        # Validate against sandbox
        res = validator.validate_call(cand_func, cand_args)
        if not res.is_success:
            return cand_func, cand_args, res

    # Fallback guaranteed failure
    fail_args = copy.deepcopy(valid_args)
    fail_args["__forbidden_hallucinated_param__"] = True
    res = validator.validate_call(func_name, fail_args)
    return func_name, fail_args, res


def mine_dpo_pairs(num_pairs: int = 1500) -> List[Dict[str, Any]]:
    """
    Constructs execution-verified DPO preference pairs (chosen vs rejected).
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

    validator = ToolCallValidator()
    dpo_pairs = []

    print(f"[*] Mining {min(num_pairs, len(records))} execution-verified DPO preference pairs...")
    for idx, r in enumerate(records[:num_pairs]):
        query = r["query"]
        target_fn = r["target_function"]
        valid_args = r["target_arguments"]

        # 1. Chosen candidate (strictly passes sandbox validation with HTTP 200)
        chosen_call = {"name": target_fn, "arguments": valid_args}
        chosen_res = validator.validate_call(target_fn, valid_args)
        assert chosen_res.status == "success", f"Chosen candidate failed validation: {chosen_res}"

        # 2. Rejected near-miss candidate (strictly fails sandbox validation)
        rej_fn, rej_args, rejected_res = generate_verified_near_miss(validator, target_fn, valid_args)
        rejected_call = {"name": rej_fn, "arguments": rej_args}
        assert rejected_res.status != "success", f"Rejected candidate passed validation: {rejected_res}"

        dpo_pairs.append({
            "id": f"dpo_pair_{idx:04d}",
            "prompt": query,
            "chosen": json.dumps(chosen_call),
            "rejected": json.dumps(rejected_call),
            "chosen_status": chosen_res.status,
            "rejected_status": rejected_res.status,
            "rejected_error_detail": rejected_res.error_message,
        })

    # Save to data/train_dpo.jsonl
    out_file = DATA_DIR / "train_dpo.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"[+] Saved {len(dpo_pairs)} execution-verified DPO pairs to {out_file.name}")
    status_counts = {}
    for p in dpo_pairs:
        s = p["rejected_status"]
        status_counts[s] = status_counts.get(s, 0) + 1
    print(f"    Rejected failure breakdown: {status_counts}")
    return dpo_pairs


if __name__ == "__main__":
    mine_dpo_pairs(1000)
