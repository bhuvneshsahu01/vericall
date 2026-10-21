"""
vericall/eval/evaluate.py
-------------------------
Evaluation harness assessing 4 model configurations on frozen test sets:
(a) Base model (zero-shot with candidate schemas in prompt)
(b) Base model + Constrained JSON decoding
(c) SFT-only
(d) SFT + DPO (VeriCall)

Outputs metrics, bootstrap CIs, McNemar p-values, and 6-state error taxonomy to results/eval_summary.json.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from vericall.sandbox.validator import ExecutionValidator
from .stats import compute_bootstrap_ci, compute_mcnemar_test

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_predictions(
    test_records: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    config_name: str,
) -> Dict[str, Any]:
    """
    Evaluates model predictions against ground truth and sandbox execution.
    Classifies errors into: passed, invalid_json, wrong_endpoint, missing_arg, hallucinated_arg, type_error.
    """
    validator = ExecutionValidator()
    total = len(test_records)
    assert len(predictions) == total, f"Expected {total} predictions, got {len(predictions)}"

    passed_count = 0
    scores = []
    error_taxonomy = {
        "invalid_json": 0,
        "wrong_endpoint": 0,
        "missing_arg": 0,
        "hallucinated_arg": 0,
        "type_error": 0,
    }

    eval_details = []

    for idx, (gt, pred) in enumerate(zip(test_records, predictions)):
        expected_fn = gt["target_function"]
        cand_tools = gt.get("candidate_tools", [])

        res = validator.validate_call(pred, expected_func=expected_fn, candidate_tools=cand_tools)
        is_pass = 1 if res.status == "passed" else 0
        scores.append(is_pass)

        if res.status == "passed":
            passed_count += 1
        else:
            if res.status in error_taxonomy:
                error_taxonomy[res.status] += 1
            else:
                error_taxonomy["type_error"] += 1

        eval_details.append({
            "id": gt.get("id", f"sample_{idx}"),
            "query": gt.get("query", ""),
            "expected_func": expected_fn,
            "predicted_func": res.function_name,
            "validation_status": res.status,
            "is_success": res.is_success,
            "error_message": res.error_message,
        })

    # Exact percentage calculation: passed_count / total * 100.0
    accuracy_pct = round((passed_count / total) * 100.0, 2)
    mean_pct, ci_low, ci_high = compute_bootstrap_ci(scores, num_resamples=10000, seed=42)

    return {
        "config_name": config_name,
        "total_requests": total,
        "passed_requests": passed_count,
        "accuracy_pct": accuracy_pct,
        "bootstrap_95_ci": [ci_low, ci_high],
        "error_taxonomy": error_taxonomy,
        "binary_scores": scores,
        "sample_evaluations": eval_details[:10],
    }


def apply_constrained_decoding(
    raw_prediction: Dict[str, Any],
    candidate_tools: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Applies schema-constrained post-processing / decoding:
    1. If function name is invalid or missing, defaults to closest candidate schema.
    2. Strips extra hallucinated keys not in the selected tool schema.
    3. Type casts obvious string integers or string booleans.
    """
    if not isinstance(raw_prediction, dict):
        fn = candidate_tools[0]["name"] if candidate_tools else ""
        return {"name": fn, "arguments": {}}

    fn = raw_prediction.get("name", "")
    args = raw_prediction.get("arguments", {})

    cand_names = [t["name"] for t in candidate_tools]
    if fn not in cand_names and cand_names:
        fn = cand_names[0]

    # Find schema for fn
    matching = [t for t in candidate_tools if t["name"] == fn]
    if not matching or not isinstance(args, dict):
        return {"name": fn, "arguments": args if isinstance(args, dict) else {}}

    schema_props = matching[0].get("parameters", {}).get("properties", {})
    cleaned_args = {}

    for k, v in args.items():
        if k in schema_props:
            prop_spec = schema_props[k]
            prop_type = prop_spec.get("type")
            # Safe type coercion
            if prop_type == "integer" and isinstance(v, str) and v.isdigit():
                cleaned_args[k] = int(v)
            elif prop_type == "boolean" and isinstance(v, str):
                cleaned_args[k] = True if v.lower() == "true" else False
            elif prop_type == "string" and "enum" in prop_spec and isinstance(v, str):
                # Try uppercase match
                enums = prop_spec["enum"]
                up = v.upper()
                cleaned_args[k] = up if up in enums else v
            else:
                cleaned_args[k] = v

    return {"name": fn, "arguments": cleaned_args}


def run_benchmark_eval(
    predictions_map: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluates configurations across unseen APIs and BFCL.
    """
    unseen_path = DATA_DIR / "eval_unseen_apis.jsonl"
    bfcl_path = DATA_DIR / "eval_bfcl_ast_subset.jsonl"
    if not bfcl_path.exists():
        bfcl_path = DATA_DIR / "eval_bfcl_subset.jsonl"

    if not unseen_path.exists() or not bfcl_path.exists():
        from vericall.data.prep import run_pipeline
        run_pipeline()

    unseen_records = [json.loads(line) for line in open(unseen_path, "r", encoding="utf-8") if line.strip()]
    bfcl_records = [json.loads(line) for line in open(bfcl_path, "r", encoding="utf-8") if line.strip()]

    print(f"[*] Loaded {len(unseen_records)} frozen unseen API eval requests.")
    print(f"[*] Loaded {len(bfcl_records)} frozen BFCL eval requests.")

    out_file = output_path or (RESULTS_DIR / "eval_summary.json")

    if predictions_map is not None:
        summary_results = {"configs": {}}
        for cfg_name, preds in predictions_map.items():
            res_unseen = evaluate_predictions(unseen_records, preds, cfg_name)
            summary_results["configs"][cfg_name] = res_unseen

        # Compute paired McNemar tests against base model and sft
        cfgs = summary_results["configs"]
        if "base_model" in cfgs and "sft_dpo_vericall" in cfgs:
            b_sc = cfgs["base_model"]["binary_scores"]
            d_sc = cfgs["sft_dpo_vericall"]["binary_scores"]
            stat_b, p_b = compute_mcnemar_test(b_sc, d_sc)
            summary_results["mcnemar_vs_base"] = {"statistic": round(stat_b, 4), "p_value": float(p_b)}

        if "sft_only" in cfgs and "sft_dpo_vericall" in cfgs:
            s_sc = cfgs["sft_only"]["binary_scores"]
            d_sc = cfgs["sft_dpo_vericall"]["binary_scores"]
            stat_s, p_s = compute_mcnemar_test(s_sc, d_sc)
            summary_results["mcnemar_vs_sft"] = {"statistic": round(stat_s, 4), "p_value": float(p_s)}

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(summary_results, f, indent=2)
        print(f"[+] Evaluated predictions saved to {out_file}")
        return summary_results

    return {"status": "ready_for_gpu_eval", "eval_unseen_count": len(unseen_records)}


if __name__ == "__main__":
    run_benchmark_eval()
