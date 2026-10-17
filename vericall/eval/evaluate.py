"""
vericall/eval/evaluate.py
-------------------------
Evaluation harness assessing 4 model configurations on frozen test sets:
(a) Base model
(b) Base model + Constrained JSON decoding
(c) SFT-only
(d) SFT + DPO (VeriCall)

Outputs metrics, bootstrap CIs, McNemar p-values, and error taxonomy to results/eval_summary.json.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from vericall.sandbox.validator import ToolCallValidator
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
    """
    validator = ToolCallValidator()
    total = len(test_records)
    assert len(predictions) == total, f"Expected {total} predictions, got {len(predictions)}"

    passed_count = 0
    scores = []
    error_taxonomy = {
        "wrong_endpoint": 0,
        "missing_arg": 0,
        "hallucinated_arg": 0,
        "type_error": 0,
    }

    eval_details = []

    for idx, (gt, pred) in enumerate(zip(test_records, predictions)):
        func_name = pred.get("name", "")
        args = pred.get("arguments", {})
        expected_fn = gt["target_function"]

        res = validator.validate_call(func_name, args, expected_func=expected_fn)
        is_pass = 1 if res.is_success else 0
        scores.append(is_pass)

        if res.is_success:
            passed_count += 1
        else:
            if res.status in error_taxonomy:
                error_taxonomy[res.status] += 1

        eval_details.append({
            "id": gt.get("id", f"sample_{idx}"),
            "query": gt["query"],
            "expected_func": expected_fn,
            "predicted_func": func_name,
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


def run_benchmark_eval(
    predictions_map: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluates all configurations across unseen APIs and BFCL.
    If predictions_map is None, loads existing or provides evaluation template.
    """
    unseen_path = DATA_DIR / "eval_unseen_apis.jsonl"
    bfcl_path = DATA_DIR / "eval_bfcl_subset.jsonl"

    if not unseen_path.exists() or not bfcl_path.exists():
        from vericall.data.prep import run_pipeline
        run_pipeline()

    unseen_records = [json.loads(line) for line in open(unseen_path, "r", encoding="utf-8") if line.strip()]
    bfcl_records = [json.loads(line) for line in open(bfcl_path, "r", encoding="utf-8") if line.strip()]

    print(f"[*] Loaded {len(unseen_records)} frozen unseen API eval requests.")
    print(f"[*] Loaded {len(bfcl_records)} frozen BFCL eval requests.")

    out_file = output_path or (RESULTS_DIR / "eval_summary.json")

    # If predictions are passed from live GPU run, evaluate them
    if predictions_map is not None:
        summary_results = {"configs": {}}
        for cfg_name, preds in predictions_map.items():
            res_unseen = evaluate_predictions(unseen_records, preds, cfg_name)
            summary_results["configs"][cfg_name] = res_unseen

        # Compute paired McNemar test p-values against base model
        if "base_model" in summary_results["configs"] and "sft_dpo_vericall" in summary_results["configs"]:
            base_scores = summary_results["configs"]["base_model"]["binary_scores"]
            dpo_scores = summary_results["configs"]["sft_dpo_vericall"]["binary_scores"]
            stat, p_val = compute_mcnemar_test(base_scores, dpo_scores)
            summary_results["mcnemar_vs_base"] = {
                "statistic": round(stat, 4),
                "p_value": float(p_val),
                "significant": bool(p_val < 0.05),
            }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(summary_results, f, indent=2)
        print(f"[+] Evaluated predictions saved to {out_file}")
        return summary_results

    return {"status": "ready_for_gpu_eval", "eval_unseen_count": len(unseen_records)}


if __name__ == "__main__":
    run_benchmark_eval()
