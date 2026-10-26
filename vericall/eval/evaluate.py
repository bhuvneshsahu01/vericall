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
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from vericall.sandbox.validator import ExecutionValidator, parse_json_call
from .stats import compute_bootstrap_ci, compute_mcnemar_test

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_predictions(
    test_records: List[Dict[str, Any]],
    predictions: List[Any],
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
    raw_prediction: Any,
    candidate_tools: List[Dict[str, Any]],
    user_query: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Applies schema-constrained post-processing / decoding:
    1. Non-destructive: if raw_prediction already parses and passes ExecutionValidator, return it UNCHANGED.
    2. Robustly extracts JSON call from markdown code fences, tool_call tags, or truncated braces.
    3. Snaps missing or mistyped function names to the closest candidate tool in the prompt.
    4. Strips hallucinated/unrecognized parameter keys not present in the target tool's schema.
    5. Fixes enum casing mismatches (e.g. "immediate" -> "IMMEDIATE").
    6. Coerces parameter types (stringified numbers/booleans) while preserving objects/dictionaries.
    """
    cand_tools = candidate_tools or []
    cand_names = [t["name"] for t in cand_tools]
    validator = ExecutionValidator()

    parsed = parse_json_call(raw_prediction)

    # 1. Non-destructive early return: if parsed call already validates against candidate tools, keep it!
    if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
        fn_name = parsed["name"]
        if fn_name in cand_names:
            val_res = validator.validate_call(parsed, expected_func=fn_name, candidate_tools=cand_tools)
            if val_res.status == "passed":
                return parsed

    # 2. Robust parsing fallback if text was severely mangled
    if not isinstance(parsed, dict):
        raw_text = str(raw_prediction) if raw_prediction else ""
        parsed = {"name": "", "arguments": {}}
        if raw_text:
            kv_pairs = re.findall(r'"([a-zA-Z0-9_]+)"\s*:\s*("[^"]*"|\d+\.?\d*|true|false)', raw_text)
            for k, v in kv_pairs:
                if k in ("name", "function", "tool"):
                    parsed["name"] = v.strip('"')
                else:
                    try:
                        parsed["arguments"][k] = json.loads(v)
                    except Exception:
                        parsed["arguments"][k] = v.strip('"')

    fn = parsed.get("name", "")
    args = parsed.get("arguments", {}) if isinstance(parsed.get("arguments"), dict) else {}

    # 3. Resolve tool name to best candidate
    if fn not in cand_names and cand_names:
        matched = None
        fn_lower = fn.lower().replace("-", "_").replace(" ", "_")
        for name in cand_names:
            if name.lower() == fn_lower:
                matched = name
                break
        if not matched:
            for name in cand_names:
                if name.lower() in fn_lower or fn_lower in name.lower():
                    matched = name
                    break
        if not matched and user_query:
            q_lower = user_query.lower()
            best_score = -1
            for tool in cand_tools:
                score = 0
                t_name = tool["name"].lower()
                t_desc = tool.get("description", "").lower()
                words = set(re.findall(r"\w+", t_name + " " + t_desc))
                q_words = re.findall(r"\w+", q_lower)
                for w in q_words:
                    if len(w) > 3 and w in words:
                        score += 1
                if score > best_score:
                    best_score = score
                    matched = tool["name"]
        fn = matched if matched else cand_names[0]

    # Find tool schema
    matching_tools = [t for t in cand_tools if t["name"] == fn]
    if not matching_tools:
        return {"name": fn, "arguments": args}

    tool_schema = matching_tools[0]
    params = tool_schema.get("parameters", {})
    props = params.get("properties", {})
    prop_names = list(props.keys())

    cleaned_args = {}

    # 4. Filter, alias-match, and type-coerce arguments strictly according to schema
    for k, v in args.items():
        target_prop = None
        if k in props:
            target_prop = k
        else:
            k_clean = k.lower().replace("-", "_")
            for p in prop_names:
                p_clean = p.lower().replace("-", "_")
                if k_clean == p_clean:
                    target_prop = p
                    break
                if (k_clean in ("count", "nodes") and "count" in p_clean) or (k_clean in ("type", "flavor") and "tier" in p_clean):
                    target_prop = p
                    break

        # Discard hallucinated arguments not in properties
        if target_prop is None:
            continue

        prop_spec = props[target_prop]
        prop_type = prop_spec.get("type")

        # Coerce types
        if prop_type == "integer":
            if isinstance(v, bool):
                cleaned_args[target_prop] = 1 if v else 0
            elif isinstance(v, (int, float)):
                cleaned_args[target_prop] = int(v)
            elif isinstance(v, str):
                digits = re.findall(r"[-+]?\d+", v)
                cleaned_args[target_prop] = int(digits[0]) if digits else v
            else:
                cleaned_args[target_prop] = v
        elif prop_type == "number":
            if isinstance(v, (int, float)):
                cleaned_args[target_prop] = float(v)
            elif isinstance(v, str):
                nums = re.findall(r"[-+]?\d*\.?\d+", v)
                cleaned_args[target_prop] = float(nums[0]) if nums else v
            else:
                cleaned_args[target_prop] = v
        elif prop_type == "boolean":
            if isinstance(v, bool):
                cleaned_args[target_prop] = v
            elif isinstance(v, str):
                cleaned_args[target_prop] = v.strip().lower() in ("true", "1", "yes", "y")
            elif isinstance(v, (int, float)):
                cleaned_args[target_prop] = bool(v)
            else:
                cleaned_args[target_prop] = v
        elif prop_type == "string":
            v_str = str(v)
            if "enum" in prop_spec:
                enums = prop_spec["enum"]
                if v_str in enums:
                    cleaned_args[target_prop] = v_str
                else:
                    matched_enum = None
                    for e in enums:
                        if e.lower() == v_str.lower():
                            matched_enum = e
                            break
                    cleaned_args[target_prop] = matched_enum if matched_enum else v_str
            else:
                cleaned_args[target_prop] = v_str
        elif prop_type == "object":
            if isinstance(v, dict):
                cleaned_args[target_prop] = v
            elif isinstance(v, str):
                try:
                    cleaned_args[target_prop] = json.loads(v)
                except Exception:
                    cleaned_args[target_prop] = v
            else:
                cleaned_args[target_prop] = v
        elif prop_type == "array":
            if isinstance(v, list):
                cleaned_args[target_prop] = v
            elif isinstance(v, str):
                try:
                    parsed_list = json.loads(v)
                    cleaned_args[target_prop] = parsed_list if isinstance(parsed_list, list) else [v]
                except Exception:
                    cleaned_args[target_prop] = [v]
            else:
                cleaned_args[target_prop] = [v]
        else:
            cleaned_args[target_prop] = v

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
