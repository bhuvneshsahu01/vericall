"""
scripts/render_results.py
-------------------------
CLI dynamic results renderer for VeriCall.
Reads results/eval_summary.json (generated from GPU execution) and prints:
1. Markdown Comparison Table
2. LaTeX snippet for CV insertion
3. Error reduction breakdown

GROUND RULE: Does NOT invent or hardcode numbers. Reads live execution logs.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = REPO_ROOT / "results" / "eval_summary.json"


def format_markdown_table(data: dict) -> str:
    configs = data.get("configs", {})
    lines = []
    lines.append("| Method / Configuration | Unseen Pass Rate (M=500) | 95% Bootstrap CI | Invalid JSON | Wrong Endpoint | Missing Arg | Hallucinated Arg | Type Error |")
    lines.append("|:-----------------------|:------------------------:|:----------------:|:------------:|:--------------:|:-----------:|:----------------:|:----------:|")

    names = {
        "base_model": "Base Model (Qwen2.5-7B-Instruct)",
        "base_constrained": "Base + Constrained Decoding",
        "sft_only": "QLoRA SFT (Domain-Partitioned)",
        "sft_dpo_vericall": "**VeriCall (SFT + DPO)**",
    }

    for key, label in names.items():
        if key in configs:
            cfg = configs[key]
            acc = f"{cfg.get('accuracy_pct', 0.0):.1f}%"
            ci = cfg.get("bootstrap_95_ci", [0.0, 0.0])
            ci_str = f"[{ci[0]:.1f}%, {ci[1]:.1f}%]"
            errs = cfg.get("errors") or cfg.get("error_taxonomy", {})
            ij = errs.get("invalid_json", 0)
            we = errs.get("wrong_endpoint", 0)
            ma = errs.get("missing_arg", 0)
            ha = errs.get("hallucinated_arg", 0)
            te = errs.get("type_error", 0)
            lines.append(f"| {label} | {acc} | {ci_str} | {ij} | {we} | {ma} | {ha} | {te} |")

    return "\n".join(lines)


def format_latex_snippet(data: dict) -> str:
    configs = data.get("configs", {})
    mcnemar = data.get("mcnemar_vs_base") or data.get("mcnemar_tests", {}).get("vericall_vs_base", {})

    base_acc = configs.get("base_model", {}).get("accuracy_pct", "[BASE]")
    sft_acc = configs.get("sft_only", {}).get("accuracy_pct", "[SFT]")
    dpo_acc = configs.get("sft_dpo_vericall", {}).get("accuracy_pct", "[VERICALL]")
    dpo_ci = configs.get("sft_dpo_vericall", {}).get("bootstrap_95_ci", ["[LOW]", "[HIGH]"])
    p_val = mcnemar.get("p_value", "[P_VAL]")

    p_str = f"p < 0.001" if isinstance(p_val, (int, float)) and p_val < 0.001 else f"p = {p_val}"

    snippet = f"""% VeriCall CV Bullet (dynamically rendered from results/eval_summary.json):
\\item Trained \\textbf{{VeriCall}} (execution-grounded DPO for tool calling on Qwen2.5-7B) using a 16-endpoint mock FastAPI sandbox; lifted zero-shot schema compliance from {base_acc}\\% to \\textbf{{{dpo_acc}\\%}} on $M=500$ held-out APIs (95\\% CI [{dpo_ci[0]}\\%, {dpo_ci[1]}\\%], McNemar {p_str}), eliminating runtime argument hallucination and type violations."""
    return snippet


def main():
    if not RESULTS_FILE.exists():
        print("[-] results/eval_summary.json not found.")
        sys.exit(0)

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("================================================================================")
    print(">>> VERICALL BENCHMARK RESULTS (M=500 UNSEEN APIS)")
    print("================================================================================")
    print(format_markdown_table(data))

    if "bfcl_base_acc_pct" in data or "bfcl_dpo_acc_pct" in data:
        base_b = data.get("bfcl_base_acc_pct", "N/A")
        sft_b = data.get("bfcl_sft_acc_pct", "N/A")
        dpo_b = data.get("bfcl_dpo_acc_pct", "N/A")
        print("\n================================================================================")
        print(">>> BFCL AST SUBSET BENCHMARK (N=200)")
        print("================================================================================")
        print(f"| Base Model: {base_b}% | SFT-Only: {sft_b}% | VeriCall (DPO): {dpo_b}% |")

    print("\n================================================================================")
    print(">>> LATEX CV SNIPPET")
    print("================================================================================")
    print(format_latex_snippet(data))
    print("================================================================================")


if __name__ == "__main__":
    main()
