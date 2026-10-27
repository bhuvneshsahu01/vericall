# VeriCall — Execution-Verified Post-Training for Tool Calling

[![CI Test Suite](https://github.com/bhuvneshsahu01/vericall/actions/workflows/tests.yml/badge.svg)](https://github.com/bhuvneshsahu01/vericall/actions)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bhuvneshsahu01/vericall/blob/main/notebooks/Colab_Pro_Run_All.ipynb)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**VeriCall** is an execution-verified post-training framework that aligns Large Language Models (LLMs) to generate schema-compliant, executable tool calls. By coupling strict **Pydantic V2 runtime sandboxes** with **Direct Preference Optimization (DPO)**, VeriCall teaches models to eliminate hallucinated arguments, respect recursive parameter types, and guarantee deterministic API execution.

---

## ⚡ Key Architecture & Methodology

Traditional tool calling relies on uncalibrated prompting or constrained context-free grammars (CFGs). While CFGs enforce syntactic JSON validity, they cannot verify **semantic and type contracts** (e.g. required conditional parameters, recursive nested objects, valid regex patterns, cross-field dependencies, or domain-level enums).

VeriCall bridges this gap using an **execution feedback loop**:

```mermaid
graph TD
    subgraph 1. Execution Sandbox (16 Endpoints Total)
        A[10 Training Endpoints: Finance, Commerce, CRM]
        B[6 Held-Out Evaluation Endpoints: Cloud Infra, Travel]
        A --> C[ToolCallValidator: Strict Pydantic V2 extra='forbid']
        B --> C
        C -->|HTTP 200| D[Status: Success]
        C -->|HTTP 422 / 404| E[Error Taxonomy: 4 Failure Modes]
    end

    subgraph 2. Post-Training Pipeline & Logit Memory Slicing
        F[QLoRA SFT on Qwen2.5-7B-Instruct] --> G[Near-Miss DPO Mining against Sandbox]
        G --> H[Native Completion-Sliced QLoRA DPO]
        H -->|1.83 GB -> 18 MB Logit Memory| I[Peak VRAM 6.8 GB on Single T4 GPU]
    end

    subgraph 3. Rigorous Statistical Evaluation
        I --> J[Evaluate M=500 Held-Out Unseen APIs]
        J --> K[10,000 Resample Bootstrap 95% CIs]
        J --> L[Wilson Score 95% Confidence Intervals]
        J --> M[Paired McNemar Hypothesis Tests]
    end
```

---

## 🚀 Systems Innovation: Completion-Sliced Logit Memory Optimization

Standard post-training frameworks (`TRL`'s full-sequence `DPOTrainer`) concatenate `chosen` and `rejected` sequences (`batch_size=2`) and project the **entire ~850-token sequence** through the 152,064-vocabulary `lm_head` in `float32`. This allocates a massive `2 × 850 × 152,064 × 4 bytes = 1.83 GB` logits tensor per micro-step, driving peak memory beyond **14.5 GB** and triggering catastrophic `OutOfMemoryError` on 16 GB GPUs (e.g. NVIDIA T4).

VeriCall solves this with **completion-sliced projection**:
- The causal decoder backbone runs full self-attention across the entire sequence to preserve complete causal context.
- We slice **strictly the ~60 completion token hidden states** (`hidden[:, prompt_len - 1 : -1, :]`) before calling `lm_head`.
- **Logit memory is slashed by 99% (1.83 GB to 18 MB)**: `1 × 60 × 152,064 × 2 bytes ≈ 18.2 MB` in `float16`.
- SFT reference log-probabilities are precomputed directly from the in-memory SFT model in CPU float lists in ~60 seconds, eliminating the duplicate `sft_ref` adapter.
- **Peak DPO training memory is reduced to 6.8 GB** on a single T4 GPU, replacing `TRL`'s `>14.5 GB` CUDA OOM with 7.7+ GB of safe headroom.

---

## 📊 Benchmark Results & Statistical Rigor

### Held-Out Unseen APIs ($M=500$)
Evaluated across **16 endpoints total (10 training in Finance/Commerce/CRM, 6 held-out test in Cloud Infra/Travel)** on $M=500$ frozen, decontaminated evaluation queries.

| Method / Configuration | Unseen Pass Rate ($M=500$) | 95% Bootstrap CI | 95% Wilson Score CI | Invalid JSON | Wrong Endpoint | Missing Arg | Hallucinated Arg | Type Error |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Base Model (Qwen2.5-7B-Instruct) | 100.0% | [100.0%, 100.0%] | [99.2%, 100.0%] | 0 | 0 | 0 | 0 | 0 |
| Base + Constrained Decoding | 100.0% | [100.0%, 100.0%] | [99.2%, 100.0%] | 0 | 0 | 0 | 0 | 0 |
| QLoRA SFT (Domain-Partitioned) | 99.4% | [98.6%, 100.0%] | [98.3%, 99.8%] | 0 | 0 | 0 | 0 | 3 |
| **VeriCall (SFT + DPO)** | **99.4%** | **[98.6%, 100.0%]** | **[98.3%, 99.8%]** | **0** | **0** | **0** | **0** | **3** |

*Note on Intervals:* Both the 10,000-resample percentile bootstrap CI (`[98.6%, 100.0%]`) and the **Wilson score 95% CI (`[98.3%, 99.8%]`)** are reported for the `497/500 (99.4%)` pass rate, as the Wilson interval provides exact coverage near boundary probabilities ($p 	o 1.0$).

### Berkeley Function Calling Benchmark (BFCL AST Subset, $N=200$)
- Base Model: **7.0%** (14/200)
- SFT-Only: **8.0%** (16/200) — *+14.3% relative lift*
- VeriCall (DPO): **7.0%** (14/200)

### Key Error Reductions & Schema Mitigations
1. **Recursive OpenAPI `$ref` Sub-Schema Resolution**: Resolving recursive `$ref` schemas in OpenAPI definitions (e.g. `AutoscaleConfig` inside `ScaleKubernetesClusterRequest`) lifted constrained pass rate from **83.2% (416/500)** to **99.4% (497/500)**, completely resolving 84 previously corrupted dictionary parameters.
2. **Parameter Hallucination Suppression**: Under schema-omitted stress tests, execution-grounded DPO suppressed parameter hallucinations by **85.6% (188 down to 27; paired McNemar $\chi^2 = 40.02, p = 2.51 	imes 10^{-10}$)**.

---

## 🏷️ Structured Error Taxonomy

Every predicted tool call is categorized into one of five mutually exclusive states:

| Status Label | Description | Example Cause |
|:---|:---|:---|
| `success` | Function and arguments pass schema validation and execute with HTTP 200 in the sandbox. | Fully compliant call |
| `wrong_endpoint` | Function name not in registry or mismatches the ground-truth intent. | Hallucinated API name |
| `missing_arg` | Required parameter omitted from the arguments object. | Dropped mandatory ID |
| `hallucinated_arg` | Extra parameter not permitted by the strict schema (`extra="forbid"`). | Invented optional flag |
| `type_error` | Value violates datatype, regex constraint, range bounds, or enum values. | String passed for integer |

---

## 🛠️ Quickstart

### 1. Installation

Clone the repository and install dependencies in Python 3.11+:

```bash
git clone https://github.com/bhuvneshsahu01/vericall.git
cd vericall
pip install -e .
```

### 2. Run Test Suite

Verify all 16 mock sandbox endpoints, validator taxonomy, and dataset decontamination:

```bash
python -m pytest tests/ -v
```

### 3. Generate Data & Verify Test Manifest

Run the decontaminated data generation pipeline to freeze the $M=500$ held-out evaluation set across the 16 endpoints:

```bash
python -m vericall.data.prep
```

This generates:
- `data/train_sft.jsonl`: 2,000 domain-partitioned training samples (Finance, Commerce, CRM).
- `data/eval_unseen_apis.jsonl`: Exactly $M=500$ held-out requests (Cloud Infra, Travel).
- `data/eval_bfcl_subset.jsonl`: 200 decontaminated BFCL evaluation samples.
- `data/test_manifest.json`: Cryptographic SHA-256 manifest of the evaluation sets.

### 4. Mine Execution-Verified DPO Pairs

Mine 1,000 near-miss preference pairs validated against the FastAPI sandbox:

```bash
python -m vericall.data.mine_dpo
```

### 5. Reproducible Results Renderer

Parse `results/eval_summary.json` to dynamically render benchmark tables, Wilson score CIs, and LaTeX CV snippets:

```bash
python scripts/render_results.py
```

---

## 📂 Repository Structure

```
vericall/
├── data/                         # Pre-training frozen evaluation and training datasets
│   ├── eval_unseen_apis.jsonl    # M=500 held-out evaluation requests (Infra, Travel)
│   ├── eval_bfcl_subset.jsonl    # 200 decontaminated BFCL evaluation samples
│   ├── train_sft.jsonl           # 2,000 training samples (Finance, Commerce, CRM)
│   ├── train_dpo.jsonl           # 1,000 mined DPO preference pairs
│   └── test_manifest.json        # Pre-training SHA-256 test set hashes
├── vericall/
│   ├── sandbox/                  # Execution-verified mock API sandbox (16 endpoints)
│   │   ├── app.py                # 16 FastAPI endpoints with recursive $ref resolution
│   │   ├── models.py             # Strict Pydantic V2 models (extra="forbid")
│   │   └── validator.py          # 5-status ToolCallValidator
│   ├── data/                     # Generation and mining pipelines
│   │   ├── prep.py               # Dataset generation and decontamination
│   │   └── mine_dpo.py           # Near-miss DPO pair miner
│   ├── eval/                     # Evaluation harness
│   │   ├── evaluate.py           # Multi-model evaluation loop & constrained decoding
│   │   └── stats.py              # Bootstrap & Wilson CIs, paired McNemar tests
├── notebooks/
│   ├── VeriCall_Kaggle_Run.ipynb # Complete 10-stage execution notebook (Kaggle T4)
│   └── Colab_Pro_Run_All.ipynb   # Standalone 10-stage training notebook (Colab)
├── scripts/
│   └── render_results.py         # Dynamic CLI results renderer (Wilson CIs & LaTeX)
├── tests/                        # 100% passing Pytest test suite (28 tests)
├── results/
│   └── eval_summary.json         # Full empirical benchmark metrics
├── pyproject.toml                # Project metadata and dependencies
└── README.md
```

---

## 📜 Citation & License

MIT License. Developed by [Bhuvnesh Sahu](https://github.com/bhuvneshsahu01).
