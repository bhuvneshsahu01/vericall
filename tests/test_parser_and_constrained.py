"""
tests/test_parser_and_constrained.py
------------------------------------
Unit tests verifying:
1. Robust JSON parsing and fence stripping in parse_json_call.
2. Auto-repair of truncated closing braces and quotes.
3. Parameter snapping, type coercion, and tool matching in apply_constrained_decoding.
4. Multi-tool 6-candidate schema integrity in BFCL AST dataset.
"""

import json
from pathlib import Path
import pytest

from vericall.sandbox.validator import parse_json_call, ExecutionValidator
from vericall.eval.evaluate import apply_constrained_decoding

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_parse_json_call_markdown_fences():
    raw = """```json
{
  "name": "provision_compute_node",
  "arguments": {
    "instance_tier": "m6i.4xlarge",
    "cloud_region": "ap-southeast-1",
    "cluster_count": 15,
    "environment_tag": "PRODUCTION",
    "resource_tags": {"owner": "platform"}
  }
}
```"""
    parsed = parse_json_call(raw)
    assert parsed is not None
    assert parsed["name"] == "provision_compute_node"
    assert parsed["arguments"]["cluster_count"] == 15


def test_parse_json_call_conversational_preamble_and_trailing():
    raw = """Sure! Here is the tool call you requested:
```json
{"name": "provision_compute_node", "arguments": {"instance_tier": "c6i.xlarge"}}
```
Let me know if you need more instances!"""
    parsed = parse_json_call(raw)
    assert parsed is not None
    assert parsed["name"] == "provision_compute_node"
    assert parsed["arguments"]["instance_tier"] == "c6i.xlarge"


def test_parse_json_call_truncated_repair():
    raw = '{"name": "scale_kubernetes_cluster", "arguments": {"node_pool": "worker-pool", "desired_replicas": 10'
    parsed = parse_json_call(raw)
    assert parsed is not None
    assert parsed["name"] == "scale_kubernetes_cluster"
    assert parsed["arguments"]["desired_replicas"] == 10


def test_parse_json_call_alias_normalization():
    raw = '{"function": "rotate_iam_credentials", "parameters": {"service_account_id": "sa-123456"}}'
    parsed = parse_json_call(raw)
    assert parsed is not None
    assert parsed["name"] == "rotate_iam_credentials"
    assert parsed["arguments"]["service_account_id"] == "sa-123456"


def test_apply_constrained_decoding_type_coercion_and_alias_snapping():
    candidate_tools = [{
        "name": "provision_compute_node",
        "description": "Spin up worker nodes",
        "parameters": {
            "type": "object",
            "properties": {
                "instance_tier": {"type": "string", "enum": ["c6i.xlarge", "m6i.4xlarge"]},
                "cluster_count": {"type": "integer"},
                "environment_tag": {"type": "string", "enum": ["DEVELOPMENT", "STAGING", "PRODUCTION"]},
                "force_start": {"type": "boolean"},
            },
            "required": ["instance_tier", "cluster_count", "environment_tag"],
        }
    }]

    raw_pred = """```json
{
  "name": "provision_compute_node",
  "arguments": {
    "instance_tier": "m6i.4xlarge",
    "count": "15",
    "environment_tag": "production",
    "force_start": "true",
    "hallucinated_extra_key": "drop_this"
  }
}
```"""
    cleaned = apply_constrained_decoding(raw_pred, candidate_tools)
    assert cleaned["name"] == "provision_compute_node"
    args = cleaned["arguments"]
    assert args["cluster_count"] == 15
    assert args["environment_tag"] == "PRODUCTION"
    assert args["force_start"] is True
    assert "hallucinated_extra_key" not in args


def test_apply_constrained_decoding_tool_selection_from_query():
    candidate_tools = [
        {"name": "get_current_weather", "description": "Fetch real-time weather observations for a city."},
        {"name": "search_multi_leg_flights", "description": "Query available commercial flight schedules and airfares."},
        {"name": "calculate", "description": "Perform basic arithmetic calculations."},
    ]

    raw_pred = '{"name": "unknown_func", "arguments": {"city": "London"}}'
    cleaned = apply_constrained_decoding(raw_pred, candidate_tools, user_query="What is the weather in London right now?")
    assert cleaned["name"] == "get_current_weather"


def test_bfcl_ast_subset_has_6_candidates_per_prompt():
    bfcl_path = REPO_ROOT / "data/eval_bfcl_ast_subset.jsonl"
    assert bfcl_path.exists()
    records = [json.loads(line) for line in open(bfcl_path, "r", encoding="utf-8") if line.strip()]
    assert len(records) == 200

    for r in records:
        assert len(r["candidate_tools"]) == 6, f"Sample {r['id']} does not have 6 candidate tools"
        assert r["target_function"] in [t["name"] for t in r["candidate_tools"]]
        assert "Available Tools:" in r["prompt"]
