"""
vericall/data/prep.py
---------------------
Dataset generation and strict domain-split decontamination.
Generates:
- data/eval_unseen_apis.jsonl: Exactly M=500 requests across 6 unseen endpoints (infra, travel).
  Every prompt contains all 6 unseen tool schemas (1 target + 5 decoys, shuffled).
- data/train_sft.jsonl: Exactly N=2,000 requests across 10 training endpoints (finance, commerce, crm).
  Every prompt contains 6 candidate schemas (1 target + 5 decoys sampled from train endpoints).
- data/eval_bfcl_subset.jsonl: 200 decontaminated general function calling requests.
- data/test_manifest.json: SHA-256 hashes of frozen evaluation and training sets.
"""

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

from vericall.sandbox.app import ENDPOINT_REGISTRY, get_tool_json_schema
from vericall.sandbox.validator import ExecutionValidator

SEED = 42
random.seed(SEED)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Endpoint classifications
TRAIN_ENDPOINTS = [
    "issue_refund",
    "create_invoice",
    "transfer_funds",
    "audit_ledger_entries",
    "query_inventory",
    "apply_discount_code",
    "update_subscription",
    "create_support_ticket",
    "update_customer_tier",
    "schedule_webhook",
]

UNSEEN_ENDPOINTS = [
    "provision_compute_node",
    "scale_kubernetes_cluster",
    "rotate_iam_credentials",
    "search_multi_leg_flights",
    "reserve_corporate_hotel",
    "cancel_itinerary_segment",
]


def build_tool_calling_prompt(user_query: str, candidate_schemas: List[Dict[str, Any]]) -> str:
    """
    Standard tool-calling prompt injection format.
    Every prompt contains 6 JSON tool schemas in Available Tools.
    """
    tools_block = "\n".join(json.dumps(s, separators=(",", ":")) for s in candidate_schemas)
    system_msg = (
        "You are a precise REST API tool-calling assistant. Given the user request, select "
        "the tool from `Available Tools` below and output ONLY a valid JSON object with keys "
        "\"name\" and \"arguments\".\n\n"
        f"Available Tools:\n{tools_block}\n\n"
        "Output format:\n"
        '{\"name\": \"<tool_name>\", \"arguments\": {<exact_schema_parameters>}}'
    )
    return (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def sample_candidate_tools(target_tool: str, domain_pool: List[str], total_count: int = 6, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Samples candidate tools (target + decoys) and shuffles deterministically.
    """
    rng = random.Random(seed)
    decoys = [ep for ep in domain_pool if ep != target_tool]
    sampled_decoys = rng.sample(decoys, min(total_count - 1, len(decoys)))
    selected_tools = [target_tool] + sampled_decoys
    rng.shuffle(selected_tools)
    return [get_tool_json_schema(name) for name in selected_tools]


# ============================================================================
# TEMPLATE GENERATORS FOR 10 TRAINING APIS
# ============================================================================

def generate_finance_samples(count_per_endpoint: int = 200) -> List[Dict[str, Any]]:
    samples = []
    
    # 1. issue_refund
    reasons = ["CUSTOMER_RETURN", "DUPLICATE_CHARGE", "FRAUDULENT", "SERVICE_OUTAGE"]
    currencies = ["USD", "EUR", "GBP", "INR"]
    for i in range(count_per_endpoint):
        tx_id = f"txn_{random.randint(10000000, 99999999)}"
        amt = round(random.uniform(15.0, 4500.0), 2)
        curr = random.choice(currencies)
        rsn = random.choice(reasons)
        agent = f"agent_{random.randint(10, 99)}"
        args = {
            "transaction_id": tx_id,
            "amount": amt,
            "currency": curr,
            "reason": rsn,
            "metadata": {"approved_by": agent, "dept": "finance"},
        }
        query = f"Issue a full refund of {amt} {curr} for transaction {tx_id} due to {rsn.replace('_', ' ').lower()}. Verified by {agent}."
        samples.append(("issue_refund", query, args))

    # 2. create_invoice
    for i in range(count_per_endpoint):
        cli_id = f"cli_{random.randint(1000, 99999)}"
        curr = random.choice(currencies)
        day_issue = random.randint(1, 15)
        day_due = day_issue + 14
        iss_date = f"2026-11-{day_issue:02d}"
        due_date = f"2026-11-{day_due:02d}"
        items = [
            {"description": "Enterprise Platform License", "unit_price": float(random.randint(500, 3000)), "quantity": 1},
            {"description": "Technical Support Hours", "unit_price": 150.0, "quantity": random.randint(2, 10)},
        ]
        args = {
            "client_id": cli_id,
            "issue_date": iss_date,
            "due_date": due_date,
            "line_items": items,
            "currency": curr,
        }
        query = f"Create an invoice for client {cli_id} issued on {iss_date} and due on {due_date} in {curr} with {len(items)} line items."
        samples.append(("create_invoice", query, args))

    # 3. transfer_funds
    networks = ["ACH", "WIRE", "SWIFT", "SEPA"]
    for i in range(count_per_endpoint):
        src = f"acc_{random.randint(100000, 999999)}"
        dst = f"acc_{random.randint(100000, 999999)}"
        amt = float(random.randint(100, 25000))
        net = random.choice(networks)
        memo = f"Vendor payout settlement ref {random.randint(100, 999)}"
        args = {
            "source_account": src,
            "destination_account": dst,
            "amount": amt,
            "transfer_method": net,
            "memo": memo,
        }
        query = f"Transfer {amt} from {src} to {dst} via {net} with memo '{memo}'."
        samples.append(("transfer_funds", query, args))

    # 4. audit_ledger_entries
    entry_types = ["CREDIT", "DEBIT", "ADJUSTMENT", "TRANSFER"]
    for i in range(count_per_endpoint):
        ledg = f"ledg_{random.randint(100000, 999999)}"
        s_date = f"2026-10-01"
        e_date = f"2026-10-31"
        etype = random.choice(entry_types)
        args = {
            "ledger_id": ledg,
            "start_date": s_date,
            "end_date": e_date,
            "entry_type": etype,
            "filter_flags": {"reconciled_only": True, "flagged": False},
        }
        query = f"Audit {etype} ledger entries for partition {ledg} from {s_date} to {e_date} with reconciled_only set to True."
        samples.append(("audit_ledger_entries", query, args))

    return samples


def generate_commerce_samples(count_per_endpoint: int = 200) -> List[Dict[str, Any]]:
    samples = []
    
    # 5. query_inventory
    channels = ["IN_STORE", "WAREHOUSE_DIRECT", "DROPSHIP"]
    for i in range(count_per_endpoint):
        skus = [f"SKU-{random.choice(['PROD', 'PART', 'TOOL'])}{random.randint(10, 99)}" for _ in range(random.randint(1, 3))]
        wh = f"wh_{random.choice(['us', 'eu', 'ap'])}-{random.randint(10, 99)}"
        chn = random.choice(channels)
        args = {
            "sku_list": skus,
            "warehouse_id": wh,
            "include_out_of_stock": False,
            "fulfillment_channel": chn,
        }
        query = f"Check in-stock inventory levels for items {', '.join(skus)} at facility {wh} via {chn} fulfillment."
        samples.append(("query_inventory", query, args))

    # 6. apply_discount_code
    pos_channels = ["ONLINE_WEB", "MOBILE_APP", "POS_REGISTER"]
    for i in range(count_per_endpoint):
        cid = f"cart_{random.randint(100000, 999999)}"
        promo = f"SAVE_{random.choice(['FALL', 'WINTER', 'VIP'])}{random.randint(10, 50)}"
        chn = random.choice(pos_channels)
        pin = random.randint(1000, 9999)
        args = {
            "cart_id": cid,
            "promo_code": promo,
            "channel": chn,
            "customer_loyalty_pin": pin,
        }
        query = f"Apply discount promo code '{promo}' to shopping cart {cid} from channel {chn} with loyalty PIN {pin}."
        samples.append(("apply_discount_code", query, args))

    # 7. update_subscription
    plans = ["BASIC_MONTHLY", "PRO_MONTHLY", "ENTERPRISE_ANNUAL"]
    for i in range(count_per_endpoint):
        sid = f"sub_{random.randint(10000000, 99999999)}"
        plan = random.choice(plans)
        eff = f"2026-12-01"
        args = {
            "subscription_id": sid,
            "plan_tier": plan,
            "auto_renew": True,
            "addon_features": ["audit_logs", "priority_support"],
            "effective_date": eff,
        }
        query = f"Update subscription {sid} to plan tier {plan} with auto-renew enabled and effective date {eff}."
        samples.append(("update_subscription", query, args))

    return samples


def generate_crm_samples(count_per_endpoint: int = 200) -> List[Dict[str, Any]]:
    samples = []
    
    # 8. create_support_ticket
    priorities = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    depts = ["BILLING", "TECHNICAL", "ACCOUNT", "SHIPPING"]
    for i in range(count_per_endpoint):
        cid = f"cust_{random.randint(10000, 99999)}"
        prio = random.choice(priorities)
        dept = random.choice(depts)
        subj = f"Issue regarding recent service order #{random.randint(1000, 9999)}"
        args = {
            "customer_id": cid,
            "priority": prio,
            "department": dept,
            "subject": subj,
            "ticket_context": {"origin": "web_portal", "browser": "chrome"},
        }
        query = f"Open a {prio} priority {dept} support ticket for customer {cid} regarding '{subj}'."
        samples.append(("create_support_ticket", query, args))

    # 9. update_customer_tier
    tiers = ["BRONZE", "SILVER", "GOLD", "PLATINUM"]
    for i in range(count_per_endpoint):
        cid = f"cust_{random.randint(10000, 99999)}"
        tier = random.choice(tiers)
        rep = f"rep_{random.randint(1000, 9999)}"
        reason = f"Annual spending threshold exceeded tier criteria #{random.randint(100, 999)}"
        args = {
            "customer_id": cid,
            "target_tier": tier,
            "assigned_rep_id": rep,
            "override_reason": reason,
            "benefits": ["waived_fees", "dedicated_rep"],
        }
        query = f"Upgrade customer {cid} to tier {tier} by account rep {rep}. Justification: {reason}."
        samples.append(("update_customer_tier", query, args))

    # 10. schedule_webhook
    events = ["invoice.paid", "order.created", "ticket.resolved", "refund.processed"]
    for i in range(count_per_endpoint):
        dest = f"https://api.partner{random.randint(10, 99)}.com/webhooks/v1"
        sub_events = random.sample(events, random.randint(1, 3))
        secret = f"whsec_{random.randint(10000000, 99999999)}"
        args = {
            "destination_url": dest,
            "subscribed_events": sub_events,
            "retry_count": 3,
            "signing_secret": secret,
            "headers": {"X-Client-ID": "partner_app", "Content-Type": "application/json"},
        }
        query = f"Register webhook destination {dest} subscribed to events {sub_events} with signing secret {secret}."
        samples.append(("schedule_webhook", query, args))

    return samples


def generate_training_data(total_samples: int = 2000) -> List[Dict[str, Any]]:
    """Generates 2,000 SFT training records across the 10 training endpoints."""
    raw = []
    raw.extend(generate_finance_samples(200))
    raw.extend(generate_commerce_samples(200))
    raw.extend(generate_crm_samples(200))

    assert len(raw) == total_samples, f"Expected {total_samples}, got {len(raw)}"
    validator = ExecutionValidator()
    train_records = []

    for idx, (fn, query, args) in enumerate(raw):
        # Verify ground truth strictly passes sandbox validation
        res = validator.validate_call({"name": fn, "arguments": args}, expected_func=fn)
        assert res.status == "passed", f"Ground truth failed for {fn}: {res.error_message}"

        # Sample candidate tools: target + 5 decoys from the other 9 training endpoints
        candidate_tools = sample_candidate_tools(fn, TRAIN_ENDPOINTS, total_count=6, seed=idx + 100)
        prompt = build_tool_calling_prompt(query, candidate_tools)
        response_str = json.dumps({"name": fn, "arguments": args})

        train_records.append({
            "id": f"train_sft_{idx:04d}",
            "domain": ENDPOINT_REGISTRY[fn]["domain"],
            "query": query,
            "target_function": fn,
            "target_arguments": args,
            "candidate_tools": candidate_tools,
            "prompt": prompt,
            "response": response_str,
            "text": f"{prompt}{response_str}<|im_end|>",
        })

    return train_records


# ============================================================================
# TEMPLATE GENERATORS FOR 6 UNSEEN TEST APIS (Infra & Travel)
# ============================================================================

def generate_unseen_eval_data(total_samples: int = 500) -> List[Dict[str, Any]]:
    """
    Generates M=500 requests across the 6 unseen Infrastructure & Travel endpoints.
    Every single prompt includes all 6 unseen endpoints (shuffled).
    """
    validator = ExecutionValidator()
    raw_eval = []

    instance_tiers = ["c6i.xlarge", "c6i.2xlarge", "g5.xlarge", "m6i.4xlarge"]
    regions = ["us-east-1", "us-west-2", "eu-central-1", "ap-southeast-1"]
    envs = ["DEVELOPMENT", "STAGING", "PRODUCTION"]
    
    strategies = ["IMMEDIATE", "DRAIN_GRADUAL", "SURGE_FIRST"]
    cred_types = ["OAUTH2_KEY", "SSH_RSA", "X509_CERTIFICATE", "API_TOKEN"]
    cabin_classes = ["ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST"]
    room_categories = ["STANDARD_KING", "EXECUTIVE_SUITE", "DOUBLE_QUEEN", "PRESIDENTIAL"]
    cancel_reasons = ["CLIENT_RESCHEDULE", "WEATHER_DISRUPTION", "MEETING_CANCELLED", "DUPLICATE_BOOKING"]
    airports = ["JFK", "LHR", "HND", "SFO", "ORD", "FRA", "SIN", "CDG"]

    count_per_ep = total_samples // len(UNSEEN_ENDPOINTS)  # 83 each
    remainder = total_samples % len(UNSEEN_ENDPOINTS)       # 2 left

    for ep_idx, fn in enumerate(UNSEEN_ENDPOINTS):
        n_items = count_per_ep + (1 if ep_idx < remainder else 0)
        for i in range(n_items):
            if fn == "provision_compute_node":
                tier = random.choice(instance_tiers)
                reg = random.choice(regions)
                cnt = random.randint(1, 20)
                env = random.choice(envs)
                tags = {"owner": f"team_{random.choice(['infra', 'core', 'platform'])}", "cost_center": f"cc_{random.randint(100, 999)}"}
                args = {
                    "instance_tier": tier,
                    "cloud_region": reg,
                    "cluster_count": cnt,
                    "environment_tag": env,
                    "resource_tags": tags,
                }
                query = f"Provision {cnt} compute nodes of tier {tier} in region {reg} for {env} environment. Tag owner as {tags['owner']}."

            elif fn == "scale_kubernetes_cluster":
                cl_name = f"k8s-prod-{random.randint(100, 999)}"
                pool = random.choice(["worker-pool", "gpu-pool", "compute-nodes"])
                rep = random.randint(3, 40)
                strat = random.choice(strategies)
                args = {
                    "cluster_name": cl_name,
                    "node_pool": pool,
                    "desired_replicas": rep,
                    "scaling_strategy": strat,
                    "autoscale_config": {"min_nodes": 2, "max_nodes": 60},
                }
                query = f"Scale node pool '{pool}' in cluster {cl_name} to {rep} desired replicas using {strat} strategy with min_nodes=2 and max_nodes=60."

            elif fn == "rotate_iam_credentials":
                sa = f"sa-{random.choice(['data', 'api', 'etl'])}{random.randint(1000, 9999)}"
                ctype = random.choice(cred_types)
                grace = random.choice([24, 48, 72])
                args = {
                    "service_account_id": sa,
                    "credential_type": ctype,
                    "grace_period_hours": grace,
                    "notify_emails": [f"security-alerts@company{random.randint(1, 9)}.com"],
                    "force_revocation": False,
                }
                query = f"Rotate {ctype} credentials for service account {sa} with a {grace}-hour grace period and alert security team."

            elif fn == "search_multi_leg_flights":
                orig, mid, dest = random.sample(airports, 3)
                cab = random.choice(cabin_classes)
                pax = random.randint(1, 4)
                d1 = f"2026-11-{random.randint(10, 15):02d}"
                d2 = f"2026-11-{random.randint(18, 24):02d}"
                args = {
                    "itinerary_segments": [
                        {"departure_airport": orig, "arrival_airport": mid, "flight_date": d1},
                        {"departure_airport": mid, "arrival_airport": dest, "flight_date": d2},
                    ],
                    "cabin_class": cab,
                    "passenger_count": pax,
                    "flexible_dates": True,
                    "preferred_alliances": ["STAR_ALLIANCE"],
                }
                query = f"Search multi-leg flights: {orig} to {mid} on {d1}, then {mid} to {dest} on {d2} in {cab} class for {pax} passengers."

            elif fn == "reserve_corporate_hotel":
                prop = f"HTL-{random.choice(['NY', 'SF', 'LN', 'TK'])}-{random.randint(1000, 9999)}"
                cin = f"2026-12-{random.randint(5, 12):02d}"
                cout = f"2026-12-{random.randint(15, 20):02d}"
                room = random.choice(room_categories)
                corp = f"CORP-{random.randint(10000, 99999)}"
                manifest = [{"first_name": "Alex", "last_name": f"Smith_{i}"}]
                args = {
                    "property_code": prop,
                    "check_in_date": cin,
                    "check_out_date": cout,
                    "room_category": room,
                    "corporate_billing_code": corp,
                    "guest_manifest": manifest,
                }
                query = f"Reserve a {room} room at {prop} checking in {cin} and checking out {cout} for {manifest[0]['first_name']} {manifest[0]['last_name']}. Bill to {corp}."

            else:  # cancel_itinerary_segment
                pnr = f"PNR-{random.randint(100000, 999999)}"
                seg = f"seg_{random.randint(10, 99)}"
                rsn = random.choice(cancel_reasons)
                notes = f"Cancellation authorized per travel policy exemption #{random.randint(100, 999)}"
                args = {
                    "booking_reference": pnr,
                    "segment_id": seg,
                    "cancellation_reason": rsn,
                    "request_credit_voucher": True,
                    "refund_override_notes": notes,
                }
                query = f"Cancel segment {seg} on booking {pnr} due to {rsn.replace('_', ' ').lower()}. Request credit voucher. Notes: {notes}."

            raw_eval.append((fn, query, args))

    assert len(raw_eval) == total_samples, f"Expected exactly {total_samples}, got {len(raw_eval)}"
    eval_records = []

    for idx, (fn, query, args) in enumerate(raw_eval):
        # 1. Verify ground-truth call strictly passes sandbox validation
        res = validator.validate_call({"name": fn, "arguments": args}, expected_func=fn)
        assert res.status == "passed", f"Ground truth eval call failed for {fn}: {res.error_message}"

        # 2. Candidate tools: ALL 6 unseen endpoints (shuffled deterministically)
        candidate_tools = sample_candidate_tools(fn, UNSEEN_ENDPOINTS, total_count=6, seed=idx + 500)
        prompt = build_tool_calling_prompt(query, candidate_tools)

        eval_records.append({
            "id": f"eval_unseen_{idx:03d}",
            "domain": ENDPOINT_REGISTRY[fn]["domain"],
            "query": query,
            "target_function": fn,
            "target_arguments": args,
            "candidate_tools": candidate_tools,
            "prompt": prompt,
        })

    return eval_records


def generate_bfcl_subset(num_samples: int = 200) -> List[Dict[str, Any]]:
    """
    Generates N=200 decontaminated BFCL evaluation samples.
    Every prompt contains 6 candidate tool schemas (1 target + 5 decoys shuffled)
    requiring strict parameter type, enum, and required-field adherence.
    """
    random.seed(SEED + 99)

    tool_pool = {
        "calculate": {
            "name": "calculate",
            "description": "Perform basic arithmetic calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["add", "multiply", "subtract", "divide"]},
                    "a": {"type": "number", "description": "First operand"},
                    "b": {"type": "number", "description": "Second operand"},
                },
                "required": ["operation", "a", "b"],
            },
        },
        "get_current_weather": {
            "name": "get_current_weather",
            "description": "Fetch real-time weather observations for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and state/country"},
                    "unit": {"type": "string", "enum": ["metric", "imperial"]},
                },
                "required": ["location", "unit"],
            },
        },
        "create_calendar_event": {
            "name": "create_calendar_event",
            "description": "Add an appointment or sync event to user calendar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Event title"},
                    "date": {"type": "string", "pattern": "^\d{4}-\d{2}-\d{2}$", "description": "Date (YYYY-MM-DD)"},
                    "time": {"type": "string", "pattern": "^\d{2}:\d{2}$", "description": "Time (HH:MM)"},
                    "duration_minutes": {"type": "integer", "description": "Duration in minutes"},
                },
                "required": ["title", "date", "time", "duration_minutes"],
            },
        },
        "convert_units": {
            "name": "convert_units",
            "description": "Convert physical quantities between measurement units.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "Numerical magnitude"},
                    "from_unit": {"type": "string", "enum": ["km", "miles", "celsius", "fahrenheit", "kg", "pounds"]},
                    "to_unit": {"type": "string", "enum": ["km", "miles", "celsius", "fahrenheit", "kg", "pounds"]},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
        "send_email": {
            "name": "send_email",
            "description": "Send an electronic mail message to a recipient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string", "description": "Destination email address"},
                    "subject": {"type": "string", "description": "Subject header"},
                    "body": {"type": "string", "description": "Email body content"},
                    "urgent": {"type": "boolean", "description": "Urgent delivery flag"},
                },
                "required": ["recipient", "subject", "body", "urgent"],
            },
        },
        "search_database": {
            "name": "search_database",
            "description": "Execute structured query against indexed database records.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string", "enum": ["customers", "orders", "inventory", "transactions"]},
                    "filter_query": {"type": "string", "description": "Search filter string"},
                    "limit": {"type": "integer", "description": "Max rows to return"},
                },
                "required": ["table_name", "filter_query", "limit"],
            },
        },
        "set_device_alarm": {
            "name": "set_device_alarm",
            "description": "Set a wakeup or reminder alarm on smart device.",
            "parameters": {
                "type": "object",
                "properties": {
                    "alarm_time": {"type": "string", "pattern": "^\d{2}:\d{2}$", "description": "Alarm time (HH:MM)"},
                    "label": {"type": "string", "description": "Alarm label"},
                    "repeat_daily": {"type": "boolean", "description": "Repeat alarm daily"},
                },
                "required": ["alarm_time", "label", "repeat_daily"],
            },
        },
        "generate_image": {
            "name": "generate_image",
            "description": "Synthesize a visual image from descriptive text prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Visual scene description"},
                    "resolution": {"type": "string", "enum": ["512x512", "1024x1024", "1920x1080"]},
                    "aspect_ratio": {"type": "string", "enum": ["1:1", "16:9", "4:3"]},
                },
                "required": ["prompt", "resolution", "aspect_ratio"],
            },
        },
    }

    all_tool_names = list(tool_pool.keys())
    cities = ["London, UK", "New York, USA", "Tokyo, Japan", "Paris, France", "Sydney, Australia", "Singapore"]
    bfcl_data = []

    for i in range(num_samples):
        target_name = all_tool_names[i % len(all_tool_names)]
        
        # Build query and args based on target_name
        if target_name == "calculate":
            a = random.randint(15, 950)
            b = random.randint(15, 950)
            op = random.choice(["add", "multiply", "subtract", "divide"])
            q = f"Perform arithmetic computation: {op} {a} and {b}."
            args = {"operation": op, "a": float(a), "b": float(b)}
        elif target_name == "get_current_weather":
            city = random.choice(cities)
            unit = random.choice(["metric", "imperial"])
            q = f"Get the current weather forecast for {city} in {unit} units."
            args = {"location": city, "unit": unit}
        elif target_name == "create_calendar_event":
            day = random.randint(1, 28)
            time_str = f"{random.randint(9, 17):02d}:00"
            dur = random.choice([30, 45, 60, 90])
            title = random.choice(["Quarterly Roadmap Review", "Architecture Sync", "Sprint Retrospective"])
            q = f"Schedule a calendar event titled '{title}' on 2026-11-{day:02d} at {time_str} lasting {dur} minutes."
            args = {"title": title, "date": f"2026-11-{day:02d}", "time": time_str, "duration_minutes": dur}
        elif target_name == "convert_units":
            val = random.randint(5, 120)
            u_from, u_to = random.choice([("km", "miles"), ("celsius", "fahrenheit"), ("kg", "pounds")])
            q = f"Convert {val} {u_from} into {u_to}."
            args = {"value": float(val), "from_unit": u_from, "to_unit": u_to}
        elif target_name == "send_email":
            email = f"lead_{random.randint(100, 999)}@corp.net"
            subj = random.choice(["Project Milestone Report", "Urgent Incident Update", "Weekly Digest"])
            urgent = random.choice([True, False])
            q = f"Send an {'urgent ' if urgent else ''}email to {email} with subject '{subj}' and body 'Please review the attached release notes immediately.'."
            args = {"recipient": email, "subject": subj, "body": "Please review the attached release notes immediately.", "urgent": urgent}
        elif target_name == "search_database":
            tbl = random.choice(["customers", "orders", "inventory", "transactions"])
            lim = random.choice([10, 25, 50])
            q = f"Search the '{tbl}' database table for records matching 'status=active' with a limit of {lim} records."
            args = {"table_name": tbl, "filter_query": "status=active", "limit": lim}
        elif target_name == "set_device_alarm":
            alarm_time = f"{random.randint(6, 9):02d}:30"
            lbl = random.choice(["Morning Standup Alarm", "Workout Reminder", "Medication Time"])
            rep = random.choice([True, False])
            q = f"Set an alarm for {alarm_time} labeled '{lbl}' {'with daily repeat' if rep else 'without repeating'}."
            args = {"alarm_time": alarm_time, "label": lbl, "repeat_daily": rep}
        else: # generate_image
            prompt_txt = random.choice(["Futuristic city skyline at sunset in cyberpunk style", "Hyper-realistic portrait of an astronaut on Mars", "Serene Japanese zen garden in autumn foliage"])
            res = random.choice(["512x512", "1024x1024", "1920x1080"])
            aspect = random.choice(["1:1", "16:9", "4:3"])
            q = f"Generate an image of '{prompt_txt}' at {res} resolution with {aspect} aspect ratio."
            args = {"prompt": prompt_txt, "resolution": res, "aspect_ratio": aspect}

        # Select 5 decoys from the remaining 7 tools
        decoys = [name for name in all_tool_names if name != target_name]
        sampled_decoys = random.sample(decoys, 5)
        selected_tool_names = [target_name] + sampled_decoys
        random.shuffle(selected_tool_names)
        candidate_schemas = [tool_pool[name] for name in selected_tool_names]

        prompt = build_tool_calling_prompt(q, candidate_schemas)
        bfcl_data.append({
            "id": f"eval_bfcl_{i:03d}",
            "domain": "bfcl_standard",
            "query": q,
            "target_function": target_name,
            "target_arguments": args,
            "candidate_tools": candidate_schemas,
            "prompt": prompt,
        })

    return bfcl_data


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def run_pipeline():
    print("=" * 70)
    print(">>> VERICALL DATA PIPELINE: Generation & Decontamination")
    print("=" * 70)

    # 1. Generate Datasets
    print("[*] Generating SFT training set (N=2,000 across 10 Finance/Commerce/CRM endpoints)...")
    train_data = generate_training_data(2000)

    print("[*] Generating frozen unseen eval set (M=500 across 6 Infra/Travel endpoints)...")
    eval_unseen = generate_unseen_eval_data(500)

    print("[*] Generating decontaminated BFCL evaluation subset (200 requests)...")
    eval_bfcl = generate_bfcl_subset(200)

    # 2. Strict Decontamination Assertions
    print("[*] Running strict decontamination verification...")
    train_domains = set(d["domain"] for d in train_data)
    unseen_domains = set(d["domain"] for d in eval_unseen)
    domain_overlap = train_domains.intersection(unseen_domains)
    assert len(domain_overlap) == 0, f"Decontamination failure! Overlapping domains: {domain_overlap}"
    print(f"    -> Domain overlap: ZERO (Train: {train_domains} | Unseen Eval: {unseen_domains})")

    train_queries = set(d["query"].strip().lower() for d in train_data)
    unseen_queries = set(d["query"].strip().lower() for d in eval_unseen)
    bfcl_queries = set(d["query"].strip().lower() for d in eval_bfcl)

    assert len(train_queries.intersection(unseen_queries)) == 0, "Query overlap detected between train and unseen!"
    assert len(train_queries.intersection(bfcl_queries)) == 0, "Query overlap detected between train and BFCL!"
    print("    -> Query overlap: ZERO across all splits.")

    # 3. Save Files
    train_file = DATA_DIR / "train_sft.jsonl"
    with open(train_file, "w", encoding="utf-8") as f:
        for r in train_data:
            f.write(json.dumps(r) + "\n")
    print(f"[+] Saved {len(train_data)} train records to {train_file.name}")

    unseen_file = DATA_DIR / "eval_unseen_apis.jsonl"
    with open(unseen_file, "w", encoding="utf-8") as f:
        for r in eval_unseen:
            f.write(json.dumps(r) + "\n")
    print(f"[+] Saved {len(eval_unseen)} unseen eval records to {unseen_file.name}")

    bfcl_file = DATA_DIR / "eval_bfcl_subset.jsonl"
    with open(bfcl_file, "w", encoding="utf-8") as f:
        for r in eval_bfcl:
            f.write(json.dumps(r) + "\n")
    print(f"[+] Saved {len(eval_bfcl)} BFCL eval records to {bfcl_file.name}")

    # 4. Generate SHA-256 Manifest
    manifest = {
        "dataset_metadata": {
            "created_at": "2026-10-21T00:00:00Z",
            "seed": SEED,
            "train_domains": list(train_domains),
            "unseen_eval_domains": list(unseen_domains),
            "train_endpoints": TRAIN_ENDPOINTS,
            "unseen_endpoints": UNSEEN_ENDPOINTS,
            "unseen_eval_count": len(eval_unseen),
            "bfcl_eval_count": len(eval_bfcl),
            "train_count": len(train_data),
        },
        "file_hashes": {
            "eval_unseen_apis.jsonl": {
                "sha256": compute_sha256(unseen_file),
                "record_count": len(eval_unseen),
            },
            "eval_bfcl_subset.jsonl": {
                "sha256": compute_sha256(bfcl_file),
                "record_count": len(eval_bfcl),
            },
            "train_sft.jsonl": {
                "sha256": compute_sha256(train_file),
                "record_count": len(train_data),
            },
        },
    }

    manifest_file = DATA_DIR / "test_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[+] Created test manifest: {manifest_file.name}")
    print(f"    eval_unseen_apis.jsonl SHA-256: {manifest['file_hashes']['eval_unseen_apis.jsonl']['sha256']}")
    print(f"    train_sft.jsonl SHA-256:        {manifest['file_hashes']['train_sft.jsonl']['sha256']}")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
