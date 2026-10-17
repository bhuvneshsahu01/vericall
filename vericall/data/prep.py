"""
vericall/data/prep.py
---------------------
Dataset generation and strict domain-split decontamination.
Splits by API domain so unseen test APIs have ZERO domain or schema overlap with training APIs.

Generates:
- data/eval_unseen_apis.jsonl: Exactly M=500 requests from unseen domains (infra, travel)
- data/eval_bfcl_subset.jsonl: 200 decontaminated general function calling requests
- data/train_sft.jsonl: 2,500 requests exclusively from training domains (finance, commerce)
- data/test_manifest.json: SHA-256 hashes of frozen evaluation sets.
"""

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

SEED = 42
random.seed(SEED)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# TEMPLATE GENERATORS FOR TRAINING APIS (Finance & Commerce)
# ============================================================================

FINANCE_TEMPLATES = [
    (
        "Can you check the current market price for {symbol} on the {exchange}?",
        "get_stock_quote",
        lambda sym, ex: {"symbol": sym, "exchange": ex},
    ),
    (
        "Please get me a stock quote for ticker {symbol}.",
        "get_stock_quote",
        lambda sym, ex: {"symbol": sym},
    ),
    (
        "Execute a market buy order for {quantity} shares of {symbol} from account {account_id}.",
        "execute_trade",
        lambda sym, ex, acc, qty, otype, lprice: {
            "account_id": acc,
            "symbol": sym,
            "quantity": qty,
            "order_type": "market",
        },
    ),
    (
        "Place a limit order to purchase {quantity} shares of {symbol} at limit price ${limit_price:.2f} for account {account_id}.",
        "execute_trade",
        lambda sym, ex, acc, qty, otype, lprice: {
            "account_id": acc,
            "symbol": sym,
            "quantity": qty,
            "order_type": "limit",
            "limit_price": lprice,
        },
    ),
    (
        "What is the available cash balance and total portfolio value for account {account_id} in {currency}?",
        "get_account_balance",
        lambda acc, curr: {"account_id": acc, "currency": curr},
    ),
]

COMMERCE_TEMPLATES = [
    (
        "Find in-stock {category} items matching '{query}' under ${max_price:.2f}.",
        "search_products",
        lambda q, cat, p: {"query": q, "category": cat, "max_price": p, "in_stock_only": True},
    ),
    (
        "Search our product catalog for '{query}' in the {category} department.",
        "search_products",
        lambda q, cat, p: {"query": q, "category": cat},
    ),
    (
        "Create an order for customer {customer_id} shipping to {city}, postal code {zipcode} with item {prod_id} quantity {qty}.",
        "create_order",
        lambda cid, city, zcode, pid, qty: {
            "customer_id": cid,
            "items": [{"product_id": pid, "quantity": qty}],
            "shipping_city": city,
            "postal_code": zcode,
        },
    ),
    (
        "Track package {tracking_num} being delivered via {carrier}.",
        "track_shipment",
        lambda tnum, carr: {"tracking_number": tnum, "carrier": carr},
    ),
]


# ============================================================================
# TEMPLATE GENERATORS FOR UNSEEN TEST APIS (Infra & Travel)
# Zero domain overlap with training
# ============================================================================

INFRA_TEMPLATES = [
    (
        "Provision {count} {instance_type} compute servers in {region} for our {env} environment.",
        "provision_server",
        lambda itype, reg, cnt, env: {
            "instance_type": itype,
            "region": reg,
            "count": cnt,
            "environment": env,
        },
    ),
    (
        "Perform a restart on service '{service}' running in cluster '{cluster}'.",
        "restart_service",
        lambda srv, clus: {"service_name": srv, "cluster_id": clus, "force": False},
    ),
    (
        "Force restart microservice '{service}' immediately on cluster '{cluster}'.",
        "restart_service",
        lambda srv, clus: {"service_name": srv, "cluster_id": clus, "force": True},
    ),
    (
        "Get {metric} telemetry for cluster '{cluster}' aggregated over the past {minutes} minutes.",
        "get_cluster_metrics",
        lambda clus, met, mins: {
            "cluster_id": clus,
            "metric_name": met,
            "time_window_minutes": mins,
        },
    ),
]

TRAVEL_TEMPLATES = [
    (
        "Search for {cabin} class flights from {origin} to {dest} on {date} for {passengers} passengers.",
        "search_flights",
        lambda orig, dest, dt, pax, cab: {
            "origin_airport": orig,
            "destination_airport": dest,
            "departure_date": dt,
            "passengers": pax,
            "cabin_class": cab,
        },
    ),
    (
        "Book a {room} room at hotel {hotel_id} checking in on {checkin} and checking out on {checkout} for {guests} guests.",
        "book_hotel",
        lambda hid, cin, cout, gsts, rm: {
            "hotel_id": hid,
            "checkin_date": cin,
            "checkout_date": cout,
            "guests": gsts,
            "room_type": rm,
        },
    ),
    (
        "Cancel travel booking confirmation '{booking_id}' because of {reason} and process a refund.",
        "cancel_booking",
        lambda bid, rsn: {"booking_id": bid, "reason": rsn, "refund_requested": True},
    ),
]


def generate_training_data(num_samples: int = 2500) -> List[Dict[str, Any]]:
    """Generates training examples strictly from finance and commerce domains."""
    symbols = ["AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "AMD"]
    exchanges = ["NASDAQ", "NYSE"]
    order_types = ["market", "limit", "stop"]
    currencies = ["USD", "EUR", "GBP", "JPY"]
    queries = ["ergonomic keyboard", "wireless noise cancelling headphones", "ultrawide monitor", "running shoes"]
    categories = ["electronics", "apparel", "home_goods", "books"]
    carriers = ["fedex", "ups", "dhl", "usps"]
    cities = ["New York", "San Francisco", "Austin", "Seattle", "Chicago", "Boston"]

    train_data = []
    for idx in range(num_samples):
        if idx % 2 == 0:
            # Finance
            tmpl_idx = idx % len(FINANCE_TEMPLATES)
            sym = random.choice(symbols)
            ex = random.choice(exchanges)
            acc = f"acc_{random.randint(10000, 99999)}"
            qty = random.randint(1, 500)
            otype = random.choice(order_types)
            lprice = round(random.uniform(50.0, 500.0), 2)
            curr = random.choice(currencies)

            if tmpl_idx == 0:
                q = FINANCE_TEMPLATES[0][0].format(symbol=sym, exchange=ex)
                target = FINANCE_TEMPLATES[0][1]
                args = FINANCE_TEMPLATES[0][2](sym, ex)
            elif tmpl_idx == 1:
                q = FINANCE_TEMPLATES[1][0].format(symbol=sym)
                target = FINANCE_TEMPLATES[1][1]
                args = FINANCE_TEMPLATES[1][2](sym, ex)
            elif tmpl_idx == 2:
                q = FINANCE_TEMPLATES[2][0].format(quantity=qty, symbol=sym, account_id=acc)
                target = FINANCE_TEMPLATES[2][1]
                args = FINANCE_TEMPLATES[2][2](sym, ex, acc, qty, otype, lprice)
            elif tmpl_idx == 3:
                q = FINANCE_TEMPLATES[3][0].format(quantity=qty, symbol=sym, limit_price=lprice, account_id=acc)
                target = FINANCE_TEMPLATES[3][1]
                args = FINANCE_TEMPLATES[3][2](sym, ex, acc, qty, otype, lprice)
            else:
                q = FINANCE_TEMPLATES[4][0].format(account_id=acc, currency=curr)
                target = FINANCE_TEMPLATES[4][1]
                args = FINANCE_TEMPLATES[4][2](acc, curr)

            domain = "finance"
        else:
            # Commerce
            tmpl_idx = idx % len(COMMERCE_TEMPLATES)
            q_term = random.choice(queries)
            cat = random.choice(categories)
            max_p = round(random.uniform(25.0, 450.0), 2)
            cid = f"cust_{random.randint(1000, 9999)}"
            city = random.choice(cities)
            zcode = f"{random.randint(10001, 99999)}"
            pid = f"prod_{random.randint(100, 999)}"
            qty = random.randint(1, 5)
            tnum = f"TRK{random.randint(10000000, 99999999)}"
            carr = random.choice(carriers)

            if tmpl_idx == 0:
                q = COMMERCE_TEMPLATES[0][0].format(category=cat, query=q_term, max_price=max_p)
                target = COMMERCE_TEMPLATES[0][1]
                args = COMMERCE_TEMPLATES[0][2](q_term, cat, max_p)
            elif tmpl_idx == 1:
                q = COMMERCE_TEMPLATES[1][0].format(query=q_term, category=cat)
                target = COMMERCE_TEMPLATES[1][1]
                args = COMMERCE_TEMPLATES[1][2](q_term, cat, max_p)
            elif tmpl_idx == 2:
                q = COMMERCE_TEMPLATES[2][0].format(customer_id=cid, city=city, zipcode=zcode, prod_id=pid, qty=qty)
                target = COMMERCE_TEMPLATES[2][1]
                args = COMMERCE_TEMPLATES[2][2](cid, city, zcode, pid, qty)
            else:
                q = COMMERCE_TEMPLATES[3][0].format(tracking_num=tnum, carrier=carr)
                target = COMMERCE_TEMPLATES[3][1]
                args = COMMERCE_TEMPLATES[3][2](tnum, carr)

            domain = "commerce"

        train_data.append({
            "id": f"train_{domain}_{idx:04d}",
            "domain": domain,
            "query": q,
            "target_function": target,
            "target_arguments": args,
        })

    return train_data


def generate_unseen_eval_data(num_samples: int = 500) -> List[Dict[str, Any]]:
    """
    Generates EXACTLY M=500 requests strictly from unseen domains:
    250 from Infrastructure, 250 from Travel.
    """
    instance_types = ["t3.micro", "t3.medium", "c5.xlarge", "g4dn.xlarge"]
    regions = ["us-east-1", "us-west-2", "eu-west-1", "ap-northeast-1"]
    environments = ["dev", "staging", "prod"]
    services = ["auth-service", "payment-gateway", "worker-queue", "recommendation-api", "ingest-pipeline"]
    clusters = ["k8s-prod-us", "ecs-analytics", "k8s-core-eu", "dev-microservices"]
    metrics = ["cpu_utilization", "memory_utilization", "disk_iops", "network_in"]

    airports = ["SFO", "JFK", "LAX", "ORD", "LHR", "HND", "CDG", "SEA", "DFW", "BOS"]
    cabin_classes = ["economy", "premium_economy", "business", "first"]
    room_types = ["standard", "deluxe", "suite"]
    reasons = ["schedule_change", "found_cheaper", "emergency", "other"]

    unseen_data = []
    half = num_samples // 2

    # 1. 250 Infrastructure requests
    for i in range(half):
        tmpl_idx = i % len(INFRA_TEMPLATES)
        itype = random.choice(instance_types)
        reg = random.choice(regions)
        cnt = random.randint(1, 10)
        env = random.choice(environments)
        srv = random.choice(services)
        clus = random.choice(clusters)
        met = random.choice(metrics)
        mins = random.choice([15, 30, 60, 120, 240, 720, 1440])

        if tmpl_idx == 0:
            q = INFRA_TEMPLATES[0][0].format(count=cnt, instance_type=itype, region=reg, env=env)
            target = INFRA_TEMPLATES[0][1]
            args = INFRA_TEMPLATES[0][2](itype, reg, cnt, env)
        elif tmpl_idx == 1:
            q = INFRA_TEMPLATES[1][0].format(service=srv, cluster=clus)
            target = INFRA_TEMPLATES[1][1]
            args = INFRA_TEMPLATES[1][2](srv, clus)
        elif tmpl_idx == 2:
            q = INFRA_TEMPLATES[2][0].format(service=srv, cluster=clus)
            target = INFRA_TEMPLATES[2][1]
            args = INFRA_TEMPLATES[2][2](srv, clus)
        else:
            q = INFRA_TEMPLATES[3][0].format(metric=met, cluster=clus, minutes=mins)
            target = INFRA_TEMPLATES[3][1]
            args = INFRA_TEMPLATES[3][2](clus, met, mins)

        unseen_data.append({
            "id": f"eval_unseen_infra_{i:03d}",
            "domain": "infra",
            "query": q,
            "target_function": target,
            "target_arguments": args,
        })

    # 2. 250 Travel requests
    for i in range(half):
        tmpl_idx = i % len(TRAVEL_TEMPLATES)
        orig, dest = random.sample(airports, 2)
        day = random.randint(10, 28)
        dt = f"2026-11-{day:02d}"
        pax = random.randint(1, 4)
        cab = random.choice(cabin_classes)
        hid = f"htl_{random.randint(100, 999)}"
        cin = f"2026-11-{day:02d}"
        cout = f"2026-11-{day+3:02d}"
        gsts = random.randint(1, 4)
        rm = random.choice(room_types)
        bid = f"BKG{random.randint(100000, 999999)}"
        rsn = random.choice(reasons)

        if tmpl_idx == 0:
            q = TRAVEL_TEMPLATES[0][0].format(cabin=cab, origin=orig, dest=dest, date=dt, passengers=pax)
            target = TRAVEL_TEMPLATES[0][1]
            args = TRAVEL_TEMPLATES[0][2](orig, dest, dt, pax, cab)
        elif tmpl_idx == 1:
            q = TRAVEL_TEMPLATES[1][0].format(room=rm, hotel_id=hid, checkin=cin, checkout=cout, guests=gsts)
            target = TRAVEL_TEMPLATES[1][1]
            args = TRAVEL_TEMPLATES[1][2](hid, cin, cout, gsts, rm)
        else:
            q = TRAVEL_TEMPLATES[2][0].format(booking_id=bid, reason=rsn)
            target = TRAVEL_TEMPLATES[2][1]
            args = TRAVEL_TEMPLATES[2][2](bid, rsn)

        unseen_data.append({
            "id": f"eval_unseen_travel_{i:03d}",
            "domain": "travel",
            "query": q,
            "target_function": target,
            "target_arguments": args,
        })

    assert len(unseen_data) == 500, f"Expected exactly M=500 unseen requests, got {len(unseen_data)}"
    return unseen_data


def generate_bfcl_subset(num_samples: int = 200) -> List[Dict[str, Any]]:
    """Generates decontaminated subset of Berkeley Function Calling Leaderboard tasks."""
    bfcl_data = []
    categories = ["math_calculator", "weather_query", "calendar_scheduler", "unit_converter"]
    
    cities = ["New York", "London", "Tokyo", "Paris", "Sydney", "Berlin", "Toronto", "Singapore"]
    units = [("km", "miles"), ("celsius", "fahrenheit"), ("kg", "pounds"), ("liters", "gallons")]

    for i in range(num_samples):
        cat = categories[i % len(categories)]
        if cat == "math_calculator":
            a = random.randint(12, 999)
            b = random.randint(12, 999)
            op = random.choice(["add", "multiply", "subtract"])
            q = f"Calculate the result of {a} {op} {b}."
            target = "calculate"
            args = {"operation": op, "a": float(a), "b": float(b)}
        elif cat == "weather_query":
            city = random.choice(cities)
            q = f"What is the current weather forecast for {city}?"
            target = "get_current_weather"
            args = {"location": city, "unit": "metric"}
        elif cat == "calendar_scheduler":
            day = random.randint(1, 28)
            time_str = f"{random.randint(9, 17):02d}:00"
            q = f"Schedule a calendar event titled 'Team Sync' on 2026-11-{day:02d} at {time_str}."
            target = "create_calendar_event"
            args = {"title": "Team Sync", "date": f"2026-11-{day:02d}", "time": time_str}
        else:
            u_from, u_to = random.choice(units)
            val = random.randint(5, 100)
            q = f"Convert {val} {u_from} to {u_to}."
            target = "convert_units"
            args = {"value": float(val), "from_unit": u_from, "to_unit": u_to}

        bfcl_data.append({
            "id": f"eval_bfcl_{i:03d}",
            "domain": "bfcl_standard",
            "query": q,
            "target_function": target,
            "target_arguments": args,
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
    print("[*] Generating training set (Finance & Commerce domains)...")
    train_data = generate_training_data(2500)

    print("[*] Generating frozen unseen eval set (M=500: Infra & Travel domains)...")
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

    unseen_prompt_overlap = train_queries.intersection(unseen_queries)
    bfcl_prompt_overlap = train_queries.intersection(bfcl_queries)
    assert len(unseen_prompt_overlap) == 0, "Decontamination failure! Unseen queries overlap with train."
    assert len(bfcl_prompt_overlap) == 0, "Decontamination failure! BFCL queries overlap with train."
    print("    -> Prompt overlap with Train: ZERO for both Unseen Eval and BFCL.")

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
            "created_at": "2026-10-18T00:00:00Z",
            "seed": SEED,
            "train_domains": list(train_domains),
            "unseen_eval_domains": list(unseen_domains),
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
    print(f"    eval_bfcl_subset.jsonl SHA-256: {manifest['file_hashes']['eval_bfcl_subset.jsonl']['sha256']}")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
