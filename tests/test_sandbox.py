"""
tests/test_sandbox.py
---------------------
Unit tests verifying the 16 mock endpoints and the 6-state error taxonomy of ExecutionValidator.
"""

import pytest
from vericall.sandbox.validator import ExecutionValidator


@pytest.fixture
def validator():
    return ExecutionValidator()


def test_validator_registry_contains_16_endpoints(validator):
    assert len(validator.registry) == 16
    domains = {"finance", "commerce", "crm", "infra", "travel"}
    actual_domains = set(ep["domain"] for ep in validator.registry.values())
    assert actual_domains == domains


def test_validator_passed_case(validator):
    call = {
        "name": "provision_compute_node",
        "arguments": {
            "instance_tier": "c6i.xlarge",
            "cloud_region": "us-east-1",
            "cluster_count": 5,
            "environment_tag": "PRODUCTION",
            "resource_tags": {"owner": "platform", "cost_center": "cc-101"},
        }
    }
    res = validator.validate_call(call, expected_func="provision_compute_node")
    assert res.is_success is True
    assert res.status == "passed"
    assert res.execution_response["status"] == "success"


def test_validator_invalid_json(validator):
    res = validator.validate_call("not a json string {", expected_func="provision_compute_node")
    assert res.is_success is False
    assert res.status == "invalid_json"


def test_validator_wrong_endpoint(validator):
    call = {"name": "scale_kubernetes_cluster", "arguments": {}}
    res = validator.validate_call(call, expected_func="provision_compute_node")
    assert res.is_success is False
    assert res.status == "wrong_endpoint"


def test_validator_missing_required_argument(validator):
    call = {
        "name": "provision_compute_node",
        "arguments": {
            "instance_tier": "c6i.xlarge",
            "cloud_region": "us-east-1",
            # missing cluster_count, environment_tag, resource_tags
        }
    }
    res = validator.validate_call(call, expected_func="provision_compute_node")
    assert res.is_success is False
    assert res.status == "missing_arg"


def test_validator_hallucinated_argument(validator):
    call = {
        "name": "provision_compute_node",
        "arguments": {
            "instance_tier": "c6i.xlarge",
            "cloud_region": "us-east-1",
            "cluster_count": 5,
            "environment_tag": "PRODUCTION",
            "resource_tags": {"owner": "platform"},
            "unauthorized_extra_key": True,  # violates extra="forbid"
        }
    }
    res = validator.validate_call(call, expected_func="provision_compute_node")
    assert res.is_success is False
    assert res.status == "hallucinated_arg"


def test_validator_type_error(validator):
    call = {
        "name": "provision_compute_node",
        "arguments": {
            "instance_tier": "INVALID_TIER",  # invalid enum
            "cloud_region": "us-east-1",
            "cluster_count": 5,
            "environment_tag": "PRODUCTION",
            "resource_tags": {"owner": "platform"},
        }
    }
    res = validator.validate_call(call, expected_func="provision_compute_node")
    assert res.is_success is False
    assert res.status == "type_error"


def test_validator_all_16_endpoints_execute_successfully(validator):
    """Smoke test ensuring every one of the 16 registered endpoints can execute with valid data."""
    valid_samples = [
        ("issue_refund", {
            "transaction_id": "txn_12345678",
            "amount": 49.99,
            "currency": "USD",
            "reason": "CUSTOMER_RETURN",
            "metadata": {"rep": "rep_1"}
        }),
        ("create_invoice", {
            "client_id": "cli_1001",
            "issue_date": "2026-11-01",
            "due_date": "2026-11-15",
            "line_items": [{"description": "Dev", "unit_price": 100.0, "quantity": 1}],
            "currency": "USD"
        }),
        ("transfer_funds", {
            "source_account": "acc_100001",
            "destination_account": "acc_100002",
            "amount": 250.0,
            "transfer_method": "ACH",
            "memo": "Payment settlement"
        }),
        ("audit_ledger_entries", {
            "ledger_id": "ledg_100001",
            "start_date": "2026-10-01",
            "end_date": "2026-10-31",
            "entry_type": "CREDIT",
            "filter_flags": {"reconciled_only": True}
        }),
        ("query_inventory", {
            "sku_list": ["SKU-PROD01"],
            "warehouse_id": "wh_us-01",
            "include_out_of_stock": False,
            "fulfillment_channel": "WAREHOUSE_DIRECT"
        }),
        ("apply_discount_code", {
            "cart_id": "cart_100001",
            "promo_code": "SAVE20",
            "channel": "ONLINE_WEB",
            "customer_loyalty_pin": 1234
        }),
        ("update_subscription", {
            "subscription_id": "sub_10000001",
            "plan_tier": "PRO_MONTHLY",
            "auto_renew": True,
            "addon_features": ["logs"],
            "effective_date": "2026-12-01"
        }),
        ("create_support_ticket", {
            "customer_id": "cust_10001",
            "priority": "HIGH",
            "department": "TECHNICAL",
            "subject": "Platform latency issue",
            "ticket_context": {"origin": "web"}
        }),
        ("update_customer_tier", {
            "customer_id": "cust_10001",
            "target_tier": "GOLD",
            "assigned_rep_id": "rep_1001",
            "override_reason": "High spending account",
            "benefits": ["waived_fees"]
        }),
        ("schedule_webhook", {
            "destination_url": "https://api.partner.com/events",
            "subscribed_events": ["order.created"],
            "retry_count": 3,
            "signing_secret": "whsec_12345678",
            "headers": {"X-App": "test"}
        }),
        ("provision_compute_node", {
            "instance_tier": "c6i.xlarge",
            "cloud_region": "us-east-1",
            "cluster_count": 2,
            "environment_tag": "PRODUCTION",
            "resource_tags": {"team": "infra"}
        }),
        ("scale_kubernetes_cluster", {
            "cluster_name": "k8s-prod-101",
            "node_pool": "worker-pool",
            "desired_replicas": 5,
            "scaling_strategy": "SURGE_FIRST",
            "autoscale_config": {"min_nodes": 2, "max_nodes": 20}
        }),
        ("rotate_iam_credentials", {
            "service_account_id": "sa-data1001",
            "credential_type": "OAUTH2_KEY",
            "grace_period_hours": 48,
            "notify_emails": ["ops@company.com"],
            "force_revocation": False
        }),
        ("search_multi_leg_flights", {
            "itinerary_segments": [{"departure_airport": "JFK", "arrival_airport": "LHR", "flight_date": "2026-11-10"}],
            "cabin_class": "BUSINESS",
            "passenger_count": 1,
            "flexible_dates": False,
            "preferred_alliances": ["STAR_ALLIANCE"]
        }),
        ("reserve_corporate_hotel", {
            "property_code": "HTL-NY-1001",
            "check_in_date": "2026-12-01",
            "check_out_date": "2026-12-05",
            "room_category": "EXECUTIVE_SUITE",
            "corporate_billing_code": "CORP-10001",
            "guest_manifest": [{"first_name": "Jane", "last_name": "Doe"}]
        }),
        ("cancel_itinerary_segment", {
            "booking_reference": "PNR-ABC123",
            "segment_id": "seg_01",
            "cancellation_reason": "CLIENT_RESCHEDULE",
            "request_credit_voucher": True,
            "refund_override_notes": "Client request authorized"
        }),
    ]

    for fn, args in valid_samples:
        res = validator.validate_call({"name": fn, "arguments": args}, expected_func=fn)
        assert res.is_success is True, f"Failed for {fn}: {res.error_message}"
        assert res.status == "passed"
