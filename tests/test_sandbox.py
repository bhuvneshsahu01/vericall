"""
tests/test_sandbox.py
---------------------
Unit tests verifying the 12 mock endpoints and the 5-status error taxonomy of ToolCallValidator.
"""

import pytest
from vericall.sandbox.validator import ToolCallValidator
from vericall.sandbox.models import StockQuoteRequest, ServerProvisionRequest


@pytest.fixture
def validator():
    return ToolCallValidator()


def test_validator_registry_contains_12_endpoints(validator):
    assert len(validator.registry) == 12
    domains = {"finance", "commerce", "infra", "travel"}
    actual_domains = set()
    for name, ep in validator.registry.items():
        if "stock" in name or "trade" in name or "balance" in name:
            actual_domains.add("finance")
        elif "product" in name or "order" in name or "shipment" in name:
            actual_domains.add("commerce")
        elif "server" in name or "service" in name or "cluster" in name:
            actual_domains.add("infra")
        elif "flight" in name or "hotel" in name or "booking" in name:
            actual_domains.add("travel")
    assert actual_domains == domains


def test_validator_success_case(validator):
    res = validator.validate_call("get_stock_quote", {"symbol": "AAPL", "exchange": "NASDAQ"})
    assert res.is_success is True
    assert res.status == "success"
    assert res.execution_response["status"] == "success"


def test_validator_wrong_endpoint(validator):
    # Non-existent endpoint
    res = validator.validate_call("non_existent_function", {"param": 1})
    assert res.is_success is False
    assert res.status == "wrong_endpoint"

    # Mismatch with expected endpoint
    res2 = validator.validate_call("get_stock_quote", {"symbol": "AAPL"}, expected_func="execute_trade")
    assert res2.is_success is False
    assert res2.status == "wrong_endpoint"


def test_validator_missing_required_argument(validator):
    # 'account_id' is required for execute_trade
    res = validator.validate_call("execute_trade", {"symbol": "AAPL", "quantity": 10, "order_type": "market"})
    assert res.is_success is False
    assert res.status == "missing_arg"
    assert "account_id" in res.error_message


def test_validator_hallucinated_argument(validator):
    # Extra unexpected parameter forbidden by strict Pydantic config
    res = validator.validate_call("get_stock_quote", {"symbol": "AAPL", "unauthorized_param": True})
    assert res.is_success is False
    assert res.status == "hallucinated_arg"
    assert "unauthorized_param" in res.error_message


def test_validator_type_error(validator):
    # Invalid stock symbol format (regex violation: ^[A-Z]{1,5}$)
    res = validator.validate_call("get_stock_quote", {"symbol": "toolongticker"})
    assert res.is_success is False
    assert res.status == "type_error"

    # Invalid enum value
    res2 = validator.validate_call("execute_trade", {
        "account_id": "acc_12345",
        "symbol": "AAPL",
        "quantity": 10,
        "order_type": "INVALID_TYPE",
    })
    assert res2.is_success is False
    assert res2.status == "type_error"


def test_validator_all_12_endpoints_execute_successfully(validator):
    test_cases = [
        # Finance
        ("get_stock_quote", {"symbol": "MSFT"}),
        ("execute_trade", {"account_id": "acc_1001", "symbol": "MSFT", "quantity": 5, "order_type": "market"}),
        ("get_account_balance", {"account_id": "acc_1001"}),
        # Commerce
        ("search_products", {"query": "laptop", "category": "electronics"}),
        ("create_order", {"customer_id": "cust_101", "shipping_city": "Austin", "postal_code": "78701", "items": [{"product_id": "prod_101", "quantity": 1}]}),
        ("track_shipment", {"tracking_number": "TRK12345678", "carrier": "fedex"}),
        # Infra (Unseen Domain)
        ("provision_server", {"instance_type": "c5.xlarge", "region": "us-east-1", "count": 2, "environment": "prod"}),
        ("restart_service", {"service_name": "nginx", "cluster_id": "k8s-prod-us-east-1", "force": False}),
        ("get_cluster_metrics", {"cluster_id": "k8s-prod-us-east-1", "metric_name": "cpu_utilization", "time_window_minutes": 15}),
        # Travel (Unseen Domain)
        ("search_flights", {"origin_airport": "JFK", "destination_airport": "LHR", "departure_date": "2026-11-01", "passengers": 1, "cabin_class": "economy"}),
        ("book_hotel", {"hotel_id": "htl-992", "checkin_date": "2026-12-01", "checkout_date": "2026-12-05", "guests": 2, "room_type": "deluxe"}),
        ("cancel_booking", {"booking_id": "bk-58291", "reason": "schedule_change", "refund_requested": True}),
    ]

    for fn, args in test_cases:
        res = validator.validate_call(fn, args)
        assert res.is_success is True, f"Failed for {fn}: {res.error_message}"
        assert res.status == "success"
