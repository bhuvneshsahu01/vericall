"""
FastAPI Mock Server implementing 12 realistic endpoints across 4 domains.
Used for execution-verified post-training error mining and live validation.
"""

from typing import Any, Dict
from fastapi import FastAPI, HTTPException, status
from .models import (
    StockQuoteRequest,
    TradeRequest,
    AccountBalanceRequest,
    ProductSearchRequest,
    OrderCreateRequest,
    ShipmentTrackRequest,
    ServerProvisionRequest,
    ServiceRestartRequest,
    ClusterMetricsRequest,
    FlightSearchRequest,
    HotelBookingRequest,
    BookingCancelRequest,
)

app = FastAPI(
    title="VeriCall Mock API Sandbox",
    description="Execution-Verified Tool Calling API Sandbox with Strict Pydantic V2 Schemas",
    version="1.0.0",
)

# Registry mapping tool function names to metadata and schema
ENDPOINT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "get_stock_quote": {
        "domain": "finance",
        "model": StockQuoteRequest,
        "description": "Fetch real-time stock quote and market price for a given ticker symbol.",
    },
    "execute_trade": {
        "domain": "finance",
        "model": TradeRequest,
        "description": "Execute a buy or sell trade order for an account.",
    },
    "get_account_balance": {
        "domain": "finance",
        "model": AccountBalanceRequest,
        "description": "Retrieve current balance and cash positions for a brokerage account.",
    },
    "search_products": {
        "domain": "commerce",
        "model": ProductSearchRequest,
        "description": "Search product catalog with category and price constraints.",
    },
    "create_order": {
        "domain": "commerce",
        "model": OrderCreateRequest,
        "description": "Create a new customer order with line items and delivery address.",
    },
    "track_shipment": {
        "domain": "commerce",
        "model": ShipmentTrackRequest,
        "description": "Query live tracking status and carrier events for a shipment.",
    },
    "provision_server": {
        "domain": "infra",
        "model": ServerProvisionRequest,
        "description": "Spin up virtual compute instances in a target region and environment.",
    },
    "restart_service": {
        "domain": "infra",
        "model": ServiceRestartRequest,
        "description": "Perform rolling or forced restart of a microservice on a cluster.",
    },
    "get_cluster_metrics": {
        "domain": "infra",
        "model": ClusterMetricsRequest,
        "description": "Query time-series telemetry metrics for a cloud cluster.",
    },
    "search_flights": {
        "domain": "travel",
        "model": FlightSearchRequest,
        "description": "Search for one-way flight schedules, availability, and cabin classes.",
    },
    "book_hotel": {
        "domain": "travel",
        "model": HotelBookingRequest,
        "description": "Reserve hotel rooms with check-in/check-out dates and guest counts.",
    },
    "cancel_booking": {
        "domain": "travel",
        "model": BookingCancelRequest,
        "description": "Cancel an existing travel booking with stated reason and refund request.",
    },
}


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "vericall-sandbox", "endpoints": len(ENDPOINT_REGISTRY)}


# Finance Endpoints
@app.post("/finance/stock-quote")
def get_stock_quote(req: StockQuoteRequest):
    return {
        "status": "success",
        "symbol": req.symbol,
        "price": 182.45,
        "currency": "USD",
        "exchange": req.exchange or "NASDAQ",
    }


@app.post("/finance/trade")
def execute_trade(req: TradeRequest):
    if req.order_type == "limit" and req.limit_price is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="limit_price is required when order_type is 'limit'",
        )
    return {
        "status": "success",
        "order_id": f"ord_{req.account_id[-4:]}_{req.symbol}",
        "filled_quantity": req.quantity,
        "execution_price": req.limit_price or 182.45,
        "order_type": req.order_type,
    }


@app.post("/finance/balance")
def get_account_balance(req: AccountBalanceRequest):
    return {
        "status": "success",
        "account_id": req.account_id,
        "currency": req.currency,
        "available_cash": 45200.50,
        "portfolio_value": 128900.00,
    }


# Commerce Endpoints
@app.post("/commerce/products/search")
def search_products(req: ProductSearchRequest):
    return {
        "status": "success",
        "query": req.query,
        "category": req.category,
        "results_count": 8,
        "sample_item": {"id": "prod_001", "name": f"{req.query} Pro", "price": req.max_price or 99.99},
    }


@app.post("/commerce/orders/create")
def create_order(req: OrderCreateRequest):
    return {
        "status": "success",
        "order_id": f"cart_{req.customer_id}_{len(req.items)}",
        "total_items": sum(i.quantity for i in req.items),
        "status_code": "confirmed",
        "destination": f"{req.shipping_city}, {req.postal_code}",
    }


@app.post("/commerce/shipments/track")
def track_shipment(req: ShipmentTrackRequest):
    return {
        "status": "success",
        "tracking_number": req.tracking_number,
        "carrier": req.carrier,
        "current_status": "in_transit",
        "estimated_delivery_days": 2,
    }


# Infrastructure Endpoints (Unseen Test Domain 1)
@app.post("/infra/servers/provision")
def provision_server(req: ServerProvisionRequest):
    return {
        "status": "success",
        "job_id": f"prov_{req.region}_{req.count}",
        "instances": [f"i-{req.instance_type[:3]}{idx:03d}" for idx in range(req.count)],
        "environment": req.environment,
    }


@app.post("/infra/services/restart")
def restart_service(req: ServiceRestartRequest):
    return {
        "status": "success",
        "service": req.service_name,
        "cluster": req.cluster_id,
        "restart_type": "forced" if req.force else "rolling",
        "active_replicas": 3,
    }


@app.post("/infra/clusters/metrics")
def get_cluster_metrics(req: ClusterMetricsRequest):
    return {
        "status": "success",
        "cluster_id": req.cluster_id,
        "metric": req.metric_name,
        "average_value": 64.2,
        "window_minutes": req.time_window_minutes,
    }


# Travel Endpoints (Unseen Test Domain 2)
@app.post("/travel/flights/search")
def search_flights(req: FlightSearchRequest):
    return {
        "status": "success",
        "route": f"{req.origin_airport}->{req.destination_airport}",
        "departure_date": req.departure_date,
        "available_seats": 14,
        "cabin_class": req.cabin_class,
        "starting_fare_usd": 349.00,
    }


@app.post("/travel/hotels/book")
def book_hotel(req: HotelBookingRequest):
    return {
        "status": "success",
        "booking_reference": f"htl_{req.hotel_id}_{req.room_type}",
        "nights": 3,
        "guests": req.guests,
        "confirmed": True,
    }


@app.post("/travel/bookings/cancel")
def cancel_booking(req: BookingCancelRequest):
    return {
        "status": "success",
        "booking_id": req.booking_id,
        "cancellation_processed": True,
        "refund_initiated": req.refund_requested,
    }
