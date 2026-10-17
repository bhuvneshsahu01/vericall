"""
Pydantic V2 Request Schemas for VeriCall Mock API Sandbox.
Strict validation with ConfigDict(extra="forbid") to catch hallucinated arguments.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class StrictBaseModel(BaseModel):
    """Base model forbidding any hallucinated or un-modeled parameters."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ============================================================================
# DOMAIN: FINANCE (Training APIs)
# ============================================================================

class StockQuoteRequest(StrictBaseModel):
    symbol: str = Field(..., pattern=r"^[A-Z]{1,5}$", description="Stock ticker symbol (e.g., AAPL, NVDA)")
    exchange: Optional[str] = Field(default=None, description="Optional stock exchange (e.g., NASDAQ, NYSE)")


class TradeRequest(StrictBaseModel):
    account_id: str = Field(..., min_length=4, max_length=20, description="Brokerage account ID")
    symbol: str = Field(..., pattern=r"^[A-Z]{1,5}$", description="Stock ticker symbol")
    quantity: int = Field(..., ge=1, le=100000, description="Number of shares to trade (positive integer)")
    order_type: Literal["market", "limit", "stop"] = Field(..., description="Order execution type")
    limit_price: Optional[float] = Field(default=None, gt=0.0, description="Required for limit orders")


class AccountBalanceRequest(StrictBaseModel):
    account_id: str = Field(..., min_length=4, max_length=20, description="Customer account ID")
    currency: Literal["USD", "EUR", "GBP", "JPY"] = Field(default="USD", description="Currency code")


# ============================================================================
# DOMAIN: COMMERCE (Training APIs)
# ============================================================================

class ProductSearchRequest(StrictBaseModel):
    query: str = Field(..., min_length=1, max_length=100, description="Search term")
    category: Literal["electronics", "apparel", "home_goods", "books"] = Field(..., description="Product category")
    max_price: Optional[float] = Field(default=None, ge=0.0, description="Maximum price filter in USD")
    in_stock_only: bool = Field(default=True, description="Filter for in-stock products only")


class OrderItem(StrictBaseModel):
    product_id: str = Field(..., min_length=3, description="SKU or product identifier")
    quantity: int = Field(..., ge=1, le=50, description="Quantity to purchase")


class OrderCreateRequest(StrictBaseModel):
    customer_id: str = Field(..., min_length=4, description="Customer account identifier")
    items: List[OrderItem] = Field(..., min_length=1, description="List of items in the order")
    shipping_city: str = Field(..., min_length=2, description="Shipping destination city")
    postal_code: str = Field(..., min_length=3, max_length=10, description="Postal or ZIP code")


class ShipmentTrackRequest(StrictBaseModel):
    tracking_number: str = Field(..., pattern=r"^[A-Z0-9]{8,16}$", description="Tracking identifier")
    carrier: Literal["fedex", "ups", "dhl", "usps"] = Field(..., description="Logistics carrier")


# ============================================================================
# DOMAIN: INFRASTRUCTURE (Unseen Evaluation APIs)
# Zero overlap with training domain
# ============================================================================

class ServerProvisionRequest(StrictBaseModel):
    instance_type: Literal["t3.micro", "t3.medium", "c5.xlarge", "g4dn.xlarge"] = Field(
        ..., description="Cloud compute instance size"
    )
    region: Literal["us-east-1", "us-west-2", "eu-west-1", "ap-northeast-1"] = Field(
        ..., description="Deployment cloud region"
    )
    count: int = Field(..., ge=1, le=20, description="Number of instances to spin up")
    environment: Literal["dev", "staging", "prod"] = Field(..., description="Target environment")


class ServiceRestartRequest(StrictBaseModel):
    service_name: str = Field(..., min_length=2, max_length=50, description="Name of microservice to restart")
    cluster_id: str = Field(..., min_length=3, description="Target cluster ID")
    force: bool = Field(default=False, description="Whether to perform a hard termination restart")


class ClusterMetricsRequest(StrictBaseModel):
    cluster_id: str = Field(..., min_length=3, description="Target Kubernetes or ECS cluster ID")
    metric_name: Literal["cpu_utilization", "memory_utilization", "disk_iops", "network_in"] = Field(
        ..., description="Telemetry metric name"
    )
    time_window_minutes: int = Field(..., ge=1, le=1440, description="Aggregation window in minutes (1 to 1440)")


# ============================================================================
# DOMAIN: TRAVEL (Unseen Evaluation APIs)
# Zero overlap with training domain
# ============================================================================

class FlightSearchRequest(StrictBaseModel):
    origin_airport: str = Field(..., pattern=r"^[A-Z]{3}$", description="3-letter IATA departure airport code")
    destination_airport: str = Field(..., pattern=r"^[A-Z]{3}$", description="3-letter IATA arrival airport code")
    departure_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Date formatted as YYYY-MM-DD")
    passengers: int = Field(..., ge=1, le=9, description="Number of travelers")
    cabin_class: Literal["economy", "premium_economy", "business", "first"] = Field(
        default="economy", description="Cabin travel class"
    )


class HotelBookingRequest(StrictBaseModel):
    hotel_id: str = Field(..., min_length=3, description="Hotel property identifier")
    checkin_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Check-in date (YYYY-MM-DD)")
    checkout_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Check-out date (YYYY-MM-DD)")
    guests: int = Field(..., ge=1, le=8, description="Number of hotel guests")
    room_type: Literal["standard", "deluxe", "suite"] = Field(default="standard", description="Type of room")


class BookingCancelRequest(StrictBaseModel):
    booking_id: str = Field(..., min_length=4, description="Reservation or booking confirmation code")
    reason: Literal["schedule_change", "found_cheaper", "emergency", "other"] = Field(
        ..., description="Cancellation reason"
    )
    refund_requested: bool = Field(default=True, description="Whether to request an automated refund")
