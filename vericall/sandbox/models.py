"""
vericall/sandbox/models.py
--------------------------
Strict Pydantic V2 Request Schemas for VeriCall 16-Endpoint Mock Sandbox.
Configured with ConfigDict(extra="forbid", strict=True) to enforce:
1. Zero parameter hallucination (extra keys forbidden).
2. Strict type enforcement (strict=True rejects coercion like "100" -> 100 or "true" -> True).
3. Exact Enum literals (case-sensitive).
4. Pattern and range validation (ISO-8601 dates, currency codes, regex identifiers).
5. Nested objects (dicts, lists of sub-models).

Endpoints:
- 10 Training Endpoints (Finance, E-Commerce, CRM)
- 6 Unseen Evaluation Endpoints (Cloud Infrastructure, Travel Logistics)
"""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class StrictBaseModel(BaseModel):
    """Base model forbidding extra keys and type coercion."""
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)


# ============================================================================
# SUB-MODELS (Nested Structures)
# ============================================================================

class InvoiceLineItem(StrictBaseModel):
    description: str = Field(..., min_length=2, max_length=100, description="Item description")
    unit_price: float = Field(..., gt=0.0, description="Unit price per item")
    quantity: int = Field(..., ge=1, le=1000, description="Integer quantity of items")


class AutoscaleConfig(StrictBaseModel):
    min_nodes: int = Field(..., ge=1, le=50, description="Minimum cluster nodes")
    max_nodes: int = Field(..., ge=1, le=200, description="Maximum cluster nodes")


class FlightSegment(StrictBaseModel):
    departure_airport: str = Field(..., pattern=r"^[A-Z]{3}$", description="3-letter IATA departure code")
    arrival_airport: str = Field(..., pattern=r"^[A-Z]{3}$", description="3-letter IATA arrival code")
    flight_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Flight date (YYYY-MM-DD)")


class GuestManifest(StrictBaseModel):
    first_name: str = Field(..., min_length=1, max_length=50, description="Guest first name")
    last_name: str = Field(..., min_length=1, max_length=50, description="Guest last name")


# ============================================================================
# TRAINING DOMAINS (10 ENDPOINTS)
# ============================================================================

# --- Domain 1: Finance ---

class IssueRefundRequest(StrictBaseModel):
    transaction_id: str = Field(..., pattern=r"^txn_[a-zA-Z0-9]{8,16}$", description="Original transaction ID")
    amount: float = Field(..., gt=0.0, le=50000.0, description="Refund amount in currency units")
    currency: Literal["USD", "EUR", "GBP", "INR"] = Field(..., description="ISO currency code")
    reason: Literal["CUSTOMER_RETURN", "DUPLICATE_CHARGE", "FRAUDULENT", "SERVICE_OUTAGE"] = Field(
        ..., description="Standardized refund justification"
    )
    metadata: Dict[str, str] = Field(..., description="Audit metadata key-value tags")


class CreateInvoiceRequest(StrictBaseModel):
    client_id: str = Field(..., pattern=r"^cli_[0-9]{4,8}$", description="Unique client identifier")
    issue_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Invoice issue date (YYYY-MM-DD)")
    due_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Payment due date (YYYY-MM-DD)")
    line_items: List[InvoiceLineItem] = Field(..., min_length=1, description="List of invoiced line items")
    currency: Literal["USD", "EUR", "GBP", "INR"] = Field(..., description="Billing currency code")


class TransferFundsRequest(StrictBaseModel):
    source_account: str = Field(..., pattern=r"^acc_[0-9]{6,12}$", description="Originating account ID")
    destination_account: str = Field(..., pattern=r"^acc_[0-9]{6,12}$", description="Receiving account ID")
    amount: float = Field(..., gt=0.0, description="Transfer amount")
    transfer_method: Literal["ACH", "WIRE", "SWIFT", "SEPA"] = Field(..., description="Clearing network")
    memo: str = Field(..., min_length=3, max_length=100, description="Transfer explanation note")


class AuditLedgerEntriesRequest(StrictBaseModel):
    ledger_id: str = Field(..., pattern=r"^ledg_[a-z0-9]{6,10}$", description="Target ledger partition")
    start_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Audit window start (YYYY-MM-DD)")
    end_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Audit window end (YYYY-MM-DD)")
    entry_type: Literal["CREDIT", "DEBIT", "ADJUSTMENT", "TRANSFER"] = Field(..., description="Ledger entry category")
    filter_flags: Dict[str, bool] = Field(..., description="Audit filtering boolean flags")


# --- Domain 2: E-Commerce ---

class QueryInventoryRequest(StrictBaseModel):
    sku_list: List[str] = Field(..., min_length=1, description="List of product SKU identifiers")
    warehouse_id: str = Field(..., pattern=r"^wh_[a-z]{2}-[0-9]{2}$", description="Facility code (e.g. wh_us-01)")
    include_out_of_stock: bool = Field(..., description="Whether to include zero-quantity stock")
    fulfillment_channel: Literal["IN_STORE", "WAREHOUSE_DIRECT", "DROPSHIP"] = Field(
        ..., description="Logistics fulfillment route"
    )


class ApplyDiscountCodeRequest(StrictBaseModel):
    cart_id: str = Field(..., pattern=r"^cart_[a-zA-Z0-9]{6,12}$", description="Active shopping cart ID")
    promo_code: str = Field(..., pattern=r"^[A-Z0-9_-]{4,16}$", description="Alphanumeric discount code")
    channel: Literal["ONLINE_WEB", "MOBILE_APP", "POS_REGISTER"] = Field(..., description="Point of sale channel")
    customer_loyalty_pin: int = Field(..., ge=1000, le=9999, description="4-digit customer PIN")


class UpdateSubscriptionRequest(StrictBaseModel):
    subscription_id: str = Field(..., pattern=r"^sub_[a-z0-9]{8,12}$", description="Customer subscription ID")
    plan_tier: Literal["BASIC_MONTHLY", "PRO_MONTHLY", "ENTERPRISE_ANNUAL"] = Field(
        ..., description="Target membership plan"
    )
    auto_renew: bool = Field(..., description="Automatic recurring renewal toggle")
    addon_features: List[str] = Field(..., description="Enabled supplementary add-on modules")
    effective_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Activation date (YYYY-MM-DD)")


# --- Domain 3: CRM ---

class CreateSupportTicketRequest(StrictBaseModel):
    customer_id: str = Field(..., pattern=r"^cust_[0-9]{5,10}$", description="Verified customer identifier")
    priority: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(..., description="Incident severity level")
    department: Literal["BILLING", "TECHNICAL", "ACCOUNT", "SHIPPING"] = Field(..., description="Handling department")
    subject: str = Field(..., min_length=5, max_length=120, description="Brief summary of issue")
    ticket_context: Dict[str, str] = Field(..., description="Client diagnostic context tags")


class UpdateCustomerTierRequest(StrictBaseModel):
    customer_id: str = Field(..., pattern=r"^cust_[0-9]{5,10}$", description="Customer identifier")
    target_tier: Literal["BRONZE", "SILVER", "GOLD", "PLATINUM"] = Field(..., description="New tier assignment")
    assigned_rep_id: str = Field(..., pattern=r"^rep_[0-9]{4}$", description="Account manager employee ID")
    override_reason: str = Field(..., min_length=5, max_length=150, description="Business justification for update")
    benefits: List[str] = Field(..., min_length=1, description="Active tier perks enabled")


class ScheduleWebhookRequest(StrictBaseModel):
    destination_url: str = Field(..., pattern=r"^https://.*", description="HTTPS endpoint receiver URL")
    subscribed_events: List[str] = Field(..., min_length=1, description="Topic events to broadcast")
    retry_count: int = Field(..., ge=0, le=5, description="Maximum delivery retry attempts (0 to 5)")
    signing_secret: str = Field(..., min_length=8, max_length=64, description="HMAC SHA-256 signing key")
    headers: Dict[str, str] = Field(..., description="Custom HTTP headers to include with dispatch")


# ============================================================================
# UNSEEN EVALUATION DOMAINS (6 ENDPOINTS: ZERO OVERLAP WITH TRAINING)
# ============================================================================

# --- Domain 4: Cloud Infrastructure ---

class ProvisionComputeNodeRequest(StrictBaseModel):
    instance_tier: Literal["c6i.xlarge", "c6i.2xlarge", "g5.xlarge", "m6i.4xlarge"] = Field(
        ..., description="Compute instance flavor specification"
    )
    cloud_region: Literal["us-east-1", "us-west-2", "eu-central-1", "ap-southeast-1"] = Field(
        ..., description="Target cloud deployment region"
    )
    cluster_count: int = Field(..., ge=1, le=50, description="Number of worker nodes to spin up")
    environment_tag: Literal["DEVELOPMENT", "STAGING", "PRODUCTION"] = Field(
        ..., description="Deployment life-cycle tier"
    )
    resource_tags: Dict[str, str] = Field(..., description="Cost allocation and owner metadata tags")


class ScaleKubernetesClusterRequest(StrictBaseModel):
    cluster_name: str = Field(..., pattern=r"^k8s-[a-z0-9-]{4,20}$", description="Kubernetes cluster identifier")
    node_pool: str = Field(..., min_length=2, max_length=30, description="Target node group pool name")
    desired_replicas: int = Field(..., ge=0, le=100, description="Target worker node count")
    scaling_strategy: Literal["IMMEDIATE", "DRAIN_GRADUAL", "SURGE_FIRST"] = Field(
        ..., description="Workload migration strategy"
    )
    autoscale_config: AutoscaleConfig = Field(..., description="Autoscaler boundary limits")


class RotateIamCredentialsRequest(StrictBaseModel):
    service_account_id: str = Field(..., pattern=r"^sa-[a-z0-9]{6,12}$", description="Service account principal ID")
    credential_type: Literal["OAUTH2_KEY", "SSH_RSA", "X509_CERTIFICATE", "API_TOKEN"] = Field(
        ..., description="Secret authentication mechanism"
    )
    grace_period_hours: int = Field(..., ge=1, le=720, description="Overlap hours before old secret expires")
    notify_emails: List[str] = Field(..., min_length=1, description="Notification email distribution list")
    force_revocation: bool = Field(..., description="Whether to immediately invalidate compromised secret")


# --- Domain 5: Travel Logistics ---

class SearchMultiLegFlightsRequest(StrictBaseModel):
    itinerary_segments: List[FlightSegment] = Field(..., min_length=1, description="Ordered flight journey legs")
    cabin_class: Literal["ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST"] = Field(
        ..., description="Travel seating cabin class"
    )
    passenger_count: int = Field(..., ge=1, le=9, description="Number of traveling passengers")
    flexible_dates: bool = Field(..., description="Whether +/- 3 day fare window is allowed")
    preferred_alliances: List[str] = Field(..., description="Airline alliance carrier preferences")


class ReserveCorporateHotelRequest(StrictBaseModel):
    property_code: str = Field(..., pattern=r"^HTL-[A-Z]{2}-[0-9]{4}$", description="Hotel property identifier")
    check_in_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Arrival date (YYYY-MM-DD)")
    check_out_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Departure date (YYYY-MM-DD)")
    room_category: Literal["STANDARD_KING", "EXECUTIVE_SUITE", "DOUBLE_QUEEN", "PRESIDENTIAL"] = Field(
        ..., description="Room classification"
    )
    corporate_billing_code: str = Field(..., pattern=r"^CORP-[0-9]{4,8}$", description="Corporate cost center code")
    guest_manifest: List[GuestManifest] = Field(..., min_length=1, description="Registered guest details")


class CancelItinerarySegmentRequest(StrictBaseModel):
    booking_reference: str = Field(..., pattern=r"^PNR-[A-Z0-9]{6}$", description="Passenger Name Record locator")
    segment_id: str = Field(..., pattern=r"^seg_[0-9]{2,4}$", description="Individual itinerary leg ID")
    cancellation_reason: Literal["CLIENT_RESCHEDULE", "WEATHER_DISRUPTION", "MEETING_CANCELLED", "DUPLICATE_BOOKING"] = Field(
        ..., description="Standard cancellation code"
    )
    request_credit_voucher: bool = Field(..., description="Whether to issue airline voucher instead of cash")
    refund_override_notes: str = Field(..., min_length=5, max_length=200, description="Agent audit remarks")
