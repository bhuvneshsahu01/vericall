"""
vericall/sandbox/app.py
-----------------------
FastAPI Mock Server implementing 16 realistic endpoints across 5 domains.
Used for execution-verified post-training error mining and live validation.
"""

from typing import Any, Dict
from fastapi import FastAPI, HTTPException, status
from .models import (
    IssueRefundRequest,
    CreateInvoiceRequest,
    TransferFundsRequest,
    AuditLedgerEntriesRequest,
    QueryInventoryRequest,
    ApplyDiscountCodeRequest,
    UpdateSubscriptionRequest,
    CreateSupportTicketRequest,
    UpdateCustomerTierRequest,
    ScheduleWebhookRequest,
    ProvisionComputeNodeRequest,
    ScaleKubernetesClusterRequest,
    RotateIamCredentialsRequest,
    SearchMultiLegFlightsRequest,
    ReserveCorporateHotelRequest,
    CancelItinerarySegmentRequest,
)

app = FastAPI(
    title="VeriCall Mock API Sandbox",
    description="Execution-Verified Tool Calling API Sandbox with 16 Strict Pydantic V2 Schemas",
    version="2.0.0",
)

# Registry mapping tool function names to domain, model, description, and route
ENDPOINT_REGISTRY: Dict[str, Dict[str, Any]] = {
    # --- Training Domain: Finance ---
    "issue_refund": {
        "domain": "finance",
        "model": IssueRefundRequest,
        "description": "Process an official customer refund against a previous transaction with audit tags.",
        "route": "/finance/refund",
    },
    "create_invoice": {
        "domain": "finance",
        "model": CreateInvoiceRequest,
        "description": "Generate a new itemized corporate invoice with line items, issue date, and due date.",
        "route": "/finance/invoice",
    },
    "transfer_funds": {
        "domain": "finance",
        "model": TransferFundsRequest,
        "description": "Transfer funds between accounts via ACH, Wire, SWIFT, or SEPA networks.",
        "route": "/finance/transfer",
    },
    "audit_ledger_entries": {
        "domain": "finance",
        "model": AuditLedgerEntriesRequest,
        "description": "Query reconciled accounting ledger entries over a date range with filter flags.",
        "route": "/finance/ledger/audit",
    },

    # --- Training Domain: E-Commerce ---
    "query_inventory": {
        "domain": "commerce",
        "model": QueryInventoryRequest,
        "description": "Check current stock levels for SKU items across regional warehouse facilities.",
        "route": "/commerce/inventory",
    },
    "apply_discount_code": {
        "domain": "commerce",
        "model": ApplyDiscountCodeRequest,
        "description": "Apply a promotional discount voucher code to an active shopping cart session.",
        "route": "/commerce/discount",
    },
    "update_subscription": {
        "domain": "commerce",
        "model": UpdateSubscriptionRequest,
        "description": "Modify customer SaaS tier, recurring billing auto-renew, and optional add-on modules.",
        "route": "/commerce/subscription",
    },

    # --- Training Domain: CRM ---
    "create_support_ticket": {
        "domain": "crm",
        "model": CreateSupportTicketRequest,
        "description": "Open a categorized customer support incident with priority and client context.",
        "route": "/crm/tickets",
    },
    "update_customer_tier": {
        "domain": "crm",
        "model": UpdateCustomerTierRequest,
        "description": "Promote or reclassify customer loyalty tier with account rep override reasoning.",
        "route": "/crm/customers/tier",
    },
    "schedule_webhook": {
        "domain": "crm",
        "model": ScheduleWebhookRequest,
        "description": "Register an HTTPS webhook callback with event subscriptions, HMAC secret, and headers.",
        "route": "/crm/webhooks",
    },

    # --- Unseen Evaluation Domain: Cloud Infrastructure ---
    "provision_compute_node": {
        "domain": "infra",
        "model": ProvisionComputeNodeRequest,
        "description": "Spin up virtual cloud worker nodes with specific instance tier, region, and tags.",
        "route": "/infra/nodes/provision",
    },
    "scale_kubernetes_cluster": {
        "domain": "infra",
        "model": ScaleKubernetesClusterRequest,
        "description": "Scale node pool replicas for a Kubernetes cluster using specified surge/drain strategy.",
        "route": "/infra/k8s/scale",
    },
    "rotate_iam_credentials": {
        "domain": "infra",
        "model": RotateIamCredentialsRequest,
        "description": "Rotate API tokens, OAuth keys, or certificates for a service account principal.",
        "route": "/infra/iam/rotate",
    },

    # --- Unseen Evaluation Domain: Travel Logistics ---
    "search_multi_leg_flights": {
        "domain": "travel",
        "model": SearchMultiLegFlightsRequest,
        "description": "Search available multi-segment flight itineraries across passenger cabin classes.",
        "route": "/travel/flights/search",
    },
    "reserve_corporate_hotel": {
        "domain": "travel",
        "model": ReserveCorporateHotelRequest,
        "description": "Book corporate lodging using company billing code and registered guest manifests.",
        "route": "/travel/hotels/reserve",
    },
    "cancel_itinerary_segment": {
        "domain": "travel",
        "model": CancelItinerarySegmentRequest,
        "description": "Cancel a specific itinerary leg from a PNR booking locator with reason codes.",
        "route": "/travel/itinerary/cancel",
    },
}


def get_tool_json_schema(name: str) -> Dict[str, Any]:
    """Generates a standardized OpenAPI/JSON schema block for a tool."""
    if name not in ENDPOINT_REGISTRY:
        raise KeyError(f"Tool '{name}' not found in ENDPOINT_REGISTRY.")
    meta = ENDPOINT_REGISTRY[name]
    model_cls = meta["model"]
    raw_schema = model_cls.model_json_schema()
    
    # Strip unnecessary title metadata for compact prompt injection
    props = raw_schema.get("properties", {})
    reqs = raw_schema.get("required", [])
    defs = raw_schema.get("$defs", {})

    params_obj = {
        "type": "object",
        "properties": props,
        "required": reqs,
        "additionalProperties": False,
    }
    if defs:
        params_obj["$defs"] = defs

    return {
        "name": name,
        "description": meta["description"],
        "parameters": params_obj,
    }


# ============================================================================
# API ROUTE HANDLERS (HTTP 200 Mock Execution)
# ============================================================================

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "vericall-sandbox", "endpoints": len(ENDPOINT_REGISTRY)}


# --- Finance ---
@app.post("/finance/refund")
def handle_refund(req: IssueRefundRequest):
    return {
        "status": "success",
        "refund_id": f"ref_{req.transaction_id[-8:]}",
        "processed_amount": req.amount,
        "currency": req.currency,
        "reason": req.reason,
    }

@app.post("/finance/invoice")
def handle_invoice(req: CreateInvoiceRequest):
    total = sum(item.unit_price * item.quantity for item in req.line_items)
    return {
        "status": "success",
        "invoice_id": f"inv_{req.client_id}_{len(req.line_items)}",
        "total_amount": round(total, 2),
        "currency": req.currency,
        "due_date": req.due_date,
    }

@app.post("/finance/transfer")
def handle_transfer(req: TransferFundsRequest):
    return {
        "status": "success",
        "transfer_id": f"xfer_{req.source_account[-4:]}_to_{req.destination_account[-4:]}",
        "amount": req.amount,
        "network": req.transfer_method,
    }

@app.post("/finance/ledger/audit")
def handle_ledger_audit(req: AuditLedgerEntriesRequest):
    return {
        "status": "success",
        "ledger": req.ledger_id,
        "entry_count": 42,
        "entry_type": req.entry_type,
        "window": f"{req.start_date} to {req.end_date}",
    }

# --- E-Commerce ---
@app.post("/commerce/inventory")
def handle_inventory(req: QueryInventoryRequest):
    return {
        "status": "success",
        "warehouse": req.warehouse_id,
        "inventory": {sku: 15 for sku in req.sku_list},
        "channel": req.fulfillment_channel,
    }

@app.post("/commerce/discount")
def handle_discount(req: ApplyDiscountCodeRequest):
    return {
        "status": "success",
        "cart_id": req.cart_id,
        "promo_code": req.promo_code,
        "discount_applied_pct": 15.0,
        "channel": req.channel,
    }

@app.post("/commerce/subscription")
def handle_subscription(req: UpdateSubscriptionRequest):
    return {
        "status": "success",
        "subscription_id": req.subscription_id,
        "tier": req.plan_tier,
        "auto_renew": req.auto_renew,
        "effective": req.effective_date,
    }

# --- CRM ---
@app.post("/crm/tickets")
def handle_ticket(req: CreateSupportTicketRequest):
    return {
        "status": "success",
        "ticket_id": f"tkt_{req.customer_id[-4:]}_01",
        "priority": req.priority,
        "department": req.department,
    }

@app.post("/crm/customers/tier")
def handle_customer_tier(req: UpdateCustomerTierRequest):
    return {
        "status": "success",
        "customer_id": req.customer_id,
        "new_tier": req.target_tier,
        "representative": req.assigned_rep_id,
    }

@app.post("/crm/webhooks")
def handle_webhook(req: ScheduleWebhookRequest):
    return {
        "status": "success",
        "target": req.destination_url,
        "events": req.subscribed_events,
        "active": True,
    }

# --- Infrastructure ---
@app.post("/infra/nodes/provision")
def handle_provision(req: ProvisionComputeNodeRequest):
    return {
        "status": "success",
        "batch_id": f"node_{req.cloud_region}_{req.cluster_count}",
        "nodes": [f"i-{req.instance_tier[:3]}-{i:02d}" for i in range(req.cluster_count)],
        "env": req.environment_tag,
    }

@app.post("/infra/k8s/scale")
def handle_scale_k8s(req: ScaleKubernetesClusterRequest):
    return {
        "status": "success",
        "cluster": req.cluster_name,
        "pool": req.node_pool,
        "replicas": req.desired_replicas,
        "strategy": req.scaling_strategy,
    }

@app.post("/infra/iam/rotate")
def handle_rotate_iam(req: RotateIamCredentialsRequest):
    return {
        "status": "success",
        "principal": req.service_account_id,
        "credential_type": req.credential_type,
        "grace_hours": req.grace_period_hours,
        "revoked_immediate": req.force_revocation,
    }

# --- Travel ---
@app.post("/travel/flights/search")
def handle_search_flights(req: SearchMultiLegFlightsRequest):
    return {
        "status": "success",
        "legs": len(req.itinerary_segments),
        "cabin": req.cabin_class,
        "passengers": req.passenger_count,
        "matches_found": 8,
    }

@app.post("/travel/hotels/reserve")
def handle_hotel_reserve(req: ReserveCorporateHotelRequest):
    return {
        "status": "success",
        "booking_id": f"corp_htl_{req.property_code[-4:]}",
        "room": req.room_category,
        "guests": len(req.guest_manifest),
        "billed_to": req.corporate_billing_code,
    }

@app.post("/travel/itinerary/cancel")
def handle_cancel_itinerary(req: CancelItinerarySegmentRequest):
    return {
        "status": "success",
        "pnr": req.booking_reference,
        "segment": req.segment_id,
        "voucher_issued": req.request_credit_voucher,
        "confirmed": True,
    }
