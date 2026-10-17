"""
ToolCallValidator: Execution-Verified Tool Calling Analysis.
Inspects LLM function calls against strict Pydantic schemas and returns
mutually exclusive error taxonomy labels:
- success
- wrong_endpoint
- missing_arg
- hallucinated_arg
- type_error
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pydantic import ValidationError

from .app import ENDPOINT_REGISTRY, app
from fastapi.testclient import TestClient

client = TestClient(app)


@dataclass
class ValidationResult:
    """Standardized verification outcome for a candidate tool call."""
    status: str  # "success", "wrong_endpoint", "missing_arg", "hallucinated_arg", "type_error"
    is_success: bool
    function_name: str
    arguments: Dict[str, Any]
    error_message: str = ""
    error_field: Optional[str] = None
    execution_response: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "is_success": self.is_success,
            "function_name": self.function_name,
            "arguments": self.arguments,
            "error_message": self.error_message,
            "error_field": self.error_field,
            "execution_response": self.execution_response,
        }


class ToolCallValidator:
    """
    Validates LLM tool invocations against schema definitions and runs them
    in the execution sandbox.
    """

    def __init__(self, registry: Optional[Dict[str, Dict[str, Any]]] = None):
        self.registry = registry or ENDPOINT_REGISTRY

    def validate_call(
        self,
        func_name: str,
        arguments: Any,
        expected_func: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validates function invocation.

        Parameters:
        -----------
        func_name: Candidate function name predicted by the model.
        arguments: Candidate dictionary or JSON string of arguments.
        expected_func: Optional ground-truth function name for endpoint validation.

        Returns:
        --------
        ValidationResult with structured error taxonomy label.
        """
        # Parse arguments if passed as JSON string
        parsed_args = arguments
        if isinstance(arguments, str):
            try:
                parsed_args = json.loads(arguments)
            except Exception as e:
                return ValidationResult(
                    status="type_error",
                    is_success=False,
                    function_name=func_name,
                    arguments={},
                    error_message=f"Invalid JSON in arguments: {str(e)}",
                )

        if not isinstance(parsed_args, dict):
            return ValidationResult(
                status="type_error",
                is_success=False,
                function_name=func_name,
                arguments={"raw": str(arguments)},
                error_message=f"Arguments must be a key-value dictionary, got {type(parsed_args).__name__}",
            )

        # 1. Endpoint Existence & Selection Check
        if func_name not in self.registry:
            return ValidationResult(
                status="wrong_endpoint",
                is_success=False,
                function_name=func_name,
                arguments=parsed_args,
                error_message=f"Function '{func_name}' is not in the allowed API schema registry.",
            )

        if expected_func and func_name != expected_func:
            return ValidationResult(
                status="wrong_endpoint",
                is_success=False,
                function_name=func_name,
                arguments=parsed_args,
                error_message=f"Selected wrong endpoint '{func_name}' (expected '{expected_func}').",
            )

        spec = self.registry[func_name]
        model_cls = spec["model"]

        # 2. Check for Hallucinated Arguments (Extra keys not defined in Pydantic schema)
        defined_fields = set(model_cls.model_fields.keys())
        provided_fields = set(parsed_args.keys())
        extra_fields = provided_fields - defined_fields
        if extra_fields:
            first_extra = sorted(list(extra_fields))[0]
            return ValidationResult(
                status="hallucinated_arg",
                is_success=False,
                function_name=func_name,
                arguments=parsed_args,
                error_message=f"Hallucinated argument '{first_extra}' not present in {func_name} schema.",
                error_field=first_extra,
            )

        # 3. Pydantic Model Validation
        try:
            validated_instance = model_cls.model_validate(parsed_args)
        except ValidationError as val_err:
            first_err = val_err.errors()[0]
            err_type = first_err.get("type", "")
            err_loc = first_err.get("loc", ["unknown"])[0]
            err_msg = first_err.get("msg", "Validation error")

            # Check if missing required parameter
            if "missing" in err_type or "missing" in err_msg.lower():
                return ValidationResult(
                    status="missing_arg",
                    is_success=False,
                    function_name=func_name,
                    arguments=parsed_args,
                    error_message=f"Missing required parameter '{err_loc}' in call to {func_name}.",
                    error_field=str(err_loc),
                )
            elif "extra_forbidden" in err_type:
                return ValidationResult(
                    status="hallucinated_arg",
                    is_success=False,
                    function_name=func_name,
                    arguments=parsed_args,
                    error_message=f"Hallucinated argument '{err_loc}'.",
                    error_field=str(err_loc),
                )
            else:
                return ValidationResult(
                    status="type_error",
                    is_success=False,
                    function_name=func_name,
                    arguments=parsed_args,
                    error_message=f"Type or constraint validation error on '{err_loc}': {err_msg}",
                    error_field=str(err_loc),
                )

        # 4. Sandbox Execution
        # Map function name to FastAPI endpoint route
        endpoint_routes = {
            "get_stock_quote": "/finance/stock-quote",
            "execute_trade": "/finance/trade",
            "get_account_balance": "/finance/balance",
            "search_products": "/commerce/products/search",
            "create_order": "/commerce/orders/create",
            "track_shipment": "/commerce/shipments/track",
            "provision_server": "/infra/servers/provision",
            "restart_service": "/infra/services/restart",
            "get_cluster_metrics": "/infra/clusters/metrics",
            "search_flights": "/travel/flights/search",
            "book_hotel": "/travel/hotels/book",
            "cancel_booking": "/travel/bookings/cancel",
        }

        route = endpoint_routes.get(func_name)
        exec_resp = None
        if route:
            resp = client.post(route, json=validated_instance.model_dump())
            if resp.status_code == 200:
                exec_resp = resp.json()
            else:
                return ValidationResult(
                    status="type_error",
                    is_success=False,
                    function_name=func_name,
                    arguments=parsed_args,
                    error_message=f"Sandbox endpoint returned HTTP {resp.status_code}: {resp.text}",
                )

        return ValidationResult(
            status="success",
            is_success=True,
            function_name=func_name,
            arguments=parsed_args,
            execution_response=exec_resp,
        )
