"""
vericall/sandbox/validator.py
-----------------------------
ExecutionValidator for VeriCall Tool Calling Sandbox.
Enforces the strict 6-state error taxonomy:
- "passed" (HTTP 200 execution, 0 schema errors)
- "invalid_json" (malformed JSON or missing "name"/"arguments")
- "wrong_endpoint" (selected decoy tool or unknown endpoint)
- "missing_arg" (omitted required parameter)
- "hallucinated_arg" (added parameter key not in schema, violating extra="forbid")
- "type_error" (wrong data type, invalid Enum value, or regex/pattern failure)
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pydantic import ValidationError
from fastapi.testclient import TestClient

from .app import ENDPOINT_REGISTRY, app

client = TestClient(app)


@dataclass
class ValidationResult:
    """Standardized verification outcome for a candidate tool call."""
    status: str  # "passed", "invalid_json", "wrong_endpoint", "missing_arg", "hallucinated_arg", "type_error"
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


class ExecutionValidator:
    """
    Validates LLM tool invocations against strict Pydantic V2 schemas and runs them
    in the execution sandbox.
    """

    def __init__(self, registry: Optional[Dict[str, Dict[str, Any]]] = None):
        self.registry = registry or ENDPOINT_REGISTRY

    def validate_call(
        self,
        raw_prediction: Any,
        expected_func: Optional[str] = None,
        candidate_tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ValidationResult:
        """
        Validates function invocation against schema and execution sandbox.

        Parameters:
        -----------
        raw_prediction: Dict or JSON string predicted by model, e.g. {"name": "...", "arguments": {...}}
                        or raw function name string if arguments passed separately.
        expected_func: Expected ground-truth tool name.
        candidate_tools: Optional list of candidate tool schema dicts available in the prompt context.

        Returns:
        --------
        ValidationResult with structured error taxonomy label.
        """
        func_name = ""
        arguments: Any = {}

        # 1. Parse JSON / Structure Check (with markdown fence & regex recovery)
        if isinstance(raw_prediction, str):
            clean_str = raw_prediction.strip()
            if clean_str.startswith("```json"):
                clean_str = clean_str[7:].strip()
            elif clean_str.startswith("```"):
                clean_str = clean_str[3:].strip()
            if clean_str.endswith("```"):
                clean_str = clean_str[:-3].strip()

            try:
                parsed = json.loads(clean_str)
            except Exception:
                import re
                match = re.search(r"\{.*\}", clean_str, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group(0))
                    except Exception as e:
                        return ValidationResult(
                            status="invalid_json",
                            is_success=False,
                            function_name="",
                            arguments={},
                            error_message=f"Malformed JSON: {str(e)}",
                        )
                else:
                    return ValidationResult(
                        status="invalid_json",
                        is_success=False,
                        function_name="",
                        arguments={},
                        error_message="No JSON object found in prediction",
                    )
        elif isinstance(raw_prediction, dict):
            parsed = raw_prediction
        else:
            return ValidationResult(
                status="invalid_json",
                is_success=False,
                function_name="",
                arguments={},
                error_message=f"Prediction must be a dict or JSON string, got {type(raw_prediction).__name__}",
            )

        # Check for standard "name" and "arguments" keys
        if not isinstance(parsed, dict) or "name" not in parsed or "arguments" not in parsed:
            # Fallback if prediction is already the arguments dict and expected_func is given
            if expected_func and isinstance(parsed, dict) and "name" not in parsed:
                func_name = expected_func
                arguments = parsed
            else:
                return ValidationResult(
                    status="invalid_json",
                    is_success=False,
                    function_name=str(parsed.get("name", "")),
                    arguments={},
                    error_message="Prediction missing required 'name' or 'arguments' keys.",
                )
        else:
            func_name = parsed.get("name", "")
            arguments = parsed.get("arguments", {})

        if not isinstance(arguments, dict):
            return ValidationResult(
                status="type_error",
                is_success=False,
                function_name=str(func_name),
                arguments={},
                error_message=f"'arguments' must be a key-value dictionary, got {type(arguments).__name__}",
            )

        # 2. Endpoint Existence & Selection Check
        if func_name not in self.registry:
            return ValidationResult(
                status="wrong_endpoint",
                is_success=False,
                function_name=func_name,
                arguments=arguments,
                error_message=f"Tool '{func_name}' is not in the allowed API schema registry.",
            )

        if expected_func and func_name != expected_func:
            return ValidationResult(
                status="wrong_endpoint",
                is_success=False,
                function_name=func_name,
                arguments=arguments,
                error_message=f"Selected wrong endpoint '{func_name}' (expected '{expected_func}').",
            )

        spec = self.registry[func_name]
        model_cls = spec["model"]
        route = spec["route"]

        # 3. Check for Hallucinated Arguments (Extra top-level keys not in model_fields)
        defined_fields = set(model_cls.model_fields.keys())
        provided_fields = set(arguments.keys())
        extra_fields = provided_fields - defined_fields
        if extra_fields:
            first_extra = sorted(list(extra_fields))[0]
            return ValidationResult(
                status="hallucinated_arg",
                is_success=False,
                function_name=func_name,
                arguments=arguments,
                error_message=f"Hallucinated argument '{first_extra}' not present in {func_name} schema.",
                error_field=first_extra,
            )

        # 4. Pydantic Model Validation (Catches missing args, type errors, enum mismatches, nested extra keys)
        try:
            validated_instance = model_cls.model_validate(arguments)
        except ValidationError as val_err:
            first_err = val_err.errors()[0]
            err_type = first_err.get("type", "")
            err_loc = ".".join(str(x) for x in first_err.get("loc", ["unknown"]))
            err_msg = first_err.get("msg", "Validation error")

            # Check if missing required parameter
            if "missing" in err_type or "missing" in err_msg.lower():
                return ValidationResult(
                    status="missing_arg",
                    is_success=False,
                    function_name=func_name,
                    arguments=arguments,
                    error_message=f"Missing required parameter '{err_loc}' in call to {func_name}.",
                    error_field=str(err_loc),
                )
            elif "extra_forbidden" in err_type or "extra" in err_msg.lower():
                return ValidationResult(
                    status="hallucinated_arg",
                    is_success=False,
                    function_name=func_name,
                    arguments=arguments,
                    error_message=f"Hallucinated argument '{err_loc}' violating extra='forbid'.",
                    error_field=str(err_loc),
                )
            else:
                return ValidationResult(
                    status="type_error",
                    is_success=False,
                    function_name=func_name,
                    arguments=arguments,
                    error_message=f"Type or constraint validation error on '{err_loc}': {err_msg}",
                    error_field=str(err_loc),
                )

        # 5. Sandbox Live Execution via TestClient
        try:
            payload = validated_instance.model_dump()
            resp = client.post(route, json=payload)
            if resp.status_code == 200:
                return ValidationResult(
                    status="passed",
                    is_success=True,
                    function_name=func_name,
                    arguments=arguments,
                    execution_response=resp.json(),
                )
            else:
                return ValidationResult(
                    status="type_error",
                    is_success=False,
                    function_name=func_name,
                    arguments=arguments,
                    error_message=f"Sandbox endpoint returned HTTP {resp.status_code}: {resp.text}",
                )
        except Exception as e:
            return ValidationResult(
                status="type_error",
                is_success=False,
                function_name=func_name,
                arguments=arguments,
                error_message=f"Execution error: {str(e)}",
            )


# Backwards compatibility alias
ToolCallValidator = ExecutionValidator
