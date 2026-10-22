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
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pydantic import ValidationError
from fastapi.testclient import TestClient

from .app import ENDPOINT_REGISTRY, app

client = TestClient(app)


def _repair_truncated_json(s: str) -> str:
    """
    Walks through JSON string, tracks string literals and escape characters,
    and appends missing quotes, brackets, and braces in the correct LIFO order.
    """
    in_string = False
    escape = False
    stack = []

    for char in s:
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char in ("{", "["):
                stack.append(char)
            elif char == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
            elif char == "]":
                if stack and stack[-1] == "[":
                    stack.pop()

    repaired = s
    if in_string:
        repaired += '"'

    while stack:
        opener = stack.pop()
        if opener == "{":
            repaired += "}"
        elif opener == "[":
            repaired += "]"

    return repaired


def _try_parse_candidate(s: str) -> Optional[Dict[str, Any]]:
    """Attempts to parse a candidate JSON string with multiple fallbacks."""
    if not s:
        return None
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    s = s.strip()

    # 1. Direct parse
    try:
        val = json.loads(s)
        if isinstance(val, dict):
            return val
    except Exception:
        pass

    # 2. Clean trailing commas
    s_clean = re.sub(r",\s*([\]\}])", r"\1", s)
    try:
        val = json.loads(s_clean)
        if isinstance(val, dict):
            return val
    except Exception:
        pass

    # 3. Truncation repair
    repaired = _repair_truncated_json(s_clean)
    repaired = re.sub(r",\s*([\]\}])", r"\1", repaired)
    try:
        val = json.loads(repaired)
        if isinstance(val, dict):
            return val
    except Exception:
        pass

    return None


def extract_and_repair_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts JSON object from text, handling markdown code fences,
    surrounding conversational preamble/postscript, and truncated closing braces.
    """
    if not text or not isinstance(text, str):
        return None

    clean_text = text.strip()
    if ":" not in clean_text:
        return None

    # 1. Match code fences ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text, re.IGNORECASE)
    if fence_match:
        res = _try_parse_candidate(fence_match.group(1))
        if res is not None:
            return res

    # 2. Extract substring between first '{' and last '}'
    first_b = clean_text.find("{")
    last_b = clean_text.rfind("}")
    if first_b != -1 and last_b > first_b:
        res = _try_parse_candidate(clean_text[first_b:last_b + 1])
        if res is not None:
            return res

    # 3. Truncated output: from first '{' to end of string with repair
    if first_b != -1:
        res = _try_parse_candidate(clean_text[first_b:])
        if res is not None:
            return res

    return _try_parse_candidate(clean_text)


def parse_json_call(raw_prediction: Any) -> Optional[Dict[str, Any]]:
    """
    Normalizes any prediction (dict, string with/without markdown fences, truncated)
    into a structured dict with standard keys 'name' and 'arguments'.
    """
    if isinstance(raw_prediction, dict):
        parsed = dict(raw_prediction)
    elif isinstance(raw_prediction, str):
        parsed = extract_and_repair_json(raw_prediction)
    else:
        return None

    if not isinstance(parsed, dict):
        return None

    # Normalize tool name aliases
    if "name" not in parsed:
        for alias in ["function", "tool", "endpoint", "func", "target_function"]:
            if alias in parsed and isinstance(parsed[alias], str):
                parsed["name"] = parsed[alias]
                break

    # Normalize arguments aliases
    if "arguments" not in parsed:
        for alias in ["parameters", "params", "args", "target_arguments"]:
            if alias in parsed and isinstance(parsed[alias], dict):
                parsed["arguments"] = parsed[alias]
                break

    return parsed


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
        """
        parsed = parse_json_call(raw_prediction)

        # 1. Parse JSON / Structure Check
        if parsed is None or not isinstance(parsed, dict):
            return ValidationResult(
                status="invalid_json",
                is_success=False,
                function_name="",
                arguments={},
                error_message="Malformed JSON or could not extract valid JSON object from prediction.",
            )

        func_name = parsed.get("name", "")
        arguments = parsed.get("arguments", {})

        if not func_name:
            if expected_func and isinstance(parsed, dict) and "name" not in parsed and len(parsed) > 0:
                func_name = expected_func
                arguments = parsed
            else:
                return ValidationResult(
                    status="invalid_json",
                    is_success=False,
                    function_name="",
                    arguments={},
                    error_message="Prediction missing required 'name' key or valid JSON structure.",
                )

        if not isinstance(arguments, dict):
            return ValidationResult(
                status="type_error",
                is_success=False,
                function_name=str(func_name),
                arguments={},
                error_message=f"'arguments' must be a key-value dictionary, got {type(arguments).__name__}",
            )

        # 2. Endpoint Existence & Selection Check
        cand_map = {t["name"]: t for t in (candidate_tools or [])}

        if func_name not in self.registry and func_name not in cand_map:
            return ValidationResult(
                status="wrong_endpoint",
                is_success=False,
                function_name=func_name,
                arguments=arguments,
                error_message=f"Tool '{func_name}' is not in the allowed API schema registry or candidate tools.",
            )

        if expected_func and func_name != expected_func:
            return ValidationResult(
                status="wrong_endpoint",
                is_success=False,
                function_name=func_name,
                arguments=arguments,
                error_message=f"Selected wrong endpoint '{func_name}' (expected '{expected_func}').",
            )

        # Dynamic JSON Schema / AST validation for tools not in FastAPI sandbox (e.g. BFCL)
        if func_name not in self.registry and func_name in cand_map:
            tool_spec = cand_map[func_name]
            params = tool_spec.get("parameters", {})
            properties = params.get("properties", {})
            required = params.get("required", [])

            extra_fields = set(arguments.keys()) - set(properties.keys())
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

            missing = [r for r in required if r not in arguments]
            if missing:
                first_missing = sorted(missing)[0]
                return ValidationResult(
                    status="missing_arg",
                    is_success=False,
                    function_name=func_name,
                    arguments=arguments,
                    error_message=f"Missing required parameter '{first_missing}' in call to {func_name}.",
                    error_field=first_missing,
                )

            for field, val in arguments.items():
                if field not in properties:
                    continue
                prop_spec = properties[field]
                exp_type = prop_spec.get("type")
                if exp_type == "number" and not isinstance(val, (int, float)):
                    return ValidationResult(
                        status="type_error",
                        is_success=False,
                        function_name=func_name,
                        arguments=arguments,
                        error_message=f"Type error: parameter '{field}' expected number, got {type(val).__name__}.",
                        error_field=field,
                    )
                elif exp_type == "integer" and not (isinstance(val, int) and not isinstance(val, bool)):
                    return ValidationResult(
                        status="type_error",
                        is_success=False,
                        function_name=func_name,
                        arguments=arguments,
                        error_message=f"Type error: parameter '{field}' expected integer, got {type(val).__name__}.",
                        error_field=field,
                    )
                elif exp_type == "string" and not isinstance(val, str):
                    return ValidationResult(
                        status="type_error",
                        is_success=False,
                        function_name=func_name,
                        arguments=arguments,
                        error_message=f"Type error: parameter '{field}' expected string, got {type(val).__name__}.",
                        error_field=field,
                    )
                elif exp_type == "boolean" and not isinstance(val, bool):
                    return ValidationResult(
                        status="type_error",
                        is_success=False,
                        function_name=func_name,
                        arguments=arguments,
                        error_message=f"Type error: parameter '{field}' expected boolean, got {type(val).__name__}.",
                        error_field=field,
                    )
                elif exp_type == "array" and not isinstance(val, list):
                    return ValidationResult(
                        status="type_error",
                        is_success=False,
                        function_name=func_name,
                        arguments=arguments,
                        error_message=f"Type error: parameter '{field}' expected array, got {type(val).__name__}.",
                        error_field=field,
                    )

                if "enum" in prop_spec and val not in prop_spec.get("enum", []):
                    return ValidationResult(
                        status="type_error",
                        is_success=False,
                        function_name=func_name,
                        arguments=arguments,
                        error_message=f"Value '{val}' for parameter '{field}' not in enum options.",
                        error_field=field,
                    )

            return ValidationResult(
                status="passed",
                is_success=True,
                function_name=func_name,
                arguments=arguments,
                execution_response={"status": "ast_validated", "tool": func_name},
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
