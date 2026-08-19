"""Small deterministic evaluators for LabResearchBench."""

from __future__ import annotations

from typing import Any


def evaluate_case(output: dict[str, Any], expectations: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one graph output with simple deterministic checks."""
    status = _enum_value(output.get("status"))
    report = _as_dict(output.get("report"))
    evidence = output.get("evidence") or []
    resolved_professor = output.get("resolved_professor")
    publications = output.get("publications") or []

    checks = {
        "status_match": True,
        "has_professor_identity": True,
        "has_publications": True,
        "has_evidence": True,
        "has_trace": bool(output.get("trace")),
        "has_tool_logs": bool(output.get("tool_logs")),
    }

    expected_status = expectations.get("expected_status")
    if expected_status:
        checks["status_match"] = status == expected_status

    if expectations.get("requires_professor_identity"):
        checks["has_professor_identity"] = bool(resolved_professor)

    if expectations.get("requires_publications"):
        checks["has_publications"] = bool(publications)

    if expectations.get("requires_evidence"):
        checks["has_evidence"] = bool(evidence) or bool(report.get("evidence"))

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "status": status,
    }


def _enum_value(value: Any) -> str | None:
    if hasattr(value, "value"):
        return value.value
    return value


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return {}
