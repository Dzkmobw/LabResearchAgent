"""Deterministic quality checks for generated lab reports."""

from __future__ import annotations

from agent.schemas import Evidence, LabProfile, QualityIssue, ResearchRunSpec, SourceType


def evaluate_lab_profile(
    profile: LabProfile | None,
    evidence: list[Evidence],
    run_spec: ResearchRunSpec,
) -> list[QualityIssue]:
    """Return structured issues found in a lab profile."""
    issues: list[QualityIssue] = []

    if profile is None:
        return [
            QualityIssue(
                code="missing_lab_profile",
                severity="error",
                message="No lab profile was generated.",
                suggested_action="Run discovery and report generation again.",
            )
        ]

    if profile.professor is None:
        issues.append(
            QualityIssue(
                code="missing_professor_identity",
                severity="error",
                message="Professor identity was not resolved.",
                suggested_action="Search academic and official sources before reporting.",
            )
        )
    elif profile.professor.is_ambiguous:
        issues.append(
            QualityIssue(
                code="professor_identity_ambiguous",
                severity="warning",
                message="Professor identity has low confidence or competing candidates.",
                suggested_action="Ask the user to confirm the professor identity.",
            )
        )

    if len(evidence) < run_spec.min_sources_per_lab:
        issues.append(
            QualityIssue(
                code="insufficient_sources",
                severity="warning",
                message="The profile has fewer evidence records than required.",
                suggested_action="Collect more official or academic sources.",
            )
        )

    if not profile.publications:
        issues.append(
            QualityIssue(
                code="missing_publications",
                severity="warning",
                message="No recent publications were attached to the profile.",
                suggested_action="Query OpenAlex or Semantic Scholar for recent works.",
            )
        )

    if run_spec.require_official_admission_source:
        has_official_admission = any(
            ev.source_type == SourceType.OFFICIAL
            and ("admission" in ev.claim.lower() or "招生" in ev.claim)
            for ev in evidence
        )
        if not has_official_admission:
            issues.append(
                QualityIssue(
                    code="missing_official_admission_source",
                    severity="warning",
                    message="No official admission evidence was collected.",
                    suggested_action="Search school or lab official admission pages.",
                )
            )

    unsupported_claims = [
        ev for ev in evidence if not ev.source_url and not ev.is_inference
    ]
    if unsupported_claims:
        issues.append(
            QualityIssue(
                code="unsupported_evidence",
                severity="error",
                message="Some non-inferred evidence lacks a source URL.",
                suggested_action="Attach source URLs or mark the item as inference.",
            )
        )

    return issues
