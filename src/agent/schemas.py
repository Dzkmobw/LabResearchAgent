"""Structured models for the Lab Research Agent."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(timezone.utc)


class RunStatus(str, Enum):
    """High-level state of a research run."""

    CREATED = "created"
    NEEDS_CLARIFICATION = "needs_clarification"
    PLANNED = "planned"
    SEARCHING = "searching"
    RESOLVED = "resolved"
    NEEDS_REVIEW = "needs_review"
    PARTIAL = "partial"
    COMPLETE = "complete"
    FAILED = "failed"


class SourceType(str, Enum):
    """Source categories used by evidence records."""

    OFFICIAL = "official"
    ACADEMIC_API = "academic_api"
    WEB_SEARCH = "web_search"
    WEB_PAGE = "web_page"
    PAPER = "paper"
    AGGREGATOR = "aggregator"
    MODEL_INFERENCE = "model_inference"
    SYSTEM = "system"


class ApplicantProfile(BaseModel):
    """Applicant background used for match analysis."""

    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    degree_target: Literal["master", "phd"] = "phd"
    research_interests: list[str] = Field(default_factory=list)
    background_summary: str | None = None
    target_country: str = "Japan"
    target_schools: list[str] = Field(default_factory=list)
    enrollment_term: str | None = None


class ResearchRunSpec(BaseModel):
    """Validated task contract for one research run."""

    model_config = ConfigDict(extra="ignore")

    target_country: str
    degree: Literal["master", "phd"]
    research_interests: list[str] = Field(min_length=1)
    target_schools: list[str] = Field(default_factory=list)
    target_professor: str | None = None
    target_lab: str | None = None
    lab_count: int = Field(default=1, ge=1, le=10)
    publication_years: int = Field(default=5, ge=1, le=10)
    max_tool_calls: int = Field(default=8, ge=1, le=50)
    max_runtime_seconds: int = Field(default=120, ge=5, le=1200)
    tool_timeout_seconds: int = Field(default=12, ge=1, le=120)
    max_retries: int = Field(default=1, ge=0, le=5)
    min_sources_per_lab: int = Field(default=2, ge=1, le=10)
    require_official_admission_source: bool = True

    @model_validator(mode="after")
    def require_research_target(self) -> "ResearchRunSpec":
        if not self.target_professor and not self.target_lab and not self.target_schools:
            msg = "At least one of target_professor, target_lab, or target_schools is required."
            raise ValueError(msg)
        return self


class Evidence(BaseModel):
    """One source-backed or explicitly inferred fact."""

    model_config = ConfigDict(extra="ignore")

    evidence_id: str = Field(default_factory=lambda: f"ev_{uuid4().hex[:12]}")
    claim: str
    source_url: str | None = None
    source_title: str | None = None
    source_type: SourceType = SourceType.SYSTEM
    retrieved_at: datetime = Field(default_factory=utc_now)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    is_inference: bool = False
    supports: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProfessorCandidate(BaseModel):
    """A possible professor identity from search or academic APIs."""

    model_config = ConfigDict(extra="ignore")

    candidate_id: str
    display_name: str
    alternative_names: list[str] = Field(default_factory=list)
    affiliations: list[str] = Field(default_factory=list)
    homepage_url: str | None = None
    source_ids: dict[str, str] = Field(default_factory=dict)
    topics: list[str] = Field(default_factory=list)
    works_count: int | None = None
    cited_by_count: int | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class ResolvedProfessor(BaseModel):
    """A selected professor identity with confidence and notes."""

    model_config = ConfigDict(extra="ignore")

    display_name: str
    openalex_id: str | None = None
    affiliation: str | None = None
    homepage_url: str | None = None
    normalized_names: list[str] = Field(default_factory=list)
    identity_confidence: float = Field(ge=0.0, le=1.0)
    is_ambiguous: bool = False
    disambiguation_notes: list[str] = Field(default_factory=list)
    source_evidence_ids: list[str] = Field(default_factory=list)


class Publication(BaseModel):
    """Normalized publication record."""

    model_config = ConfigDict(extra="ignore")

    publication_id: str
    title: str
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    cited_by_count: int | None = None
    url: str | None = None
    authors: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    source_evidence_ids: list[str] = Field(default_factory=list)


class ResearchTrend(BaseModel):
    """Compact trend summary from recent publications."""

    model_config = ConfigDict(extra="ignore")

    recent_years: int
    top_keywords: list[str] = Field(default_factory=list)
    matched_interests: list[str] = Field(default_factory=list)
    summary: str
    evidence_ids: list[str] = Field(default_factory=list)


class AdmissionRequirement(BaseModel):
    """Admission-related requirement extracted from official or web sources."""

    model_config = ConfigDict(extra="ignore")

    requirement_type: str
    value: str
    source_evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    is_missing: bool = False


class LabProfile(BaseModel):
    """One lab/professor profile used in the report."""

    model_config = ConfigDict(extra="ignore")

    lab_name: str | None = None
    professor: ResolvedProfessor | None = None
    publications: list[Publication] = Field(default_factory=list)
    research_trend: ResearchTrend | None = None
    admission_requirements: list[AdmissionRequirement] = Field(default_factory=list)
    match_score: float = Field(default=0.0, ge=0.0, le=1.0)
    match_rationale: str = ""
    missing_information: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    status: RunStatus = RunStatus.PARTIAL


class QualityIssue(BaseModel):
    """Structured quality gate finding."""

    code: str
    severity: Literal["info", "warning", "error"]
    message: str
    suggested_action: str


class ToolResult(BaseModel):
    """Standardized output from any tool behind the gateway."""

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    tool_name: str
    status: Literal["ok", "skipped", "error"] = "ok"
    items: list[Any] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    message: str | None = None
    error_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCallLog(BaseModel):
    """One normalized tool call trace event."""

    model_config = ConfigDict(extra="ignore")

    tool_name: str
    arguments_summary: dict[str, Any] = Field(default_factory=dict)
    status: Literal["ok", "skipped", "error"]
    attempts: int = 0
    duration_ms: float = 0.0
    cache_hit: bool = False
    error_type: str | None = None
    message: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class TraceEvent(BaseModel):
    """Node-level trace event stored in graph state."""

    event_id: str = Field(default_factory=lambda: f"tr_{uuid4().hex[:12]}")
    node: str
    status: str
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class LabComparisonReport(BaseModel):
    """Final structured report for one or more lab profiles."""

    model_config = ConfigDict(extra="ignore")

    run_id: str
    title: str
    status: RunStatus
    generated_at: datetime = Field(default_factory=utc_now)
    executive_summary: str
    profiles: list[LabProfile] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    quality_issues: list[QualityIssue] = Field(default_factory=list)
    trace_summary: list[ToolCallLog] = Field(default_factory=list)
