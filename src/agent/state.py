"""LangGraph state for the Lab Research Agent."""

from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from agent.schemas import (
    ApplicantProfile,
    Evidence,
    LabComparisonReport,
    LabProfile,
    ProfessorCandidate,
    Publication,
    QualityIssue,
    ResearchRunSpec,
    ResearchTrend,
    ResolvedProfessor,
    RunStatus,
    ToolCallLog,
    TraceEvent,
)


class ResearchState(BaseModel):
    """Shared state passed between LangGraph nodes."""

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    run_id: str = Field(default_factory=lambda: f"run_{uuid4().hex[:12]}")
    user_request: str = ""
    applicant: ApplicantProfile | None = None
    run_spec: ResearchRunSpec | None = None

    research_plan: list[str] = Field(default_factory=list)
    professor_candidates: list[ProfessorCandidate] = Field(default_factory=list)
    resolved_professor: ResolvedProfessor | None = None
    publications: list[Publication] = Field(default_factory=list)
    research_trend: ResearchTrend | None = None
    lab_profile: LabProfile | None = None
    report: LabComparisonReport | None = None

    evidence: list[Evidence] = Field(default_factory=list)
    tool_call_count: int = 0
    tool_logs: list[ToolCallLog] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    quality_issues: list[QualityIssue] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    status: RunStatus = RunStatus.CREATED
