"""LangGraph workflow for the Lab Research Agent MVP."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import re
from typing import Any, Dict

from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from typing_extensions import TypedDict

from agent.harness.quality_gate import evaluate_lab_profile
from agent.harness.telemetry import node_event
from agent.harness.tool_gateway import ToolGateway
from agent.llm import draft_report_text, infer_research_run_spec
from agent.schemas import (
    AdmissionRequirement,
    Evidence,
    LabComparisonReport,
    LabProfile,
    ProfessorCandidate,
    Publication,
    ResearchRunSpec,
    ResearchTrend,
    ResolvedProfessor,
    RunStatus,
    SourceType,
)
from agent.state import ResearchState
from agent.tools.mcp_web_search import MCPWebSearchTool
from agent.tools.openalex import OpenAlexClient


class Context(TypedDict, total=False):
    """Runtime context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    enable_mcp_search: bool


@traceable(name="clarify_requirements")
async def clarify_requirements(state: ResearchState) -> Dict[str, Any]:
    """Validate the task contract before the agent starts researching."""
    if state.run_spec is None:
        if state.user_request:
            parsed_spec, missing, error = await infer_research_run_spec(state.user_request)
            if parsed_spec is not None:
                return {
                    "run_spec": parsed_spec,
                    "status": RunStatus.PLANNED,
                    "missing_fields": [],
                    "trace": state.trace
                    + [
                        node_event(
                            "clarify_requirements",
                            "ok",
                            "DeepSeek parsed user_request into ResearchRunSpec.",
                            model_provider="deepseek",
                            model_task="request_parsing",
                        )
                    ],
                }

            missing_fields = missing or ["run_spec"]
            return {
                "status": RunStatus.NEEDS_CLARIFICATION,
                "missing_fields": missing_fields,
                "errors": state.errors + ([error] if error else []),
                "trace": state.trace
                + [
                    node_event(
                        "clarify_requirements",
                        "needs_clarification",
                        "Could not parse a complete ResearchRunSpec from user_request.",
                        missing_fields=missing_fields,
                        model_provider="deepseek",
                        error=error,
                    )
                ],
            }

        missing = [
            "run_spec.target_country",
            "run_spec.degree",
            "run_spec.research_interests",
            "run_spec.target_professor or run_spec.target_lab or run_spec.target_schools",
        ]
        return {
            "status": RunStatus.NEEDS_CLARIFICATION,
            "missing_fields": missing,
            "trace": state.trace
            + [
                node_event(
                    "clarify_requirements",
                    "needs_clarification",
                    "Missing ResearchRunSpec.",
                    missing_fields=missing,
                )
            ],
        }

    return {
        "status": RunStatus.PLANNED,
        "missing_fields": [],
        "trace": state.trace
        + [
            node_event(
                "clarify_requirements",
                "ok",
                "ResearchRunSpec is valid.",
                target_country=state.run_spec.target_country,
                degree=state.run_spec.degree,
            )
        ],
    }


@traceable(name="generate_research_plan")
async def generate_research_plan(state: ResearchState) -> Dict[str, Any]:
    """Create a deterministic research plan for the run."""
    spec = state.run_spec
    if spec is None:
        return {"status": RunStatus.NEEDS_CLARIFICATION}

    target = spec.target_professor or spec.target_lab or ", ".join(spec.target_schools)
    plan = [
        f"Confirm research target: {target}",
        "Search academic author candidates from OpenAlex.",
        "Search official/lab web sources through the MCP web search slot.",
        "Resolve professor identity using name, affiliation, source ids, and topics.",
        f"Fetch recent publications from the last {spec.publication_years} years.",
        "Compare publication topics with applicant research interests.",
        "Generate a source-backed lab profile and run the quality gate.",
    ]
    return {
        "research_plan": plan,
        "status": RunStatus.PLANNED,
        "trace": state.trace
        + [
            node_event(
                "generate_research_plan",
                "ok",
                "Created a deterministic research plan.",
                steps=len(plan),
            )
        ],
    }


@traceable(name="discover_professors")
async def discover_professors(state: ResearchState) -> Dict[str, Any]:
    """Use OpenAlex to discover professor candidates."""
    spec = state.run_spec
    if spec is None:
        return {"status": RunStatus.NEEDS_CLARIFICATION}

    query = spec.target_professor or spec.target_lab or " ".join(spec.research_interests)
    gateway = _gateway_for(state, spec)
    openalex = OpenAlexClient()
    gateway.register("openalex.search_authors", openalex.search_authors)
    call = await gateway.call(
        "openalex.search_authors",
        {
            "query": query,
            "target_schools": spec.target_schools,
            "research_interests": spec.research_interests,
            "limit": max(3, spec.lab_count * 5),
        },
    )
    candidates = [
        item for item in call.result.items if isinstance(item, ProfessorCandidate)
    ]
    errors = state.errors[:]
    if call.result.status == "error":
        errors.append(call.result.message or "OpenAlex author search failed.")

    return {
        "professor_candidates": candidates,
        "evidence": state.evidence + call.result.evidence,
        "tool_logs": state.tool_logs + [call.log],
        "tool_call_count": call.next_tool_call_count,
        "errors": errors,
        "status": RunStatus.SEARCHING if candidates else RunStatus.PARTIAL,
        "trace": state.trace
        + [
            node_event(
                "discover_professors",
                call.result.status,
                f"Discovered {len(candidates)} professor candidates.",
                query=query,
            )
        ],
    }


@traceable(name="discover_web_sources")
async def discover_web_sources(state: ResearchState) -> Dict[str, Any]:
    """Use an optional MCP web search tool to find official/lab pages."""
    spec = state.run_spec
    if spec is None:
        return {"status": RunStatus.NEEDS_CLARIFICATION}

    gateway = _gateway_for(state, spec)
    mcp_search = MCPWebSearchTool()
    gateway.register("mcp.web_search", mcp_search.search)
    target = spec.target_professor or spec.target_lab or ", ".join(spec.target_schools)
    query = (
        f"{target} {' '.join(spec.target_schools)} laboratory admission professor "
        f"{spec.target_country}"
    ).strip()

    call = await gateway.call("mcp.web_search", {"query": query, "limit": 5})
    errors = state.errors[:]
    if call.result.status == "error":
        errors.append(call.result.message or "MCP web search failed.")

    return {
        "evidence": state.evidence + call.result.evidence,
        "tool_logs": state.tool_logs + [call.log],
        "tool_call_count": call.next_tool_call_count,
        "errors": errors,
        "trace": state.trace
        + [
            node_event(
                "discover_web_sources",
                call.result.status,
                call.result.message or "MCP web search completed.",
                query=query,
            )
        ],
    }


@traceable(name="resolve_professor_identity")
async def resolve_professor_identity(state: ResearchState) -> Dict[str, Any]:
    """Resolve one professor identity from candidates using deterministic signals."""
    spec = state.run_spec
    if spec is None:
        return {"status": RunStatus.NEEDS_CLARIFICATION}
    if not state.professor_candidates:
        return {
            "status": RunStatus.PARTIAL,
            "trace": state.trace
            + [
                node_event(
                    "resolve_professor_identity",
                    "partial",
                    "No professor candidates were available for disambiguation.",
                )
            ],
        }

    scored = [
        (_score_candidate(candidate, spec), candidate)
        for candidate in state.professor_candidates
    ]
    scored.sort(key=lambda row: row[0], reverse=True)
    top_score, top_candidate = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    is_ambiguous = top_score < 0.6 or (len(scored) > 1 and top_score - second_score < 0.12)

    notes = [
        f"name score target={spec.target_professor or spec.target_lab or 'not provided'}",
        f"top score={top_score:.2f}, second score={second_score:.2f}",
    ]
    if spec.target_schools and not _has_school_match(top_candidate, spec.target_schools):
        notes.append("No strong affiliation match with target_schools.")
    if is_ambiguous:
        notes.append("Identity should be confirmed by a human before final use.")

    resolved = ResolvedProfessor(
        display_name=top_candidate.display_name,
        openalex_id=top_candidate.source_ids.get("openalex", top_candidate.candidate_id),
        affiliation=top_candidate.affiliations[0] if top_candidate.affiliations else None,
        homepage_url=top_candidate.homepage_url,
        normalized_names=_candidate_names(top_candidate),
        identity_confidence=round(top_score, 3),
        is_ambiguous=is_ambiguous,
        disambiguation_notes=notes,
        source_evidence_ids=top_candidate.evidence_ids,
    )

    ev = Evidence(
        claim=(
            f"Professor identity resolved to {resolved.display_name}"
            f" with confidence {resolved.identity_confidence}."
        ),
        source_url=None,
        source_title="Deterministic professor identity resolver",
        source_type=SourceType.MODEL_INFERENCE,
        confidence=resolved.identity_confidence,
        is_inference=True,
        supports=resolved.source_evidence_ids,
        metadata={"is_ambiguous": is_ambiguous, "notes": notes},
    )

    return {
        "resolved_professor": resolved,
        "evidence": state.evidence + [ev],
        "status": RunStatus.NEEDS_REVIEW if is_ambiguous else RunStatus.RESOLVED,
        "trace": state.trace
        + [
            node_event(
                "resolve_professor_identity",
                "needs_review" if is_ambiguous else "ok",
                f"Selected {resolved.display_name}.",
                confidence=resolved.identity_confidence,
            )
        ],
    }


@traceable(name="collect_publications")
async def collect_publications(state: ResearchState) -> Dict[str, Any]:
    """Collect recent publications for the resolved professor."""
    spec = state.run_spec
    professor = state.resolved_professor
    if spec is None:
        return {"status": RunStatus.NEEDS_CLARIFICATION}
    if professor is None or not professor.openalex_id:
        return {
            "status": RunStatus.PARTIAL,
            "trace": state.trace
            + [
                node_event(
                    "collect_publications",
                    "partial",
                    "No resolved OpenAlex professor id was available.",
                )
            ],
        }

    current_year = datetime.now(timezone.utc).year
    from_year = current_year - spec.publication_years + 1
    gateway = _gateway_for(state, spec)
    openalex = OpenAlexClient()
    gateway.register("openalex.fetch_author_works", openalex.fetch_author_works)
    call = await gateway.call(
        "openalex.fetch_author_works",
        {"openalex_id": professor.openalex_id, "from_year": from_year, "limit": 12},
    )
    publications = [item for item in call.result.items if isinstance(item, Publication)]
    errors = state.errors[:]
    if call.result.status == "error":
        errors.append(call.result.message or "OpenAlex work search failed.")

    return {
        "publications": publications,
        "evidence": state.evidence + call.result.evidence,
        "tool_logs": state.tool_logs + [call.log],
        "tool_call_count": call.next_tool_call_count,
        "errors": errors,
        "status": state.status if publications else RunStatus.PARTIAL,
        "trace": state.trace
        + [
            node_event(
                "collect_publications",
                call.result.status,
                f"Collected {len(publications)} recent publications.",
                from_year=from_year,
            )
        ],
    }


@traceable(name="analyze_match")
async def analyze_match(state: ResearchState) -> Dict[str, Any]:
    """Analyze publication topics against applicant interests."""
    spec = state.run_spec
    if spec is None:
        return {"status": RunStatus.NEEDS_CLARIFICATION}

    corpus = " ".join(
        [publication.title for publication in state.publications]
        + [
            keyword
            for publication in state.publications
            for keyword in publication.keywords
        ]
    ).lower()
    interest_hits = [
        interest
        for interest in spec.research_interests
        if _interest_matches(interest, corpus)
    ]
    top_keywords = _top_keywords(state.publications)
    score = round(
        len(interest_hits) / max(len(spec.research_interests), 1),
        3,
    )
    if state.publications and score == 0:
        score = 0.2

    summary = (
        f"Matched {len(interest_hits)} of {len(spec.research_interests)} applicant "
        f"research interests against recent OpenAlex publications."
    )
    evidence_ids = [
        evidence_id
        for publication in state.publications[:8]
        for evidence_id in publication.source_evidence_ids
    ]
    trend = ResearchTrend(
        recent_years=spec.publication_years,
        top_keywords=top_keywords,
        matched_interests=interest_hits,
        summary=summary,
        evidence_ids=evidence_ids,
    )
    ev = Evidence(
        claim=f"Applicant-professor research fit score was inferred as {score}.",
        source_url=None,
        source_title="Deterministic match analyzer",
        source_type=SourceType.MODEL_INFERENCE,
        confidence=0.6 if state.publications else 0.3,
        is_inference=True,
        supports=evidence_ids,
        metadata={"matched_interests": interest_hits, "top_keywords": top_keywords},
    )

    return {
        "research_trend": trend,
        "evidence": state.evidence + [ev],
        "trace": state.trace
        + [
            node_event(
                "analyze_match",
                "ok",
                "Calculated deterministic research fit score.",
                score=score,
            )
        ],
    }


@traceable(name="generate_report")
async def generate_report(state: ResearchState) -> Dict[str, Any]:
    """Generate a structured lab profile and comparison report."""
    spec = state.run_spec
    if spec is None:
        return {"status": RunStatus.NEEDS_CLARIFICATION}

    match_score = 0.0
    if state.research_trend:
        match_score = round(
            len(state.research_trend.matched_interests)
            / max(len(spec.research_interests), 1),
            3,
        )
        if state.publications and match_score == 0:
            match_score = 0.2

    missing = []
    if not any(ev.source_type == SourceType.WEB_SEARCH for ev in state.evidence):
        missing.append("official lab or admission web source")
    if not state.publications:
        missing.append("recent publications")
    if state.resolved_professor and state.resolved_professor.is_ambiguous:
        missing.append("human-confirmed professor identity")

    admission_requirements = [
        AdmissionRequirement(
            requirement_type="official_admission_source",
            value="Not collected in this MVP run.",
            confidence=0.0,
            is_missing=True,
        )
    ]
    profile = LabProfile(
        lab_name=spec.target_lab,
        professor=state.resolved_professor,
        publications=state.publications,
        research_trend=state.research_trend,
        admission_requirements=admission_requirements,
        match_score=match_score,
        match_rationale=_match_rationale(spec, state.research_trend, match_score),
        missing_information=missing,
        evidence_ids=[ev.evidence_id for ev in state.evidence],
        status=state.status if state.status != RunStatus.PLANNED else RunStatus.PARTIAL,
    )
    professor_name = (
        state.resolved_professor.display_name
        if state.resolved_professor
        else spec.target_professor or spec.target_lab or "target lab"
    )
    llm_text = await draft_report_text(
        run_spec=spec,
        profile=profile,
        evidence_count=len(state.evidence),
        quality_issue_codes=[issue.code for issue in state.quality_issues],
    )
    if llm_text:
        profile = profile.model_copy(update={"match_rationale": llm_text["match_rationale"]})
        executive_summary = llm_text["executive_summary"]
        report_message = "Generated structured report with DeepSeek text drafting."
    else:
        executive_summary = (
            f"This MVP report analyzed {professor_name} against the applicant's "
            f"interests: {', '.join(spec.research_interests)}. Current status is "
            f"{profile.status.value}; missing items are explicitly listed."
        )
        report_message = "Generated structured report with deterministic fallback text."

    report = LabComparisonReport(
        run_id=state.run_id,
        title=f"Lab Research Report: {professor_name}",
        status=profile.status,
        executive_summary=executive_summary,
        profiles=[profile],
        evidence=state.evidence,
        quality_issues=state.quality_issues,
        trace_summary=state.tool_logs,
    )

    return {
        "lab_profile": profile,
        "report": report,
        "trace": state.trace
        + [
            node_event(
                "generate_report",
                "ok",
                report_message,
                evidence_count=len(state.evidence),
                model_provider="deepseek" if llm_text else None,
            )
        ],
    }


@traceable(name="quality_gate")
async def quality_gate(state: ResearchState) -> Dict[str, Any]:
    """Run deterministic report checks and set the final status."""
    spec = state.run_spec
    if spec is None:
        return {"status": RunStatus.NEEDS_CLARIFICATION}

    issues = evaluate_lab_profile(state.lab_profile, state.evidence, spec)
    has_error = any(issue.severity == "error" for issue in issues)
    needs_review = any(issue.code == "professor_identity_ambiguous" for issue in issues)
    if has_error:
        final_status = RunStatus.FAILED
    elif needs_review:
        final_status = RunStatus.NEEDS_REVIEW
    elif issues:
        final_status = RunStatus.PARTIAL
    else:
        final_status = RunStatus.COMPLETE

    profile = (
        state.lab_profile.model_copy(update={"status": final_status})
        if state.lab_profile
        else None
    )
    report = None
    if state.report:
        report = state.report.model_copy(
            update={
                "status": final_status,
                "profiles": [profile] if profile else [],
                "evidence": state.evidence,
                "quality_issues": issues,
                "trace_summary": state.tool_logs,
            }
        )

    return {
        "status": final_status,
        "lab_profile": profile,
        "report": report,
        "quality_issues": issues,
        "trace": state.trace
        + [
            node_event(
                "quality_gate",
                final_status.value,
                f"Quality gate found {len(issues)} issue(s).",
                issue_codes=[issue.code for issue in issues],
            )
        ],
    }


def route_after_clarification(state: ResearchState) -> str:
    """Stop early when the task contract is incomplete."""
    if state.status == RunStatus.NEEDS_CLARIFICATION:
        return END
    return "generate_research_plan"


def _gateway_for(state: ResearchState, spec: ResearchRunSpec) -> ToolGateway:
    return ToolGateway(
        max_tool_calls=spec.max_tool_calls,
        tool_timeout_seconds=spec.tool_timeout_seconds,
        max_retries=spec.max_retries,
        tool_call_count=state.tool_call_count,
    )


def _candidate_names(candidate: ProfessorCandidate) -> list[str]:
    names = [candidate.display_name, *candidate.alternative_names]
    return list(dict.fromkeys(name for name in names if name))


def _score_candidate(candidate: ProfessorCandidate, spec: ResearchRunSpec) -> float:
    score = 0.05
    target_name = spec.target_professor or spec.target_lab
    if target_name:
        target_norm = _normalize(target_name)
        name_norms = [_normalize(name) for name in _candidate_names(candidate)]
        if target_norm in name_norms:
            score += 0.45
        elif any(target_norm in name or name in target_norm for name in name_norms):
            score += 0.3
    else:
        score += 0.1

    if _has_school_match(candidate, spec.target_schools):
        score += 0.3

    topic_text = " ".join(candidate.topics).lower()
    if any(_interest_matches(interest, topic_text) for interest in spec.research_interests):
        score += 0.15

    if candidate.works_count:
        score += 0.03
    if candidate.cited_by_count:
        score += 0.02
    return min(score, 0.95)


def _has_school_match(candidate: ProfessorCandidate, schools: list[str]) -> bool:
    if not schools:
        return False
    affiliation_text = " ".join(candidate.affiliations).lower()
    return any(_normalize(school) in _normalize(affiliation_text) for school in schools)


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9一-龥ぁ-んァ-ン]+", "", value.lower())


def _interest_matches(interest: str, corpus: str) -> bool:
    interest_lower = interest.lower()
    if interest_lower in corpus:
        return True
    tokens = [token for token in re.findall(r"[a-z0-9]+", interest_lower) if len(token) > 3]
    return any(token in corpus for token in tokens)


def _top_keywords(publications: list[Publication]) -> list[str]:
    counter: Counter[str] = Counter()
    for publication in publications:
        for keyword in publication.keywords:
            normalized = keyword.strip()
            if normalized:
                counter[normalized] += 1
        for token in re.findall(r"[A-Za-z][A-Za-z-]{3,}", publication.title):
            counter[token.lower()] += 1
    return [keyword for keyword, _ in counter.most_common(8)]


def _match_rationale(
    spec: ResearchRunSpec,
    trend: ResearchTrend | None,
    match_score: float,
) -> str:
    if trend is None:
        return "No publication trend was available, so fit cannot be judged confidently."
    if trend.matched_interests:
        return (
            f"Matched interests: {', '.join(trend.matched_interests)}. "
            f"Deterministic match score: {match_score}."
        )
    return (
        "Recent publications were found, but no direct keyword match was detected for "
        f"{', '.join(spec.research_interests)}. Treat this as a weak inferred match."
    )


# Define the graph
builder = StateGraph(ResearchState, context_schema=Context)
builder.add_node("clarify_requirements", clarify_requirements)
builder.add_node("generate_research_plan", generate_research_plan)
builder.add_node("discover_professors", discover_professors)
builder.add_node("discover_web_sources", discover_web_sources)
builder.add_node("resolve_professor_identity", resolve_professor_identity)
builder.add_node("collect_publications", collect_publications)
builder.add_node("analyze_match", analyze_match)
builder.add_node("generate_report", generate_report)
builder.add_node("quality_gate", quality_gate)

builder.add_edge(START, "clarify_requirements")
builder.add_conditional_edges("clarify_requirements", route_after_clarification)
builder.add_edge("generate_research_plan", "discover_professors")
builder.add_edge("discover_professors", "discover_web_sources")
builder.add_edge("discover_web_sources", "resolve_professor_identity")
builder.add_edge("resolve_professor_identity", "collect_publications")
builder.add_edge("collect_publications", "analyze_match")
builder.add_edge("analyze_match", "generate_report")
builder.add_edge("generate_report", "quality_gate")
builder.add_edge("quality_gate", END)

graph = builder.compile(name="Lab Research Agent")
