"""OpenAlex academic data tools."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx

from agent.schemas import Evidence, ProfessorCandidate, Publication, SourceType, ToolResult


class OpenAlexClient:
    """Small async client for the OpenAlex REST API."""

    def __init__(self, base_url: str = "https://api.openalex.org") -> None:
        self.base_url = base_url.rstrip("/")
        self.mailto = os.getenv("OPENALEX_MAILTO")

    async def search_authors(
        self,
        query: str,
        target_schools: list[str] | None = None,
        research_interests: list[str] | None = None,
        limit: int = 5,
    ) -> ToolResult:
        """Search possible professor identities by name."""
        params: dict[str, Any] = {"search": query, "per-page": limit}
        if self.mailto:
            params["mailto"] = self.mailto

        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(f"{self.base_url}/authors", params=params)
            response.raise_for_status()
            data = response.json()

        request_url = str(response.request.url)
        candidates: list[ProfessorCandidate] = []
        evidence: list[Evidence] = []

        for raw_author in data.get("results", []):
            affiliations = _extract_affiliations(raw_author)
            topics = _extract_topics(raw_author)
            candidate_id = raw_author.get("id") or raw_author.get("ids", {}).get("openalex")
            if not candidate_id:
                continue

            candidate = ProfessorCandidate(
                candidate_id=candidate_id,
                display_name=raw_author.get("display_name") or query,
                alternative_names=raw_author.get("display_name_alternatives") or [],
                affiliations=affiliations,
                source_ids={
                    key: value
                    for key, value in (raw_author.get("ids") or {}).items()
                    if isinstance(value, str)
                },
                topics=topics,
                works_count=raw_author.get("works_count"),
                cited_by_count=raw_author.get("cited_by_count"),
            )
            ev = Evidence(
                claim=(
                    f"OpenAlex returned author candidate {candidate.display_name}"
                    f" with affiliations: {', '.join(affiliations[:3]) or 'unknown'}."
                ),
                source_url=request_url,
                source_title="OpenAlex Authors API",
                source_type=SourceType.ACADEMIC_API,
                confidence=0.75,
                is_inference=False,
                supports=[candidate.candidate_id],
                metadata={
                    "target_schools": target_schools or [],
                    "research_interests": research_interests or [],
                },
            )
            candidate.evidence_ids.append(ev.evidence_id)
            candidates.append(candidate)
            evidence.append(ev)

        return ToolResult(
            tool_name="openalex.search_authors",
            status="ok",
            items=candidates,
            evidence=evidence,
            metadata={"result_count": len(candidates), "retrieved_at": _now_iso()},
        )

    async def fetch_author_works(
        self,
        openalex_id: str,
        from_year: int,
        limit: int = 10,
    ) -> ToolResult:
        """Fetch recent works for one OpenAlex author id."""
        author_key = openalex_id.rstrip("/").split("/")[-1]
        filters = [
            f"authorships.author.id:{author_key}",
            f"from_publication_date:{from_year}-01-01",
        ]
        params: dict[str, Any] = {
            "filter": ",".join(filters),
            "sort": "publication_date:desc",
            "per-page": limit,
        }
        if self.mailto:
            params["mailto"] = self.mailto

        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(f"{self.base_url}/works", params=params)
            response.raise_for_status()
            data = response.json()

        request_url = str(response.request.url)
        publications: list[Publication] = []
        evidence: list[Evidence] = []

        for raw_work in data.get("results", []):
            title = raw_work.get("title")
            if not title:
                continue

            publication_id = raw_work.get("id") or raw_work.get("doi") or title
            url = raw_work.get("doi") or raw_work.get("id")
            venue = (
                ((raw_work.get("primary_location") or {}).get("source") or {}).get(
                    "display_name"
                )
            )
            keywords = _extract_work_keywords(raw_work)
            authors = [
                ((author.get("author") or {}).get("display_name") or "")
                for author in raw_work.get("authorships", [])
            ]
            authors = [author for author in authors if author]

            ev = Evidence(
                claim=f"OpenAlex lists a recent publication titled '{title}'.",
                source_url=request_url,
                source_title="OpenAlex Works API",
                source_type=SourceType.ACADEMIC_API,
                confidence=0.8,
                is_inference=False,
                supports=[publication_id],
                metadata={"openalex_work_id": raw_work.get("id")},
            )
            publications.append(
                Publication(
                    publication_id=publication_id,
                    title=title,
                    year=raw_work.get("publication_year"),
                    venue=venue,
                    doi=raw_work.get("doi"),
                    cited_by_count=raw_work.get("cited_by_count"),
                    url=url,
                    authors=authors,
                    keywords=keywords,
                    source_evidence_ids=[ev.evidence_id],
                )
            )
            evidence.append(ev)

        return ToolResult(
            tool_name="openalex.fetch_author_works",
            status="ok",
            items=publications,
            evidence=evidence,
            metadata={"result_count": len(publications), "retrieved_at": _now_iso()},
        )


def _extract_affiliations(raw_author: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for institution in raw_author.get("last_known_institutions") or []:
        name = institution.get("display_name")
        if name:
            names.append(name)
    for affiliation in raw_author.get("affiliations") or []:
        for institution in affiliation.get("institutions") or []:
            name = institution.get("display_name")
            if name and name not in names:
                names.append(name)
    return names


def _extract_topics(raw_author: dict[str, Any]) -> list[str]:
    topics: list[str] = []
    for concept in raw_author.get("x_concepts") or []:
        name = concept.get("display_name")
        if name:
            topics.append(name)
    return topics


def _extract_work_keywords(raw_work: dict[str, Any]) -> list[str]:
    keywords: list[str] = []
    for concept in raw_work.get("concepts") or []:
        name = concept.get("display_name")
        if name:
            keywords.append(name)
    for keyword in raw_work.get("keywords") or []:
        name = keyword.get("display_name") or keyword.get("keyword")
        if name and name not in keywords:
            keywords.append(name)
    return keywords[:12]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
