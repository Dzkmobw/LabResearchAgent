"""DeepSeek V4 client used by all LLM features."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from pydantic import ValidationError

from agent.schemas import LabProfile, ResearchRunSpec


class DeepSeekLLMError(RuntimeError):
    """Raised when DeepSeek cannot produce a usable response."""


class DeepSeekChatClient:
    """Small OpenAI-compatible DeepSeek chat client.

    The project intentionally keeps LLM access here instead of importing a
    provider-specific chain inside graph nodes.
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = os.getenv("LAB_AGENT_MODEL", "deepseek-v4-flash")
        self.thinking = os.getenv("LAB_AGENT_THINKING", "disabled")
        self.reasoning_effort = os.getenv("LAB_AGENT_REASONING_EFFORT", "high")

    @property
    def is_configured(self) -> bool:
        """Return whether the client has enough config to call DeepSeek."""
        return bool(self.api_key)

    async def chat_json(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        max_tokens: int = 1200,
    ) -> dict[str, Any]:
        """Call DeepSeek and parse a strict JSON object response."""
        content = await self._chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Return one valid JSON object only.\n\n"
                        + json.dumps(user_payload, ensure_ascii=False, default=str)
                    ),
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
        )
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise DeepSeekLLMError(f"DeepSeek returned invalid JSON: {content[:200]}") from exc

    async def chat_text(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        max_tokens: int = 800,
    ) -> str:
        """Call DeepSeek for short human-readable text."""
        return await self._chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False, default=str),
                },
            ],
            response_format=None,
            max_tokens=max_tokens,
        )

    async def _chat(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, str] | None,
        max_tokens: int,
    ) -> str:
        if not self.api_key:
            raise DeepSeekLLMError("DEEPSEEK_API_KEY is not configured.")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens,
            "thinking": {"type": self.thinking},
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if self.thinking == "enabled":
            payload["reasoning_effort"] = self.reasoning_effort

        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise DeepSeekLLMError(f"Unexpected DeepSeek response: {data}") from exc

        if not content:
            raise DeepSeekLLMError("DeepSeek returned empty content.")
        return content.strip()


async def infer_research_run_spec(
    user_request: str,
) -> tuple[ResearchRunSpec | None, list[str], str | None]:
    """Parse natural language into ResearchRunSpec using DeepSeek V4."""
    client = DeepSeekChatClient()
    if not client.is_configured:
        return None, ["DEEPSEEK_API_KEY"], "DeepSeek is not configured."

    system_prompt = """
You convert a lab research request into JSON for a PhD/Master lab research agent.
Do not invent missing constraints. If a required field is not clearly stated,
put it in missing_fields instead of guessing.

Required output JSON shape:
{
  "run_spec": {
    "target_country": "Japan",
    "degree": "phd",
    "research_interests": ["natural language processing"],
    "target_schools": ["Kyoto University"],
    "target_professor": "Tatsuya Kawahara",
    "target_lab": null,
    "lab_count": 1,
    "publication_years": 5,
    "max_tool_calls": 8
  },
  "missing_fields": []
}

Rules:
- degree must be "master" or "phd".
- target_country, degree, and at least one research interest are required.
- at least one of target_professor, target_lab, or target_schools is required.
- Use "Japan" only if Japan/Japanese/日本 is mentioned.
- Use null for unknown optional strings.
- Keep defaults for lab_count=1, publication_years=5, max_tool_calls=8 unless stated.
""".strip()

    try:
        parsed = await client.chat_json(
            system_prompt=system_prompt,
            user_payload={"user_request": user_request},
            max_tokens=1200,
        )
    except Exception as exc:  # noqa: BLE001 - caller should degrade to clarification.
        return None, ["run_spec"], f"DeepSeek parsing failed: {exc}"

    missing = parsed.get("missing_fields") or []
    raw_spec = parsed.get("run_spec")
    if missing or not raw_spec:
        return None, list(missing) or ["run_spec"], None

    try:
        return ResearchRunSpec.model_validate(raw_spec), [], None
    except ValidationError as exc:
        return None, ["run_spec"], f"DeepSeek output failed schema validation: {exc}"


async def draft_report_text(
    run_spec: ResearchRunSpec,
    profile: LabProfile,
    evidence_count: int,
    quality_issue_codes: list[str],
) -> dict[str, str] | None:
    """Ask DeepSeek V4 to draft report-facing text from structured facts."""
    client = DeepSeekChatClient()
    if not client.is_configured:
        return None

    system_prompt = """
You write concise report text for a lab research agent.
Use only the structured facts in the JSON. Do not invent admissions status,
funding, recruitment, or professor details. If information is missing, say it is missing.
Return valid JSON:
{
  "executive_summary": "...",
  "match_rationale": "..."
}
""".strip()

    payload = {
        "run_spec": run_spec.model_dump(mode="json"),
        "profile": profile.model_dump(mode="json"),
        "evidence_count": evidence_count,
        "quality_issue_codes": quality_issue_codes,
    }
    try:
        result = await client.chat_json(system_prompt, payload, max_tokens=1000)
    except Exception:
        return None

    executive_summary = result.get("executive_summary")
    match_rationale = result.get("match_rationale")
    if not isinstance(executive_summary, str) or not isinstance(match_rationale, str):
        return None
    return {
        "executive_summary": executive_summary.strip(),
        "match_rationale": match_rationale.strip(),
    }
