"""Unified LLM reviewer.

Supports two backends, both honoring a custom base_url:
- openai:    uses `openai` SDK (`OpenAI(base_url=..., api_key=...)`)
- anthropic: uses `anthropic` SDK (`Anthropic(base_url=..., api_key=...)`)

The model is asked to return JSON matching `ReviewResult`.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .config import LLMConfig

SYSTEM_PROMPT = """You are a senior code reviewer. Review the provided Gerrit change.

Rules:
- Only comment on real issues: bugs, security, concurrency, resource leaks,
  incorrect logic, broken tests, dangerous API use, missing error handling at
  boundaries, obvious perf foot-guns.
- Do NOT nitpick style, naming, or personal taste.
- Each inline comment MUST target a line that appears in the diff (added or
  context line near an added one) in the given file path.
- If nothing serious, return an empty `comments` list and a short `summary`.
- Output MUST be valid JSON matching the schema. No prose outside JSON.

Schema:
{
  "summary": "string, <=400 chars, high level verdict",
  "comments": [
    {"file": "path/as/in/change", "line": <int, post-image line>, "severity": "info|warn|error", "message": "string"}
  ]
}
"""


class InlineComment(BaseModel):
    file: str
    line: int = Field(ge=1)
    severity: str = "info"
    message: str


class ReviewResult(BaseModel):
    summary: str
    comments: list[InlineComment] = Field(default_factory=list)


def review(cfg: LLMConfig, user_prompt: str) -> ReviewResult:
    raw = _call_openai(cfg, user_prompt) if cfg.provider == "openai" else _call_anthropic(cfg, user_prompt)
    return _parse(raw)


def _call_openai(cfg: LLMConfig, user_prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    resp = client.chat.completions.create(
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content or ""


def _call_anthropic(cfg: LLMConfig, user_prompt: str) -> str:
    from anthropic import Anthropic

    kwargs: dict[str, Any] = {"api_key": cfg.api_key}
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    client = Anthropic(**kwargs)
    msg = client.messages.create(
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt + "\n\nRespond with a single JSON object only."}],
    )
    # Concatenate text blocks.
    parts: list[str] = []
    for block in msg.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts)


def _parse(raw: str) -> ReviewResult:
    raw = raw.strip()
    # Strip markdown fences if model wraps JSON.
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON: {e}\n---\n{raw[:500]}") from e
    try:
        return ReviewResult.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"LLM JSON failed schema: {e}") from e
