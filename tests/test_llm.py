"""Tests for the LLM layer (parsing + backend routing).

We don't touch real SDKs — `_call_openai` / `_call_anthropic` get
monkeypatched to return canned raw text, then `review()` parses it.
"""

from __future__ import annotations

import pytest

from gerrit_ai_reviewer import llm
from gerrit_ai_reviewer.config import LLMConfig
from gerrit_ai_reviewer.llm import ReviewResult, _parse, review


def _cfg(provider: str) -> LLMConfig:
    return LLMConfig(
        provider=provider,
        base_url="https://llm.internal/v1",
        api_key="sk-test",
        model="test-model",
    )


def test_parse_plain_json():
    raw = '{"summary": "lgtm", "comments": []}'
    r = _parse(raw)
    assert isinstance(r, ReviewResult)
    assert r.summary == "lgtm"
    assert r.comments == []


def test_parse_strips_markdown_fences():
    raw = """```json
    {"summary": "ok", "comments": [
        {"file": "a.py", "line": 2, "severity": "warn", "message": "off-by-one"}
    ]}
    ```"""
    r = _parse(raw)
    assert len(r.comments) == 1
    assert r.comments[0].file == "a.py"
    assert r.comments[0].line == 2


def test_parse_rejects_bad_json():
    with pytest.raises(ValueError, match="did not return valid JSON"):
        _parse("not json at all")


def test_parse_rejects_bad_schema():
    # `line` must be >= 1
    raw = '{"summary": "x", "comments": [{"file": "a.py", "line": 0, "message": "x"}]}'
    with pytest.raises(ValueError, match="failed schema"):
        _parse(raw)


def test_review_routes_openai(monkeypatch):
    called = {}

    def fake_openai(cfg, prompt):
        called["backend"] = "openai"
        called["prompt"] = prompt
        return '{"summary": "from openai", "comments": []}'

    def fake_anthropic(cfg, prompt):
        raise AssertionError("anthropic path should not fire")

    monkeypatch.setattr(llm, "_call_openai", fake_openai)
    monkeypatch.setattr(llm, "_call_anthropic", fake_anthropic)

    r = review(_cfg("openai"), "diff body")
    assert called["backend"] == "openai"
    assert called["prompt"] == "diff body"
    assert r.summary == "from openai"


def test_review_routes_anthropic(monkeypatch):
    called = {}

    def fake_openai(cfg, prompt):
        raise AssertionError("openai path should not fire")

    def fake_anthropic(cfg, prompt):
        called["backend"] = "anthropic"
        return '{"summary": "from claude", "comments": []}'

    monkeypatch.setattr(llm, "_call_openai", fake_openai)
    monkeypatch.setattr(llm, "_call_anthropic", fake_anthropic)

    r = review(_cfg("anthropic"), "diff body")
    assert called["backend"] == "anthropic"
    assert r.summary == "from claude"
