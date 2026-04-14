"""Tests for the orchestrator — wires a fake Gerrit client + stubbed LLM."""

from __future__ import annotations

from typing import Any

import pytest

from gerrit_ai_reviewer import reviewer
from gerrit_ai_reviewer.config import GerritConfig, LLMConfig
from gerrit_ai_reviewer.llm import InlineComment, ReviewResult


class FakeClient:
    """Stand-in for GerritClient — records `post_review` calls."""

    instance: FakeClient | None = None

    def __init__(self, *_args, **_kwargs):
        self.posted: dict[str, Any] | None = None
        FakeClient.instance = self

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        pass

    def get_change(self, change_id: str) -> dict[str, Any]:
        return {
            "project": "demo",
            "branch": "main",
            "owner": {"name": "alice"},
            "subject": "fix thing",
            "current_revision": "abcd1234",
        }

    def get_patch(self, change_id: str, revision: str = "current") -> str:
        return "diff --git a/src/a.py b/src/a.py\n@@ -1,2 +1,3 @@\n a\n+b\n c\n"

    def list_files(self, change_id: str, revision: str = "current") -> dict[str, Any]:
        return {"/COMMIT_MSG": {}, "src/a.py": {}}

    def post_review(self, **kwargs) -> None:
        self.posted = kwargs


def _gerrit_cfg() -> GerritConfig:
    return GerritConfig(base_url="https://g.example", user="u", http_password="p")


def _llm_cfg() -> LLMConfig:
    return LLMConfig(provider="openai", base_url=None, api_key="sk", model="m")


@pytest.fixture
def fake_env(monkeypatch):
    monkeypatch.setattr(reviewer, "GerritClient", FakeClient)
    FakeClient.instance = None
    return monkeypatch


def test_dry_run_skips_post(fake_env):
    fake_env.setattr(
        reviewer,
        "review",
        lambda cfg, prompt: ReviewResult(summary="ok", comments=[]),
    )
    rr = reviewer.run(_gerrit_cfg(), _llm_cfg(), change_id="42", dry_run=True)
    assert rr.posted is False
    assert rr.revision == "abcd1234"
    assert FakeClient.instance is not None
    assert FakeClient.instance.posted is None


def test_post_includes_grouped_comments(fake_env):
    fake_env.setattr(
        reviewer,
        "review",
        lambda cfg, prompt: ReviewResult(
            summary="2 issues",
            comments=[
                InlineComment(file="src/a.py", line=2, severity="warn", message="suspicious"),
                InlineComment(file="src/a.py", line=3, severity="error", message="bug"),
            ],
        ),
    )
    rr = reviewer.run(_gerrit_cfg(), _llm_cfg(), change_id="42")
    assert rr.posted is True

    posted = FakeClient.instance.posted
    assert posted["change_id"] == "42"
    assert posted["revision"] == "current"
    comments = posted["comments"]
    assert set(comments.keys()) == {"src/a.py"}
    assert [c["line"] for c in comments["src/a.py"]] == [2, 3]
    assert "[warn] suspicious" in comments["src/a.py"][0]["message"]
    assert "[error] bug" in comments["src/a.py"][1]["message"]
    assert posted["labels"] is None  # no vote by default


def test_hallucinated_file_path_is_dropped(fake_env):
    fake_env.setattr(
        reviewer,
        "review",
        lambda cfg, prompt: ReviewResult(
            summary="mixed",
            comments=[
                InlineComment(file="src/a.py", line=2, message="real"),
                InlineComment(file="src/imaginary.py", line=1, message="ghost"),
            ],
        ),
    )
    reviewer.run(_gerrit_cfg(), _llm_cfg(), change_id="42")

    comments = FakeClient.instance.posted["comments"]
    assert set(comments.keys()) == {"src/a.py"}
    assert len(comments["src/a.py"]) == 1


def test_vote_label_parsed(fake_env):
    fake_env.setattr(
        reviewer,
        "review",
        lambda cfg, prompt: ReviewResult(summary="bad", comments=[]),
    )
    reviewer.run(_gerrit_cfg(), _llm_cfg(), change_id="42", vote_label="Code-Review=-1")
    assert FakeClient.instance.posted["labels"] == {"Code-Review": -1}


def test_no_comments_posts_summary_only(fake_env):
    fake_env.setattr(
        reviewer,
        "review",
        lambda cfg, prompt: ReviewResult(summary="lgtm", comments=[]),
    )
    reviewer.run(_gerrit_cfg(), _llm_cfg(), change_id="42")
    posted = FakeClient.instance.posted
    assert posted["comments"] is None
    assert "lgtm" in posted["message"]
    assert "no inline issues found" in posted["message"]
