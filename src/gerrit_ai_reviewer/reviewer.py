"""Orchestration: change-id -> Gerrit fetch -> LLM review -> post back."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import GerritConfig, LLMConfig
from .gerrit import GerritClient
from .llm import InlineComment, ReviewResult, review

# Hard cap to avoid blowing token budget on huge CLs.
MAX_PATCH_CHARS = 120_000


@dataclass
class RunResult:
    change_id: str
    revision: str
    result: ReviewResult
    posted: bool


def run(
    gerrit_cfg: GerritConfig,
    llm_cfg: LLMConfig,
    change_id: str,
    revision: str = "current",
    dry_run: bool = False,
    vote_label: str | None = None,
) -> RunResult:
    with GerritClient(gerrit_cfg.base_url, gerrit_cfg.user, gerrit_cfg.http_password) as gc:
        change = gc.get_change(change_id)
        rev_sha = _current_revision_sha(change) if revision == "current" else revision
        patch = gc.get_patch(change_id, revision=revision)
        files = gc.list_files(change_id, revision=revision)

        prompt = _build_prompt(change, files, patch)
        result = review(llm_cfg, prompt)

        if dry_run:
            return RunResult(change_id, rev_sha, result, posted=False)

        comments = _group_comments(result.comments, files)
        labels = {vote_label.split("=")[0]: int(vote_label.split("=")[1])} if vote_label else None
        message = _format_message(result)
        gc.post_review(
            change_id=change_id,
            revision=revision,
            message=message,
            comments=comments or None,
            labels=labels,
        )
        return RunResult(change_id, rev_sha, result, posted=True)


def _current_revision_sha(change: dict[str, Any]) -> str:
    cur = change.get("current_revision")
    return cur or "current"


def _build_prompt(change: dict[str, Any], files: dict[str, Any], patch: str) -> str:
    subject = change.get("subject", "")
    project = change.get("project", "")
    branch = change.get("branch", "")
    owner = (change.get("owner") or {}).get("name", "")
    file_list = [p for p in files.keys() if p != "/COMMIT_MSG"]

    if len(patch) > MAX_PATCH_CHARS:
        patch = patch[:MAX_PATCH_CHARS] + "\n...[truncated]..."

    return (
        f"Gerrit change metadata:\n"
        f"- project: {project}\n"
        f"- branch: {branch}\n"
        f"- owner: {owner}\n"
        f"- subject: {subject}\n"
        f"- changed files:\n"
        + "\n".join(f"  - {p}" for p in file_list)
        + "\n\nUnified diff (context + added/removed lines):\n"
        "```diff\n"
        f"{patch}\n"
        "```\n"
    )


def _group_comments(
    comments: list[InlineComment],
    files: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    valid_files = set(files.keys())
    grouped: dict[str, list[dict[str, Any]]] = {}
    for c in comments:
        if c.file not in valid_files:
            # Model hallucinated a file path; skip silently — summary still carries context.
            continue
        grouped.setdefault(c.file, []).append(
            {"line": c.line, "message": f"[{c.severity}] {c.message}"}
        )
    return grouped


def _format_message(result: ReviewResult) -> str:
    head = "🤖 AI Code Review"
    if not result.comments:
        return f"{head}\n\n{result.summary}\n\n(no inline issues found)"
    return f"{head}\n\n{result.summary}\n\n{len(result.comments)} inline comment(s) posted."
