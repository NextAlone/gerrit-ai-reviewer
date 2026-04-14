"""CLI entry for Jenkins pipeline usage.

Example:
    export GERRIT_URL=https://gerrit.example.com
    export GERRIT_USER=jenkins-bot
    export GERRIT_HTTP_PASSWORD=xxxx
    export LLM_PROVIDER=openai
    export LLM_BASE_URL=https://llm.internal/v1
    export LLM_API_KEY=sk-xxx
    export LLM_MODEL=gpt-4o-mini
    gerrit-ai-review --change-id 12345
"""
from __future__ import annotations

import json
import sys

import click

from .config import GerritConfig, LLMConfig
from .gerrit import GerritError
from .reviewer import run


@click.command(context_settings={"show_default": True})
@click.option("--change-id", required=True, help="Gerrit change id (numeric, Change-Id, or project~branch~Change-Id)")
@click.option("--revision", default="current", help="Revision: 'current', sha, or patchset number")
@click.option("--dry-run", is_flag=True, help="Do not post back to Gerrit, print result JSON")
@click.option(
    "--vote",
    default=None,
    help="Optional label vote, e.g. 'Code-Review=-1'. Omit to skip voting.",
)
def main(change_id: str, revision: str, dry_run: bool, vote: str | None) -> None:
    try:
        gerrit_cfg = GerritConfig.from_env()
        llm_cfg = LLMConfig.from_env()
    except (RuntimeError, ValueError) as e:
        click.echo(f"config error: {e}", err=True)
        sys.exit(2)

    try:
        rr = run(
            gerrit_cfg=gerrit_cfg,
            llm_cfg=llm_cfg,
            change_id=change_id,
            revision=revision,
            dry_run=dry_run,
            vote_label=vote,
        )
    except GerritError as e:
        click.echo(f"gerrit error: {e}", err=True)
        sys.exit(3)
    except ValueError as e:
        click.echo(f"llm/output error: {e}", err=True)
        sys.exit(4)

    out = {
        "change_id": rr.change_id,
        "revision": rr.revision,
        "posted": rr.posted,
        "summary": rr.result.summary,
        "comments": [c.model_dump() for c in rr.result.comments],
    }
    click.echo(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
