"""
Agent-facing prompt for CiteCast.

The prompt carries version-1 metadata only (title, abstract, authors, primary
category, announce date) — fields frozen at announcement time. It never
mentions the dataset's citation-band composition: telling the model the mix is
oversampled toward highly-cited papers would hand every policy the
stratification prior for free.
"""

from __future__ import annotations

from typing import Any

from constants import CUTOFF, MAX_PROMPT_AUTHORS, SNAPSHOT

AGENT_PROMPT_TEMPLATE = """You are forecasting the scholarly impact of a newly announced arXiv paper.

PAPER
arXiv ID: {arxiv_id}
Announced: {announced}
Primary category: {primary_category}
Authors ({author_count}): {authors_display}
Title: {title}

Abstract:
{abstract}

TASK
Predict the total number of citations this paper will have accumulated on Semantic Scholar as of {snapshot}.

You have two research tools, web_search and web_fetch. They see the web exactly as it stood on {cutoff} — nothing published or crawled after that date is visible to you. Use them to gauge the paper's early reception, the authors' track records, how active the paper's topic area is, and the citation velocity of comparable recent work.

When you are ready, call submit_prediction exactly once with your integer prediction and a brief justification. You get one submission; there is no retry.

SCORING
Your reward depends on how close your prediction is in log space: predicting 10 when the truth is 20 earns far more than predicting 0, and overshooting by 2x costs the same as undershooting by 2x. An exact prediction earns full reward. Citation counts for papers of this age range from zero to several hundred."""


def authors_display(authors: list[str], author_count: int) -> str:
    shown = authors[:MAX_PROMPT_AUTHORS]
    text = ", ".join(shown)
    hidden = author_count - len(shown)
    if hidden > 0:
        text += f", and {hidden} others"
    return text


def render_prompt(row: dict[str, Any]) -> str:
    return AGENT_PROMPT_TEMPLATE.format(
        arxiv_id=row["arxiv_id"],
        announced=row["announced"],
        primary_category=row["primary_category"],
        author_count=row["author_count"],
        authors_display=authors_display(row["authors"], row["author_count"]),
        title=row["title"],
        abstract=row["abstract"],
        snapshot=SNAPSHOT,
        cutoff=CUTOFF,
    )
