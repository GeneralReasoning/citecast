"""
CiteCast: forecast an arXiv paper's citation count from its moment of birth.

Every task is one paper announced on arXiv in January 2026. The agent sees the
paper's version-1 metadata and must predict the total citations Semantic
Scholar records for it on the snapshot date (2026-08-23). Its only tools are
backdated web search and fetch, frozen at 2026-01-31 — the cutoff is enforced
server-side per request, so the agent can research the paper's earliest
reception, the authors' track records, and the state of the field, but can
never observe the future it is being asked to predict.

Grading is pure arithmetic (no LLM judges): a smooth log-ratio reward that
pays 1.0 for an exact prediction, ~0.47 at 2x error, ~0.02 at 10x error. The
shipped task mix is stratified across citation bands and calibrated so that no
constant prediction clears 0.40 mean reward — the only way to score well is to
actually judge papers.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional

from pydantic import BaseModel, Field

import constants  # noqa: F401  (sets OPENREWARD_WEB_CORPUS before any tool call)
from constants import CUTOFF, DATA_FILE, FETCH_MAX_CHARS, MAX_PREDICTION, SPLITS
from grading import band_of, compute_reward
from prompts import render_prompt

from openreward.environments import (
    Environment,
    JSONObject,
    Split,
    TextBlock,
    ToolOutput,
    tool,
)
from openreward.toolsets import BackSearchToolset
from openreward.toolsets._web_common import (
    WebFetchParams,
    WebSearchParams,
    to_tool_output,
)
from openreward.tools.web import FETCH_DESCRIPTION, SEARCH_DESCRIPTION, run_fetch, run_search
from openreward.web_service import WebServiceConfig


def _load_examples() -> dict[str, dict[str, Any]]:
    examples: dict[str, dict[str, Any]] = {}
    with open(DATA_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples[str(row["task_id"])] = row
    return examples


EXAMPLES = _load_examples()


class TaskSpec(BaseModel):
    task_id: str


class SubmitPredictionInput(BaseModel, extra="forbid"):
    predicted_citations: int = Field(
        ...,
        ge=0,
        le=MAX_PREDICTION,
        description="Your predicted total Semantic Scholar citation count for the paper on the snapshot date.",
    )
    reasoning: str = Field(
        ...,
        description="Brief justification for the prediction; recorded for analysis, not graded.",
    )


class CiteCastBackSearch(BackSearchToolset):
    """BackSearchToolset with three adjustments, all backdating-preserving.

    First, the session's secrets (``api_key``) are consulted when building the
    web-service config, falling back to the process environment — the stock
    toolset reads the process env only. Second, ``web_search`` passes
    ``include_snippets=True`` so results carry text snippets instead of bare
    titles and URLs. Third, ``web_fetch`` caps the fetched text at
    ``FETCH_MAX_CHARS`` and strips the SDK's full-text mirror from
    ``metadata["content"]``: training harnesses cap env tool outputs (enigma
    drops the whole output past 32,768 bytes when metadata alone exceeds the
    cap), and without this a full-paper fetch reaches the model as
    "[env tool output exceeded cap]" instead of content. The as_of cutoff
    still resolves through the parent's ``_current_as_of`` (env.web_as_of) on
    every call.
    """

    def __init__(self, env: Optional[Any] = None, **kwargs: Any) -> None:
        if kwargs.get("config") is None:
            secrets = getattr(env, "search_secrets", None)
            kwargs["config"] = WebServiceConfig.from_env(secrets)
        super().__init__(env, **kwargs)

    @tool
    async def web_search(self, params: WebSearchParams) -> ToolOutput:
        result = await run_search(
            query=params.query,
            as_of=self._current_as_of(),
            allowed_domains=params.allowed_domains,
            blocked_domains=params.blocked_domains,
            config=self.config,
            include_snippets=True,
        )
        return to_tool_output(result)

    @tool
    async def web_fetch(self, params: WebFetchParams) -> ToolOutput:
        result = await run_fetch(
            url=params.url,
            prompt=params.prompt,
            as_of=self._current_as_of(),
            config=self.config,
            max_chars=FETCH_MAX_CHARS,
        )
        out = to_tool_output(result)
        if out.metadata and "content" in out.metadata:
            slim = {k: v for k, v in out.metadata.items() if k != "content"}
            out = ToolOutput(
                blocks=out.blocks,
                metadata=slim or None,
                reward=out.reward,
                finished=out.finished,
            )
        return out


CiteCastBackSearch.web_search.__doc__ = SEARCH_DESCRIPTION
CiteCastBackSearch.web_fetch.__doc__ = FETCH_DESCRIPTION


class CiteCast(Environment):
    """Predict a January 2026 arXiv paper's citation count on 2026-08-23."""

    toolsets = [CiteCastBackSearch]
    # Read live by the toolset on every tool call; a class attribute outranks
    # the OPENREWARD_WEB_AS_OF env var, so nothing outside this file can widen
    # the cutoff.
    web_as_of = CUTOFF

    @classmethod
    def list_splits(cls) -> list[Split]:
        return [Split(name="train", type="train"), Split(name="test", type="test")]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        # The task spec carries only the id. Paper metadata, the citation band,
        # and above all the true count stay here, server-side.
        return [
            {"task_id": task_id}
            for task_id, row in EXAMPLES.items()
            if row["split"] == split
        ]

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.validated = TaskSpec.model_validate(task_spec)
        if self.validated.task_id not in EXAMPLES:
            raise ValueError(f"Unknown task_id: {self.validated.task_id}")
        self.example = EXAMPLES[self.validated.task_id]

        # Read by CiteCastBackSearch when it builds its config.
        self.search_secrets = secrets

        # Fail fast if the backdated web service is unconfigured. Without this
        # the tools would return soft "not-configured" errors, the agent would
        # flail and submit blind, and those rollouts would poison training with
        # rewards indistinguishable from honest bad forecasts.
        if WebServiceConfig.from_env(secrets) is None:
            raise ValueError(
                "Backdated web service is not configured: set OPENREWARD_API_KEY "
                "in the server process environment (or pass api_key in secrets)."
            )

        self.submitted = False

    async def get_prompt(self) -> List[TextBlock]:
        return [TextBlock(text=render_prompt(self.example))]

    @tool
    async def submit_prediction(self, params: SubmitPredictionInput) -> ToolOutput:
        """Submit your final citation-count prediction. One submission only — this ends the task."""
        if self.submitted:
            return ToolOutput(
                blocks=[TextBlock(text="A prediction was already submitted for this task.")],
                metadata={"error": "already_submitted"},
                reward=0.0,
                finished=True,
            )
        self.submitted = True

        actual = int(self.example["true_citations"])
        predicted = params.predicted_citations
        reward = compute_reward(predicted, actual)

        # The true count and band go in metadata only, never in the blocks
        # text: finished=True means the model never sees this in-episode, but
        # blocks are the part most likely to surface in a future harness or an
        # SFT-from-trajectory pipeline.
        return ToolOutput(
            blocks=[
                TextBlock(
                    text=f"Prediction recorded: {predicted} citations.\nReward: {reward:.4f}"
                )
            ],
            metadata={
                "task_id": self.validated.task_id,
                "arxiv_id": self.example["arxiv_id"],
                "predicted": predicted,
                "actual": actual,
                "band": band_of(actual),
                "reward": reward,
            },
            reward=reward,
            finished=True,
        )
