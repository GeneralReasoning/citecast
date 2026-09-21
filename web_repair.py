"""Hardening for the SDK's ``web_fetch``, vendored into each environment repo.

Two defects, both measured against the live backdated service on 2026-09-21.

**Respelled URLs 404.** ``/fetch`` matches URLs byte-for-byte, so a page the
archive holds under one spelling is a 404 under another. Taking 46 URLs that
search returned and that fetched cleanly, then applying the edits a model
actually makes, 75 of 84 respellings 404: percent-encoding a non-ASCII path
broke 9 of 9, adding a trailing slash 33 of 38, dropping one 8 of 8, adding
``www.`` 25 of 29. The ladder in :func:`url_variants` recovered 100% of them.
This is not a coverage problem: on a random 120-URL sample of real search
results, fetch succeeds 98.3% as-is and the ladder never fires.

**Every fetch ships the page twice.** ``build_fetch_output`` puts the full text
into ``data["content"]`` and ``to_tool_output`` copies that into the ToolOutput
metadata, so the same bytes travel in ``blocks`` and in ``metadata``. Nothing
reads the copy. The training harness caps environment tool output at 32,768
bytes and replaces the *entire* result with
"[env tool output exceeded cap before content rendering]" when that is
exceeded, so the duplicate alone can cost the agent the page it just fetched.
One real 65,550-byte fetch was exactly half mirror.

Note this does not cap page length, so a page longer than roughly 32 KB still
exceeds the harness budget on its own. That is a separate decision.

Usage, stock toolset::

    from web_repair import RepairingBackSearchToolset
    class MyEnv(Environment):
        toolsets = [RepairingBackSearchToolset]

Usage, an environment that already subclasses and overrides ``web_fetch``::

    from web_repair import drop_content_mirror, is_not_archived, not_archived_output, url_variants
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional
from urllib.parse import unquote, urlsplit, urlunsplit

from openreward.environments import TextBlock, ToolOutput, tool
from openreward.toolsets import BackSearchToolset, WebToolset
from openreward.toolsets._web_common import WebFetchParams

#: Cap on repair attempts, so a genuine miss costs at most this many extra calls.
MAX_REPAIR_ATTEMPTS = 3


def url_variants(url: str) -> list[str]:
    """Cosmetic respellings of ``url`` worth retrying after a 404.

    Percent-decoding is applied first and the toggles build on the decoded
    form, so a URL that is both re-encoded *and* slash-mangled is still
    repaired inside the attempt budget.
    """
    parts = urlsplit(url)
    decoded = unquote(parts.path)
    variants: list[str] = []
    base = url
    if decoded != parts.path:
        base = urlunsplit((parts.scheme, parts.netloc, decoded, parts.query, parts.fragment))
        variants.append(base)
    p = urlsplit(base)
    if p.path and p.path != "/":
        flipped = p.path[:-1] if p.path.endswith("/") else p.path + "/"
        variants.append(urlunsplit((p.scheme, p.netloc, flipped, p.query, p.fragment)))
    host = p.netloc[4:] if p.netloc.startswith("www.") else "www." + p.netloc
    variants.append(urlunsplit((p.scheme, host, p.path, p.query, p.fragment)))
    seen = {url}
    return [v for v in variants if not (v in seen or seen.add(v))][:MAX_REPAIR_ATTEMPTS]


def is_not_archived(out: ToolOutput) -> bool:
    """True only for "the archive has no capture of this URL".

    Deliberately narrow: a backend outage, a blocked domain, an unparseable URL
    or a quota failure must not trigger retries against respellings.
    """
    if (out.metadata or {}).get("error") != "web-service-error":
        return False
    text = out.blocks[0].text if out.blocks else ""
    return "HTTP 404" in text


def not_archived_output(url: str) -> ToolOutput:
    """An actionable replacement for the backend's raw 404 envelope.

    The stock message reads as an infrastructure failure and leaks internal
    corpus names, so agents re-fetch the same dead URL or guess neighbouring
    ones until the turn cap. Of 18 real fetch failures observed in rollouts,
    9 were URLs the model invented rather than ones search returned, which is
    the behaviour this wording exists to stop.
    """
    return ToolOutput(
        blocks=[TextBlock(text=(
            f"Error [page-not-archived]: No archived capture of {url}. The archive only "
            f"serves pages it captured, so this URL may never have been captured, or may "
            f"not exist. Do not retry this URL or guess variations of it - choose a "
            f"different result from web_search instead."
        ))],
        metadata={"error": "page-not-archived", "url": url},
        reward=0.0,
        finished=False,
    )


def drop_content_mirror(out: ToolOutput) -> ToolOutput:
    """Strip ``metadata["content"]``, a byte-identical copy of the block text."""
    if not out.metadata or "content" not in out.metadata:
        return out
    slim = {k: v for k, v in out.metadata.items() if k != "content"}
    return ToolOutput(
        blocks=out.blocks,
        metadata=slim or None,
        reward=out.reward,
        finished=out.finished,
    )


async def repaired_fetch(
    toolset: Any,
    base_fetch: Callable[[Any, WebFetchParams], Awaitable[ToolOutput]],
    params: WebFetchParams,
) -> ToolOutput:
    """Run ``base_fetch``, repairing a 404 against respellings of the URL."""
    out = drop_content_mirror(await base_fetch(toolset, params))
    if not is_not_archived(out):
        return out
    for candidate in url_variants(params.url):
        retry = drop_content_mirror(
            await base_fetch(toolset, WebFetchParams(url=candidate, prompt=params.prompt))
        )
        if not (retry.metadata or {}).get("error"):
            return retry
    return not_archived_output(params.url)


class RepairingBackSearchToolset(BackSearchToolset):
    """``BackSearchToolset`` with the fetch hardening described in this module."""

    @tool
    async def web_fetch(self, params: WebFetchParams) -> ToolOutput:
        return await repaired_fetch(self, BackSearchToolset.web_fetch, params)


class RepairingWebToolset(WebToolset):
    """``WebToolset`` with the fetch hardening described in this module."""

    @tool
    async def web_fetch(self, params: WebFetchParams) -> ToolOutput:
        return await repaired_fetch(self, WebToolset.web_fetch, params)


# Keep the model-facing tool descriptions identical to the bases'. Each class
# owns a distinct function object, so these do not overwrite one another.
RepairingBackSearchToolset.web_fetch.__doc__ = BackSearchToolset.web_fetch.__doc__
RepairingWebToolset.web_fetch.__doc__ = WebToolset.web_fetch.__doc__

__all__ = [
    "MAX_REPAIR_ATTEMPTS", "RepairingBackSearchToolset", "RepairingWebToolset",
    "drop_content_mirror", "is_not_archived", "not_archived_output",
    "repaired_fetch", "url_variants",
]
