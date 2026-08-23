"""
Shared source clients for CiteCast datagen.

Three sources, all keyless:

  * The arXiv API (export.arxiv.org/api/query) — the January 2026 cohort
    (day-sliced submittedDate queries) and the version-1 metadata that goes
    into prompts. v1 is requested explicitly (id_list=<id>v1) because later
    revisions can carry post-cutoff information ("accepted at ICML 2026").
  * Semantic Scholar Graph API — the ground-truth citation counts, fetched
    through the batch endpoint (500 ids per POST). The keyless pool 429s
    freely, so the client backs off patiently.
  * OpenAlex — secondary citation counts recorded in provenance only. It
    undercounts this cohort badly and never touches grading or banding.

No language model touches any label: a task's ground truth is exactly the
citationCount Semantic Scholar reported on the snapshot day.

All responses are cached to datagen/cache/ keyed by the request, so reruns and
the independent verify pass are cheap and don't re-hammer the APIs.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

USER_AGENT = "GR-OR-Envs-CiteCast/1.0 (https://gr.inc; ross@gr.inc)"

ARXIV_API = "https://export.arxiv.org/api/query"
S2_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"
OPENALEX_WORKS = "https://api.openalex.org/works"

CACHE_DIR = Path(__file__).parent / "cache"

# Politeness delay between live requests to the same endpoint. arXiv asks for
# 3s; the keyless Semantic Scholar pool is ~1 rps shared (the 429 backoff does
# the real pacing); OpenAlex tolerates 10 rps with a mailto.
_MIN_INTERVAL = {ARXIV_API: 3.0, S2_BATCH: 2.0, OPENALEX_WORKS: 0.15}
_last_request: dict[str, float] = {}

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"
_OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"


class SourceError(RuntimeError):
    """A source request failed after all retries."""


# --------------------------------------------------------------------------
# Cached HTTP
# --------------------------------------------------------------------------


def _cache_path(endpoint: str, payload: str) -> Path:
    key = hashlib.sha256(f"{endpoint}\n{payload}".encode()).hexdigest()
    return CACHE_DIR / key[:2] / f"{key}.json.gz"


def _read_cache(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(value, f)
    tmp.replace(path)


def _throttle(endpoint: str) -> None:
    interval = _MIN_INTERVAL.get(endpoint, 0.2)
    last = _last_request.get(endpoint)
    if last is not None:
        wait = interval - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
    _last_request[endpoint] = time.monotonic()


def _request(
    endpoint: str,
    *,
    params: dict[str, str] | None = None,
    post_json: Any | None = None,
    max_attempts: int = 10,
    use_cache: bool = True,
    cacheable: Any = None,
) -> str:
    """GET (params) or POST-JSON an endpoint with disk caching and backoff.

    429s back off exponentially and generously — the keyless Semantic Scholar
    pool is shared and saturates for minutes at a time. ``cacheable`` is a
    predicate over the raw body deciding whether the response is worth
    caching; the arXiv API intermittently returns valid-but-empty feeds that
    must not poison the cache.
    """
    if params is not None:
        payload = urllib.parse.urlencode(sorted(params.items()))
        url = f"{endpoint}?{payload}"
        body_bytes = None
        cache_key = payload
    else:
        url = endpoint
        body_bytes = json.dumps(post_json, sort_keys=True).encode()
        cache_key = body_bytes.decode()

    path = _cache_path(endpoint, cache_key)
    if use_cache:
        cached = _read_cache(path)
        if cached is not None:
            return cached

    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        _throttle(endpoint)
        request = urllib.request.Request(
            url,
            data=body_bytes,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "*/*",
                **({"Content-Type": "application/json"} if body_bytes else {}),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = response.read().decode("utf-8")
            if use_cache and (cacheable is None or cacheable(body)):
                _write_cache(path, body)
            return body
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code == 429:
                wait = min(8 + attempt * 8, 120)
                print(f"  SOURCE 429: {endpoint} | backing off {wait}s (attempt {attempt + 1})")
                time.sleep(wait)
                continue
            if e.code >= 500 and attempt < max_attempts - 1:
                wait = min(2**attempt, 60)
                print(f"  SOURCE RETRY: {endpoint} | HTTP {e.code} | {wait}s (attempt {attempt + 1})")
                time.sleep(wait)
                continue
            raise SourceError(f"{endpoint} failed: HTTP {e.code}") from e
        except Exception as e:  # noqa: BLE001 - urllib raises a wide range
            last_exc = e
            if attempt < max_attempts - 1:
                wait = min(2**attempt, 60)
                print(f"  SOURCE RETRY: {endpoint} | {e} | {wait}s (attempt {attempt + 1})")
                time.sleep(wait)

    raise SourceError(f"{endpoint} failed after {max_attempts} attempts: {last_exc}")


# --------------------------------------------------------------------------
# arXiv
# --------------------------------------------------------------------------


def _collapse(text: str | None) -> str:
    """Atom feeds hard-wrap titles and abstracts; collapse to single spaces."""
    return re.sub(r"\s+", " ", text or "").strip()


def parse_atom(xml_text: str) -> tuple[int, list[dict[str, Any]]]:
    """(opensearch totalResults, entries) from an arXiv API Atom feed."""
    root = ET.fromstring(xml_text)
    total_el = root.find(f"{_OPENSEARCH}totalResults")
    total = int(total_el.text) if total_el is not None and total_el.text else 0
    entries: list[dict[str, Any]] = []
    for entry in root.findall(f"{_ATOM}entry"):
        raw_id = entry.findtext(f"{_ATOM}id") or ""
        # 'http://arxiv.org/abs/2601.03192v1' -> ('2601.03192', 'v1')
        match = re.search(r"/abs/([0-9.]+)(v\d+)?$", raw_id)
        if not match:
            continue
        primary = entry.find(f"{_ARXIV_NS}primary_category")
        entries.append(
            {
                "arxiv_id": match.group(1),
                "version": match.group(2) or "",
                "title": _collapse(entry.findtext(f"{_ATOM}title")),
                "abstract": _collapse(entry.findtext(f"{_ATOM}summary")),
                "authors": [
                    _collapse(a.findtext(f"{_ATOM}name"))
                    for a in entry.findall(f"{_ATOM}author")
                ],
                "primary_category": primary.get("term") if primary is not None else "",
                "published": entry.findtext(f"{_ATOM}published") or "",
                "updated": entry.findtext(f"{_ATOM}updated") or "",
            }
        )
    return total, entries


def _arxiv_feed_ok(body: str) -> bool:
    """Cache predicate: don't keep transiently empty feeds for populated queries."""
    try:
        total, entries = parse_atom(body)
    except ET.ParseError:
        return False
    return total == 0 or bool(entries)


def arxiv_search(query: str, start: int, max_results: int) -> tuple[int, list[dict[str, Any]]]:
    """One page of an arXiv API search, with the empty-feed retry.

    The API intermittently returns an empty feed for a query that has results;
    those responses are never cached (see _arxiv_feed_ok) and are retried here.
    """
    params = {
        "search_query": query,
        "start": str(start),
        "max_results": str(max_results),
    }
    for attempt in range(5):
        body = _request(ARXIV_API, params=params, cacheable=_arxiv_feed_ok)
        total, entries = parse_atom(body)
        if total == 0 or entries:
            return total, entries
        wait = 5 * (attempt + 1)
        print(f"  ARXIV EMPTY FEED: start={start} | retrying in {wait}s")
        time.sleep(wait)
    raise SourceError(f"arXiv returned an empty feed 5 times for {query} start={start}")


def arxiv_day_slice(day: str) -> list[dict[str, Any]]:
    """Every paper submitted on one day ('2026-01-05'), paged 500 at a time."""
    compact = day.replace("-", "")
    query = f"submittedDate:[{compact}0000 TO {compact}2359]"
    out: list[dict[str, Any]] = []
    start = 0
    while True:
        total, entries = arxiv_search(query, start, 500)
        out.extend(entries)
        start += 500
        if start >= total or not entries:
            break
    return out


def arxiv_metadata_by_id(versioned_ids: list[str]) -> list[dict[str, Any]]:
    """Metadata for explicit (versioned) ids, batched through id_list.

    Requesting '<id>v1' returns version 1's own title/abstract/authors — the
    property that keeps post-cutoff revisions out of prompts.
    """
    out: list[dict[str, Any]] = []
    for i in range(0, len(versioned_ids), 100):
        chunk = versioned_ids[i : i + 100]
        params = {"id_list": ",".join(chunk), "max_results": str(len(chunk))}
        body = _request(ARXIV_API, params=params, cacheable=_arxiv_feed_ok)
        _, entries = parse_atom(body)
        out.extend(entries)
    return out


_WITHDRAWN_RE = re.compile(
    r"\bwithdrawn\b|\bwithdraw this (paper|article|manuscript|submission)\b",
    re.IGNORECASE,
)


def looks_withdrawn(latest_entry: dict[str, Any]) -> bool:
    """Withdrawal marker in the LATEST version's title or abstract.

    v1 text predates any withdrawal, so the check runs against the harvest
    metadata (which the API serves at the latest version) rather than the v1
    text that ships in prompts.
    """
    return bool(
        _WITHDRAWN_RE.search(latest_entry.get("title", ""))
        or _WITHDRAWN_RE.search(latest_entry.get("abstract", "")[:500])
    )


def normalise_title(title: str) -> str:
    """Casefold, strip punctuation, collapse whitespace — the dedup key."""
    cleaned = re.sub(r"[^\w\s]", " ", title)
    return re.sub(r"\s+", " ", cleaned).strip().casefold()


# --------------------------------------------------------------------------
# Semantic Scholar
# --------------------------------------------------------------------------


def s2_batch(arxiv_ids: list[str], *, use_cache: bool = True) -> list[dict[str, Any] | None]:
    """citationCount (+paperId, externalIds) for up to 500 arXiv ids.

    Returns one element per input id, None where Semantic Scholar has no
    record. Order matches the input.
    """
    if len(arxiv_ids) > 500:
        raise ValueError("Semantic Scholar batch endpoint takes at most 500 ids")
    endpoint = f"{S2_BATCH}?fields=citationCount,paperId,externalIds,publicationDate"
    body = _request(
        endpoint,
        post_json={"ids": [f"ARXIV:{a}" for a in arxiv_ids]},
        use_cache=use_cache,
    )
    return json.loads(body)


# --------------------------------------------------------------------------
# OpenAlex (secondary, provenance only)
# --------------------------------------------------------------------------


def openalex_counts(arxiv_ids: list[str]) -> dict[str, int]:
    """cited_by_count per arXiv id via the 10.48550 DOI, 50 ids per request.

    Missing works are simply absent from the result. Recorded in provenance
    only — OpenAlex undercounts this cohort and never touches grading.
    """
    out: dict[str, int] = {}
    for i in range(0, len(arxiv_ids), 50):
        chunk = arxiv_ids[i : i + 50]
        dois = "|".join(f"10.48550/arxiv.{a}" for a in chunk)
        params = {
            "filter": f"doi:{dois}",
            "select": "doi,cited_by_count",
            "per-page": "50",
            "mailto": "ross@gr.inc",
        }
        body = _request(OPENALEX_WORKS, params=params)
        for work in json.loads(body).get("results", []):
            doi = (work.get("doi") or "").lower()
            match = re.search(r"10\.48550/arxiv\.([0-9.]+)", doi)
            if match:
                out[match.group(1)] = int(work.get("cited_by_count", 0))
    return out
