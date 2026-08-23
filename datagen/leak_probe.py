"""
Adversarial leak probe: can the label be recovered through the agent's tools?

For the highest-cited shipped tasks (where memorization or leakage would pay
most), this runs real backdated searches and fetches at the environment's
cutoff — the same corpus fan-out the agent gets — and regex-scans everything
that comes back for citation-count figures. The probe is deliberately more
generous than the agent's surface: it scans raw API text, tries
citation-shaped queries, and fetches Semantic Scholar URLs directly.

PASS means: for every probed paper in a >=26-citation band, no recoverable
figure reaches the band's lower bound. (Small figures are expected and fine —
a paper's January citation count is legitimate forecasting signal, not the
answer.)

    python leak_probe.py --top 25
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import CITATION_BANDS, CUTOFF  # noqa: E402

DATA_FILE = Path(__file__).parent.parent / "data" / "citecast_tasks.jsonl"
BASE = "https://search.openreward.ai"
CORPUS = ["cc_arxiv", "cc_news", "cc_web"]

BAND_LO = {name: lo for name, lo, _ in CITATION_BANDS}

# Figures adjacent to citation language. Deliberately loose.
CITE_RE = re.compile(
    r"(?:cited by\s+(\d{1,6})|(\d{1,6})\s+citations?|citations?[:\s]+(\d{1,6}))",
    re.IGNORECASE,
)


def _api_key() -> str:
    import os

    key = os.environ.get("OPENREWARD_API_KEY", "").strip()
    if not key:
        env_file = Path(__file__).resolve().parent.parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("OPENREWARD_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    if not key:
        raise SystemExit("OPENREWARD_API_KEY not set and not found in ../.env")
    return key


def _post(path: str, body: dict, key: str) -> dict:
    import time

    for attempt in range(4):
        request = urllib.request.Request(
            f"{BASE}{path}",
            data=json.dumps(body).encode(),
            headers={"x-api-key": key, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:  # no capture before the cutoff — not an error
                return {}
            if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(5 * (attempt + 1))
                continue
            print(f"    probe request failed ({e.code}) for {body.get('url', body.get('query'))} — skipping")
            return {}
    return {}


def figures_in(text: str) -> list[int]:
    out = []
    for match in CITE_RE.finditer(text or ""):
        for group in match.groups():
            if group:
                out.append(int(group))
    return out


def _mentions_paper(text: str, row: dict) -> bool:
    """Does this content actually reference the probed paper?

    A citation figure only counts as recoverable when it appears in content
    that mentions the paper — otherwise every bibliometrics article and every
    highly-cited unrelated hit registers its own numbers as phantom leaks
    (first probe run: a 2009 paper's own '958 citations' flagged two 2026
    tasks it could not possibly describe).
    """
    lowered = (text or "").lower()
    if row["arxiv_id"] in lowered:
        return True
    title_words = re.sub(r"[^\w\s]", " ", row["title"].lower()).split()
    probe = " ".join(title_words[:8])
    return probe in re.sub(r"[^\w\s]", " ", lowered)


def probe_paper(row: dict, key: str) -> tuple[int, list[str]]:
    """Max citation-shaped figure recoverable ABOUT one paper, with sources."""
    found: list[tuple[int, str]] = []
    title = row["title"][:200]

    queries = [title, f"{title} citations", row["arxiv_id"]]
    urls = {
        f"https://arxiv.org/abs/{row['arxiv_id']}",
        f"https://www.semanticscholar.org/arxiv/{row['arxiv_id']}",
    }
    for query in queries:
        result = _post(
            "/search",
            {"query": query, "as_of": CUTOFF, "k": 5, "corpus": CORPUS},
            key,
        )
        for hit in result.get("hits", []):
            blob = hit.get("snippet", "") + " " + hit.get("title", "")
            if _mentions_paper(blob + " " + (hit.get("url") or ""), row):
                for fig in figures_in(blob):
                    found.append((fig, f"search snippet {hit.get('url')}"))
            urls.add(hit.get("url"))

    for url in sorted(u for u in urls if u):
        result = _post("/fetch", {"url": url, "as_of": CUTOFF}, key)
        text = result.get("text", "")
        if not _mentions_paper(text + " " + url, row):
            continue
        for fig in figures_in(text):
            found.append((fig, f"fetch {url}"))

    if not found:
        return 0, []
    top = max(found, key=lambda pair: pair[0])
    return top[0], [f"{fig} via {src}" for fig, src in sorted(found, reverse=True)[:3]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--ids", nargs="*", help="probe only these arxiv ids")
    args = parser.parse_args()
    key = _api_key()

    with open(DATA_FILE) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows.sort(key=lambda r: r["true_citations"], reverse=True)
    if args.ids:
        probed = [r for r in rows if r["arxiv_id"] in set(args.ids)]
    else:
        probed = rows[: args.top]

    failures = []
    for row in probed:
        max_fig, sources = probe_paper(row, key)
        lo = BAND_LO[row["band"]]
        leak = row["band"] in ("26-50", "51-100", "101+") and max_fig >= lo
        status = "LEAK" if leak else "ok"
        print(f"{row['arxiv_id']} band {row['band']:>6} true {row['true_citations']:>4} "
              f"max recoverable {max_fig:>4}  {status}")
        for s in sources:
            print(f"    {s}")
        if leak:
            failures.append(row["arxiv_id"])

    print(f"\nprobed {len(probed)} papers; leaks: {len(failures)}")
    if failures:
        raise SystemExit(f"LEAK PROBE FAILED: {failures}")
    print("leak probe passed: no post-cutoff citation figure recoverable")


if __name__ == "__main__":
    main()
