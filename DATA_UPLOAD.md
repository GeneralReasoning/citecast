# CiteCast data upload

The environment reads one file, `citecast_tasks.jsonl`, from `/orwd_data` in production (falling back to `data/` locally — see `constants.py`).

## Upload

`orwd upload` preserves the path it is given, so upload from **inside** `data/` — uploading from the env root would place the file at `/orwd_data/data/citecast_tasks.jsonl` while `constants.py` looks at `/orwd_data/citecast_tasks.jsonl`, and the server would fail at import.

```bash
cd data
orwd upload GeneralReasoning/CiteCast citecast_tasks.jsonl
orwd files GeneralReasoning/CiteCast   # verify: file at the mount root
```

## Row schema

One JSON object per line, 2,200 rows (2,000 train + 200 test). Every row carries every key, in this order:

| key | type | meaning |
|---|---|---|
| `task_id` | str | `cc_<arxiv_id>` — deliberately carries no citation-band marker |
| `split` | str | `train` or `test` (deterministic within-band hash split, salt in `constants.SPLIT_SALT`) |
| `band` | str | citation band the true count falls in (server-side only, never sent to agents) |
| `arxiv_id` | str | `2601.*` — announced on arXiv in January 2026 |
| `announced` | str | version-1 announcement date, `YYYY-MM-DD` |
| `title` | str | version-1 title (later revisions can carry post-cutoff information) |
| `abstract` | str | version-1 abstract |
| `authors` | list[str] | version-1 author list |
| `author_count` | int | length of the full author list |
| `primary_category` | str | arXiv primary category |
| `true_citations` | int | Semantic Scholar `citationCount` on the snapshot day — the label |
| `s2_paper_id` | str | Semantic Scholar paper id (dedup key across arXiv ids) |
| `snapshot_utc` | str | ISO timestamp of the label fetch; the date part equals `constants.SNAPSHOT` for every row |
| `provenance` | dict | `s2_publication_date`, `openalex_citations` (secondary, never grades), harvest and metadata source notes |

## Rebuilding

```bash
cd datagen
python build.py --harvest     # January 2026 cohort from the arXiv API (~24k papers)
python build.py --labels     # Semantic Scholar citation snapshot — ONE calendar day
python build.py --census     # band pools + farmability feasibility
python build.py --enrich     # v1 metadata + OpenAlex for sampled candidates
python build.py --assemble   # exclusions, stratified sampling, split, ship
```

A rebuild will not reproduce the shipped file byte for byte, and cannot: `true_citations` is a time-varying quantity frozen on the snapshot day, and a later `--labels` run snapshots a different day (delete `datagen/cache/` first, then update `constants.SNAPSHOT` to the new date). What a rebuild must reproduce is the guarantees, and `datagen/verify.py` checks exactly those: structural invariants offline, and — with `--sample N` — live drift-tolerant label checks (a live count may grow freely and shrink only within merge tolerance) plus byte-equality of the v1 prompt metadata, which is immutable.
