# CiteCast

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/CiteCast)

## Description

CiteCast is a forecasting environment: given an arXiv paper announced in January 2026, the agent predicts the total number of citations Semantic Scholar records for it on 2026-08-23. The agent sees the paper's version-1 metadata (title, abstract, authors, primary category, announcement date) and researches through backdated web search and fetch tools frozen at 2026-01-31 — it can study the paper's earliest reception, the authors' track records, and the state of the field, but can never observe the future it is asked to predict. The point-in-time cutoff is enforced server-side on every request over a frozen crawl-dated corpus that includes an arXiv sub-corpus, news, and general web.

## Capabilities

- Forecasting and calibration under genuine uncertainty (the target is a real-world outcome, not a synthetic label)
- Judging research quality and likely impact from a paper's content
- Web research: author track records, topic momentum, early reception, citation velocity of comparable work
- Reasoning about heavy-tailed outcome distributions in log space

## Compute Requirements

There is no sandbox: the environment exposes only backdated `web_search`/`web_fetch` and a typed submission tool, so the default container resources are ample.

## License

MIT.

## Tasks

2,200 tasks (2,000 train, 200 test), one per paper, drawn from the 22,483 papers announced on arXiv in January 2026 (harvested from the arXiv API; 2601.* identifiers only, so every paper was publicly visible before the tool cutoff). Tasks are stratified across citation bands so the target is not degenerate — roughly half of all January 2026 papers had zero citations by the snapshot date, but the shipped mix keeps most tasks in ranges where forecasting skill matters:

| citation band | tasks | share of eligible pool |
|---|---|---|
| 0 | 650 | 650 of 11,535 |
| 1–2 | 390 | 390 of 7,233 |
| 3–5 | 285 | 285 of 2,343 |
| 6–10 | 401 | 401 of 834 |
| 11–25 | 350 | 350 of 366 |
| 26–50 | 85 | 85 of 90 |
| 51–100 | 23 | 23 of 25 |
| 101+ | 16 | 16 of 17 |

The train/test split is a deterministic within-band hash of the arXiv id, so every band is represented in the test split. Task specs sent to agents carry only a task id: the paper metadata, the citation band, and the true count stay server-side.

## Reward Structure

Dense, deterministic, single-shot. The agent calls `submit_prediction` exactly once with an integer $p$; for true count $t$ the reward is

$$ r = \frac{1}{\cosh\left(2 \cdot \left|\ln\frac{p+1}{t+1}\right|\right)} $$

which is 1.0 at an exact hit, ~0.75 at 1.5x error, ~0.47 at 2x, ~0.22 at 3x, and ~0.02 at 10x — symmetric in log space, so overshooting by 2x costs the same as undershooting by 2x. The task mix was grid-searched so that blind policies stay low; measured floors on the shipped train split:

| blind policy | mean reward (train) |
|---|---|
| always predict 0 | 0.383 |
| global median (predict 3) | 0.368 |
| best single constant (0) | 0.383 |
| best constant per primary category | 0.524 (0.473 on test) |
| band oracle (best constant per true band — near-answer knowledge, not a reachable policy) | 0.950 |

Every number traces to per-task rewards in the committed sweep output (`sweep.py`).

We do not use LLM graders for this task.

## Data

The cohort was harvested from the arXiv API (day-sliced submittedDate queries over January 2026) and every task's ground truth is the Semantic Scholar `citationCount` for the paper, snapshotted in a single batch pass on 2026-08-23 and frozen into the task file. Prompts carry version-1 metadata fetched explicitly (`id_list=<id>v1`), because later arXiv revisions can carry post-cutoff information ("accepted at ICML 2026"); mutable article-level fields (comments, journal reference, DOI) are excluded entirely. OpenAlex counts are recorded in provenance as a secondary reference but never touch grading — OpenAlex substantially undercounts citations for this cohort. Withdrawn papers, papers missing from Semantic Scholar, and duplicates (by Semantic Scholar paper id and by normalised title) were excluded. Data files are stored on the OpenReward platform.

## Tools

- `web_search` — backdated search over the frozen corpus (arXiv, news, general web sub-corpora), returning titles, URLs, and snippets. The cutoff (2026-01-31) is injected server-side per request; it is not an agent-visible parameter.
- `web_fetch` — fetches a page as it existed on or before the cutoff (by crawl date); returns extracted text.
- `submit_prediction` — one integer prediction plus a short free-text justification (recorded, not graded). First call ends the task.

## Time Horizon

Single-episode, short-horizon: research then one submission. In a 5-task local smoke run, gpt-5.2 made 4–8 web tool calls per task (median 6) across 3–4 model turns before submitting.

## Environment Difficulty

The band-oracle ceiling (best constant per true band — close to knowing the answer) is 0.950 mean reward, against a best-blind-constant floor of 0.383; the spread between them is the skill the environment measures. In a 5-task local smoke run (too small to be a benchmark statistic), an untuned gpt-5.2 agent scored 0.294 mean reward: it correctly upweighted the one paper with genuine early buzz (predicted 22 against a true 15, reward 0.78) but systematically overshot unremarkable papers (18 against 3, 6 against 1, 3 against 0) — exactly the calibration the reward is designed to teach. The per-category-constant baseline (0.473 on test) marks the bar a policy clears with field base rates alone.

## Other Environment Requirements

No external secrets are required.

## Safety

The agent forecasts a public scholarly metric for published papers using an archived, read-only web corpus; it cannot affect the outcome it predicts, and no live web access occurs. Predictions are graded arithmetically against recorded ground truth. The main methodological caveat is memorization rather than safety: a policy whose pretraining extends past the snapshot date may remember which January 2026 papers became famous. The environment bounds that edge structurally — papers with more than 25 citations are 5.6% of the mix, and the other 94.4% are papers whose citation counts no corpus states — and the environment is sound as a forecasting benchmark only for policies with a data cutoff at or before 2026-08-23. A leak probe (datagen/leak_probe.py) ran real backdated searches and fetches for the 25 highest-cited tasks and regex-scanned everything returned for citation figures: no recoverable figure about any probed paper reached its band's lower bound, and the largest figure found at all was 10 — a January-level early count for a paper that finished at 92, which is exactly the early-velocity signal the environment intends agents to use.

## Citations

```bibtex
@dataset{GRCiteCast,
  author    = {General Reasoning Inc. Team},
  title     = {CiteCast},
  year      = {2026},
  publisher = {OpenReward},
  url       = {https://openreward.ai/GeneralReasoning/CiteCast}
}
```
