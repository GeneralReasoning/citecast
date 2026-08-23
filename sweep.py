"""
Baseline-policy reward sweep for CiteCast.

Scores blind policies against the shipped task file with the real reward
function, so the README's floor numbers trace back to individual tasks and the
farmability gate is enforced before anything ships:

  always_zero        predict 0 everywhere
  global_median      the train labels' median, everywhere
  best_constant      the single best constant prediction (HARD GATE <= 0.40)
  category_constant  best constant per primary category, fit on train
                     (soft gate <= 0.55 — above it, category alone predicts
                     the label too well and bands should be stratified within
                     category groups)
  band_oracle        best constant per true band — a context ceiling, not a
                     reachable policy (knowing the band is close to knowing
                     the answer)

Optionally (--model gpt-5.2) a closed-book model prior: the model sees only
the prompt, no tools, and must answer with an integer.

Usage:
    python sweep.py                # offline policies + gates
    python sweep.py --model gpt-5.2 --sample 60
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from constants import DATA_FILE, FARM_GATE, RESOLUTION_FLOOR, SWEEP_SEED
from grading import best_constant, compute_reward, mean_reward_for_constant


def load_rows() -> list[dict]:
    with open(DATA_FILE) as f:
        return [json.loads(line) for line in f if line.strip()]


def score(rows: list[dict], predict) -> dict:
    per_task = [
        {"task_id": r["task_id"], "predicted": predict(r), "actual": r["true_citations"],
         "reward": compute_reward(predict(r), r["true_citations"])}
        for r in rows
    ]
    return {
        "mean_reward": round(sum(t["reward"] for t in per_task) / len(per_task), 4),
        "per_task": per_task,
    }


def closed_book(rows: list[dict], model: str, sample: int) -> dict:
    """Model prior with no tools: prompt in, integer out."""
    import os
    import re

    from openai import OpenAI
    from prompts import render_prompt

    if not os.environ.get("OPENAI_API_KEY"):
        env_file = Path.home() / "Documents" / "avalon-sft" / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.removeprefix("export ").strip()
                if line.startswith("OPENAI_API_KEY="):
                    os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"')

    client = OpenAI()
    rng = random.Random(SWEEP_SEED)
    chosen = rng.sample(rows, min(sample, len(rows)))
    per_task = []
    for row in chosen:
        prompt = render_prompt(row) + (
            "\n\n(You have no tools in this run. Answer with your single best "
            "integer prediction and nothing else.)"
        )
        response = client.responses.create(model=model, input=prompt)
        text = response.output_text.strip()
        match = re.search(r"\d+", text.replace(",", ""))
        predicted = int(match.group()) if match else 0
        reward = compute_reward(predicted, row["true_citations"])
        per_task.append(
            {"task_id": row["task_id"], "predicted": predicted,
             "actual": row["true_citations"], "reward": reward}
        )
        print(f"  {row['task_id']}: predicted {predicted}, actual "
              f"{row['true_citations']}, reward {reward:.3f}")
    return {
        "mean_reward": round(sum(t["reward"] for t in per_task) / len(per_task), 4),
        "n": len(per_task),
        "per_task": per_task,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="closed-book model prior (e.g. gpt-5.2)")
    parser.add_argument("--sample", type=int, default=60)
    args = parser.parse_args()

    rows = load_rows()
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    train_labels = [r["true_citations"] for r in train]

    results: dict = {"n_train": len(train), "n_test": len(test)}

    results["always_zero"] = score(train, lambda r: 0)
    median = int(statistics.median(train_labels))
    results["global_median"] = {"constant": median, **score(train, lambda r: median)}

    const, const_reward = best_constant(train_labels)
    results["best_constant"] = {"constant": const, **score(train, lambda r: const)}

    # Per-category constants, fit on train, evaluated on train and test.
    by_cat: dict[str, list[int]] = defaultdict(list)
    for r in train:
        by_cat[r["primary_category"]].append(r["true_citations"])
    cat_const = {c: best_constant(labels)[0] for c, labels in by_cat.items()}
    fallback = const

    def cat_predict(r):
        return cat_const.get(r["primary_category"], fallback)

    results["category_constant"] = {
        "constants": cat_const,
        "train": score(train, cat_predict)["mean_reward"],
        "test": score(test, cat_predict)["mean_reward"],
    }

    # Band oracle: ceiling if the true band were known.
    by_band: dict[str, list[int]] = defaultdict(list)
    for r in train:
        by_band[r["band"]].append(r["true_citations"])
    band_const = {b: best_constant(labels)[0] for b, labels in by_band.items()}
    results["band_oracle"] = score(train, lambda r: band_const[r["band"]])["mean_reward"]

    two_x = compute_reward(9, 4)
    results["resolution_2x"] = two_x

    print(f"always_zero        {results['always_zero']['mean_reward']:.4f}")
    print(f"global_median      {results['global_median']['mean_reward']:.4f} (predict {median})")
    print(f"best_constant      {const_reward:.4f} (predict {const})  gate <= {FARM_GATE}")
    print(f"category_constant  train {results['category_constant']['train']:.4f} / "
          f"test {results['category_constant']['test']:.4f}  soft gate <= 0.55")
    print(f"band_oracle        {results['band_oracle']:.4f} (ceiling, not reachable)")
    print(f"resolution at 2x   {two_x:.4f}  floor >= {RESOLUTION_FLOOR}")

    if args.model:
        print(f"\nclosed-book prior: {args.model} on {args.sample} sampled tasks")
        results[f"closed_book_{args.model}"] = closed_book(rows, args.model, args.sample)
        print(f"closed_book mean   {results[f'closed_book_{args.model}']['mean_reward']:.4f}")

    out = Path(__file__).parent / f"sweep_{int(time.time())}.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out}")

    failures = []
    if const_reward > FARM_GATE:
        failures.append(f"best constant {const_reward:.4f} > gate {FARM_GATE}")
    if two_x < RESOLUTION_FLOOR:
        failures.append(f"2x resolution {two_x:.4f} < floor {RESOLUTION_FLOOR}")
    if results["category_constant"]["train"] > 0.55:
        print("WARNING: per-category constant clears 0.55 — category alone predicts "
              "the label too well; stratify bands within category groups.")
    if failures:
        raise SystemExit("SWEEP GATES FAILED: " + "; ".join(failures))
    print("all gates passed")


if __name__ == "__main__":
    main()
