"""
inspect.py — OTTER dataset summary
Loads the four LongBench .jsonl files and prints per-subset statistics.
"""

import json
import csv
import os
import statistics
from pathlib import Path

DATA_DIR = Path(__file__).parent
SUBSETS = ["qasper", "multifieldqa_en", "qmsum", "multi_news"]
OUTPUT_CSV = DATA_DIR / "dataset_summary.csv"


def word_count(text: str) -> int:
    return len(text.split()) if isinstance(text, str) and text.strip() else 0


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_stats(values: list[int]) -> dict:
    return {
        "min": min(values),
        "mean": round(statistics.mean(values), 1),
        "max": max(values),
    }


rows = []

for subset in SUBSETS:
    path = DATA_DIR / f"longbench_{subset}.jsonl"
    if not path.exists():
        print(f"[SKIP] {path} not found")
        continue

    records = load_jsonl(path)
    n = len(records)

    # LongBench schema: 'context' and 'input' (query)
    ctx_lengths = [word_count(r.get("context", "")) for r in records]
    qry_lengths = [word_count(r.get("input", "")) for r in records]

    ctx = compute_stats(ctx_lengths)
    qry = compute_stats(qry_lengths)

    print(f"\n{'='*50}")
    print(f"Subset : {subset}")
    print(f"Examples: {n}")
    print(f"Context length (words) — min: {ctx['min']:,}  mean: {ctx['mean']:,}  max: {ctx['max']:,}")
    print(f"Query   length (words) — min: {qry['min']:,}  mean: {qry['mean']:,}  max: {qry['max']:,}")

    rows.append({
        "subset": subset,
        "n_examples": n,
        "ctx_min": ctx["min"],
        "ctx_mean": ctx["mean"],
        "ctx_max": ctx["max"],
        "qry_min": qry["min"],
        "qry_mean": qry["mean"],
        "qry_max": qry["max"],
    })

print(f"\n{'='*50}")

fieldnames = ["subset", "n_examples",
              "ctx_min", "ctx_mean", "ctx_max",
              "qry_min", "qry_mean", "qry_max"]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSummary CSV saved to: {OUTPUT_CSV}")
