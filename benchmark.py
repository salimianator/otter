"""
benchmark.py — OTTER benchmark harness

Evaluates OTTER compression quality against a no-compression baseline
using Qwen2.5-3B-Instruct as the reader model.

Usage:
    # OTTER mode (3-example dry run)
    python benchmark.py --subset qasper --mode otter --max 3

    # Baseline mode
    python benchmark.py --subset qasper --mode baseline

    # Summarise both after completion
    python benchmark.py --subset qasper --summarise
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

# ── make src/ importable ─────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from compress import OTTERCompressor   # noqa: E402
from evaluate import QwenEvaluator     # noqa: E402
from score    import (                 # noqa: E402
    compute_f1,
    compute_rouge_l,
    select_metric,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_WORDS   = 1_500           # ~2 000 tokens — keeps Qwen latency under ~25s/ex
RESULTS_DIR = ROOT / "results"
DATA_DIR    = ROOT / "data"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def truncate_to_token_limit(text: str, max_words: int = MAX_WORDS) -> str:
    """
    Keep the first *max_words* whitespace-delimited words of *text*.

    One word ≈ 1.3 subword tokens on average; 24 000 words ≈ 31 200 tokens,
    comfortably under the 32k token context limit.
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def load_jsonl(path: Path) -> list[dict]:
    """
    Load a .jsonl file into a list of dicts.

    Handles LongBench's ``answers`` field which is sometimes stored as a
    JSON-encoded string rather than a native list.
    """
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if isinstance(rec.get("answers"), str):
                rec["answers"] = json.loads(rec["answers"])
            records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    subset: str,
    mode: str,
    max_examples: int | None = None,
    resume: bool = True,
) -> Path:
    """
    Run the benchmark for *mode* (``'otter'`` or ``'baseline'``) on *subset*.

    Results are written to ``results/{mode}_{subset}.jsonl`` one line at a
    time so progress is never lost if the process is interrupted.

    Parameters
    ----------
    subset : str
        LongBench subset name (e.g. ``'qasper'``).
    mode : str
        ``'otter'`` (compress then answer) or ``'baseline'`` (answer as-is).
    max_examples : int or None
        Cap the number of examples processed; None means all.
    resume : bool
        If True, skip examples whose ``id`` already appears in the output
        file (allows resuming interrupted runs).

    Returns
    -------
    Path
        Path to the output JSONL file.
    """
    if mode not in {"otter", "baseline"}:
        raise ValueError(f"mode must be 'otter' or 'baseline', got {mode!r}")

    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / f"{mode}_{subset}.jsonl"
    data_path   = DATA_DIR / f"longbench_{subset}.jsonl"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    examples = load_jsonl(data_path)
    if max_examples is not None:
        examples = examples[:max_examples]

    total = len(examples)

    # ── resume logic ─────────────────────────────────────────────────────────
    completed_ids: set[int] = set()
    if resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    completed_ids.add(json.loads(line)["id"])
        if completed_ids:
            print(f"Resuming — {len(completed_ids)} example(s) already done.")

    # ── model initialisation ─────────────────────────────────────────────────
    compressor: OTTERCompressor | None = None
    if mode == "otter":
        print("Initialising OTTERCompressor …")
        compressor = OTTERCompressor()

    print("Loading Qwen2.5-3B-Instruct …")
    evaluator = QwenEvaluator(load_on_init=True)

    metric       = select_metric(subset)
    metric_label = metric.upper()
    skipped      = len(completed_ids)
    todo         = total - skipped

    print(f"\nRunning {mode.upper()} benchmark on {subset}")
    print(f"  total={total}  already_done={skipped}  to_run={todo}  metric={metric_label}\n")

    # ── running accumulators for postfix display ──────────────────────────────
    scores_so_far:       list[float] = []
    latencies_so_far:    list[float] = []
    compressions_so_far: list[float] = []

    # ── main loop ────────────────────────────────────────────────────────────
    bar = tqdm(
        examples,
        desc=f"{mode}",
        unit="ex",
        total=total,
        initial=skipped,          # ETA is based on remaining work only
        dynamic_ncols=True,
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt}"
            " [{elapsed}<{remaining}, {rate_fmt}]{postfix}"
        ),
    )

    with open(output_path, "a", encoding="utf-8") as out_fh:
        for ex_idx, record in enumerate(bar):
            if ex_idx in completed_ids:
                continue

            context = record.get("context", "")
            query   = record.get("input", "")
            answers = record.get("answers", [])

            # (b) Compress or pass through
            if mode == "otter":
                result            = compressor.compress(context, query)
                input_text        = result["compressed"]
                compression_ratio = result["compression_ratio"]
                token_reduction   = result["token_reduction_pct"]
            else:
                input_text        = context
                compression_ratio = 1.0
                token_reduction   = 0.0

            input_text = truncate_to_token_limit(input_text)

            # (c) Generate answer
            t0        = time.time()
            generated = evaluator.answer(input_text, query)
            latency   = time.time() - t0

            # (d) Score
            if metric == "f1":
                f1      = compute_f1(generated, answers)
                rouge_l = None
            else:
                rouge_l = compute_rouge_l(generated, answers)
                f1      = None

            score_val = f1 if f1 is not None else rouge_l

            # (e) Build result record
            out_record = {
                "id":                  ex_idx,
                "query":               query,
                "ground_truth":        answers,
                "generated":           generated,
                "compression_ratio":   round(compression_ratio, 4),
                "token_reduction_pct": round(token_reduction, 2),
                "f1":                  round(f1, 4)      if f1      is not None else None,
                "rouge_l":             round(rouge_l, 4) if rouge_l is not None else None,
                "latency_s":           round(latency, 2),
            }

            # (f) Append immediately
            out_fh.write(json.dumps(out_record) + "\n")
            out_fh.flush()

            # (g) Update running accumulators
            scores_so_far.append(score_val)
            latencies_so_far.append(latency)
            compressions_so_far.append(compression_ratio)

            mean_score       = sum(scores_so_far)       / len(scores_so_far)
            mean_latency     = sum(latencies_so_far)    / len(latencies_so_far)
            mean_compression = sum(compressions_so_far) / len(compressions_so_far)

            # Update the bar's postfix (shown on the same line as the bar)
            bar.set_postfix(ordered_dict={
                f"mean_{metric_label}": f"{mean_score:.3f}",
                "last":                 f"{score_val:.3f}",
                "cmp":                  f"{mean_compression:.0%}",
                "lat":                  f"{mean_latency:.1f}s",
            })

    bar.close()
    print(f"\nResults written to {output_path}")
    print(
        f"Final averages — "
        f"mean {metric_label}: {sum(scores_so_far)/max(len(scores_so_far),1):.4f} | "
        f"mean compression: {sum(compressions_so_far)/max(len(compressions_so_far),1):.1%} | "
        f"mean latency: {sum(latencies_so_far)/max(len(latencies_so_far),1):.1f}s"
    )
    return output_path


# ---------------------------------------------------------------------------
# Summariser
# ---------------------------------------------------------------------------

def summarise(subset: str) -> dict:
    """
    Compare OTTER vs baseline results for *subset* and print a table.

    Loads ``results/otter_{subset}.jsonl`` and
    ``results/baseline_{subset}.jsonl``, computes per-mode averages,
    prints a comparison table, and saves a CSV summary.

    Parameters
    ----------
    subset : str
        LongBench subset name.

    Returns
    -------
    dict
        Summary statistics for both modes.
    """
    metric = select_metric(subset)
    metric_key = "f1" if metric == "f1" else "rouge_l"
    metric_label = "Mean F1" if metric == "f1" else "Mean ROUGE-L"

    def _load_results(path: Path) -> list[dict]:
        if not path.exists():
            return []
        with open(path) as fh:
            return [json.loads(l) for l in fh if l.strip()]

    def _avg(records: list[dict], key: str) -> float:
        vals = [r[key] for r in records if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    otter_recs    = _load_results(RESULTS_DIR / f"otter_{subset}.jsonl")
    baseline_recs = _load_results(RESULTS_DIR / f"baseline_{subset}.jsonl")

    summary = {}
    for label, recs in [("otter", otter_recs), ("baseline", baseline_recs)]:
        summary[label] = {
            "metric":            round(_avg(recs, metric_key), 4),
            "compression_ratio": round(_avg(recs, "compression_ratio") * 100, 1),
            "token_reduction":   round(_avg(recs, "token_reduction_pct"), 1),
            "latency_s":         round(_avg(recs, "latency_s"), 2),
            "n":                 len(recs),
        }

    o, b = summary["otter"], summary["baseline"]
    rows = [
        (metric_label,           f"{o['metric']:.3f}",           f"{b['metric']:.3f}"),
        ("Mean compression",     f"{o['compression_ratio']:.1f}%", f"{b['compression_ratio']:.1f}%"),
        ("Mean token reduction", f"{o['token_reduction']:.1f}%",  f"{b['token_reduction']:.1f}%"),
        ("Mean latency (s)",     f"{o['latency_s']:.2f}",         f"{b['latency_s']:.2f}"),
        ("Examples scored",      str(o['n']),                      str(b['n'])),
    ]

    col0 = max(len(r[0]) for r in rows) + 2
    col1 = max(len(r[1]) for r in rows) + 2
    col2 = max(len(r[2]) for r in rows) + 2

    def _row(a, b_, c, sep="│"):
        return f"{sep} {a:<{col0}}{sep} {b_:^{col1}}{sep} {c:^{col2}}{sep}"

    div_top = f"┌{'─'*(col0+2)}┬{'─'*(col1+2)}┬{'─'*(col2+2)}┐"
    div_mid = f"├{'─'*(col0+2)}┼{'─'*(col1+2)}┼{'─'*(col2+2)}┤"
    div_bot = f"└{'─'*(col0+2)}┴{'─'*(col1+2)}┴{'─'*(col2+2)}┘"

    print(f"\nSummary — {subset}  ({metric_label})")
    print(div_top)
    print(_row("Metric", "OTTER", "Baseline"))
    print(div_mid)
    for r in rows:
        print(_row(*r))
    print(div_bot)

    # Save CSV
    csv_path = RESULTS_DIR / f"summary_{subset}.csv"
    with open(csv_path, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["metric", "otter", "baseline"])
        for r in rows:
            writer.writerow(r)
    print(f"Summary CSV saved to {csv_path}\n")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="OTTER benchmark harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--subset",     default="qasper",
                   help="LongBench subset name")
    p.add_argument("--mode",       default="otter", choices=["otter", "baseline"],
                   help="Compression mode")
    p.add_argument("--max",        type=int, default=None, dest="max_examples",
                   help="Maximum number of examples to run (None = all)")
    p.add_argument("--summarise",  action="store_true",
                   help="Print comparison table instead of running benchmark")
    p.add_argument("--no-resume",  action="store_true",
                   help="Ignore existing results and rerun from scratch")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.summarise:
        summarise(args.subset)
    else:
        run_benchmark(
            subset=args.subset,
            mode=args.mode,
            max_examples=args.max_examples,
            resume=not args.no_resume,
        )
