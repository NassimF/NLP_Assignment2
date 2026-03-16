"""
analyze_results.py — Load experiment summaries and produce comparison table + figures.

Expected input files (all in results/):
  - debate_summary.json       (from run_debate.py)
  - baseline_direct_qa.json   (from run_baselines.py)
  - baseline_self_consistency.json

Outputs:
  - results/comparison_table.csv
  - results/figures/accuracy_by_method.png
  - results/figures/tokens_by_method.png
  - results/figures/latency_by_method.png
  - results/figures/confidence_vs_accuracy.png
  - Comparison table printed as markdown to stdout

Usage:
    python experiments/analyze_results.py
    python experiments/analyze_results.py --results-dir results/
"""

import sys
import json
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_config


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict | None:
    if not path.exists():
        print(f"  [SKIP] {path} not found.")
        return None
    with open(path) as f:
        return json.load(f)


def mcnemar_p(correct_a: list[bool], correct_b: list[bool]) -> float | None:
    """
    McNemar's test comparing two paired binary sequences.
    Returns the p-value or None if the sequences are not the same length
    or the table is degenerate.
    """
    if len(correct_a) != len(correct_b):
        return None
    # Build 2x2 contingency on aligned questions
    b = sum(1 for a, b_ in zip(correct_a, correct_b) if a and not b_)
    c = sum(1 for a, b_ in zip(correct_a, correct_b) if not a and b_)
    if b + c == 0:
        return None  # identical predictions — no test possible
    # Use chi2_contingency on the discordant cells
    table = [[0, b], [c, 0]]
    _, p, _, _ = chi2_contingency(table, correction=True)
    return round(float(p), 4)


def align_by_id(results_a: list[dict], results_b: list[dict]) -> tuple[list[bool], list[bool]]:
    """Return two aligned correctness lists for questions present in both sets."""
    idx_b = {r["id"]: r for r in results_b}
    correct_a, correct_b = [], []
    for r in results_a:
        if r["id"] in idx_b:
            correct_a.append(bool(r.get("correct", False)))
            correct_b.append(bool(idx_b[r["id"]].get("correct", False)))
    return correct_a, correct_b


# ─── Statistics extraction ────────────────────────────────────────────────────

def extract_stats(data: dict, method_name: str) -> dict:
    """Pull a flat stats dict from a run's saved JSON."""
    summary = data.get("summary", {})
    results = data.get("results", [])

    total    = summary.get("total_questions", len(results))
    correct  = summary.get("correct", sum(r.get("correct", False) for r in results))
    accuracy = summary.get("accuracy", correct / total if total else 0)
    failures = summary.get("parse_failures", 0)

    avg_calls   = summary.get("total_llm_calls", 0) / total if total else 0
    avg_tokens  = summary.get("avg_tokens_per_q", 0)
    avg_latency = summary.get("avg_latency_per_q", 0)

    stats = {
        "method":          method_name,
        "n":               total,
        "correct":         correct,
        "accuracy":        round(accuracy, 4),
        "parse_fail_rate": round(failures / total, 4) if total else 0,
        "avg_llm_calls":   round(avg_calls, 2),
        "avg_tokens":      round(avg_tokens, 1),
        "avg_latency_s":   round(avg_latency, 2),
    }

    # Debate-only extras
    if method_name == "debate":
        stats["avg_rounds"] = summary.get("avg_rounds", 0)

        # Consensus rate: fraction of questions where Phase 1 agreed
        consensus_count = sum(1 for r in results if r.get("consensus", False))
        stats["consensus_rate"] = round(consensus_count / total, 4) if total else 0

        # Early-stop rate: questions that stopped before max_rounds
        # (rounds list shorter than max_rounds AND at least min_rounds were run)
        config = load_config()
        max_r = config["debate"]["max_rounds"]
        early_stop = sum(
            1 for r in results
            if len(r.get("rounds", [])) < max_r and not r.get("consensus", False)
        )
        stats["early_stop_rate"] = round(early_stop / total, 4) if total else 0
    else:
        stats["avg_rounds"]      = "N/A"
        stats["consensus_rate"]  = "N/A"
        stats["early_stop_rate"] = "N/A"

    return stats


# ─── Figures ──────────────────────────────────────────────────────────────────

def bar_chart(labels: list, values: list, ylabel: str, title: str, out_path: Path, color: str = "steelblue"):
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=color, edgecolor="black", width=0.5)
    ax.bar_label(bars, fmt="%.3g", padding=3, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(values) * 1.2 + 0.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def confidence_vs_accuracy_plot(debate_results: list[dict], out_path: Path):
    """Scatter: judge confidence vs. correctness (jittered) for debate results."""
    confidences, corrects = [], []
    for r in debate_results:
        conf = r.get("judge", {}).get("confidence")
        if conf is not None:
            confidences.append(float(conf))
            corrects.append(1 if r.get("correct", False) else 0)

    if not confidences:
        print("  [SKIP] No confidence data found for confidence_vs_accuracy plot.")
        return

    jitter = np.random.default_rng(42).uniform(-0.03, 0.03, len(corrects))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(confidences, np.array(corrects) + jitter, alpha=0.4, s=20, color="coral")

    # Bin accuracy by confidence decile
    bins = np.linspace(0, 1, 6)
    bin_idx = np.digitize(confidences, bins) - 1
    bin_acc = []
    bin_centers = []
    for i in range(len(bins) - 1):
        mask = bin_idx == i
        if mask.sum() > 0:
            bin_acc.append(np.array(corrects)[mask].mean())
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
    if bin_centers:
        ax.plot(bin_centers, bin_acc, "b-o", linewidth=2, label="Bin accuracy")
        ax.legend()

    ax.set_xlabel("Judge Confidence")
    ax.set_ylabel("Correct (jittered)")
    ax.set_title("Judge Confidence vs. Accuracy (Debate)")
    ax.set_ylim(-0.2, 1.2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Wrong", "Correct"])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Markdown table ───────────────────────────────────────────────────────────

COLUMNS = [
    ("method",          "Method"),
    ("n",               "N"),
    ("accuracy",        "Accuracy"),
    ("parse_fail_rate", "Parse Fail %"),
    ("avg_llm_calls",   "Avg LLM Calls"),
    ("avg_tokens",      "Avg Tokens"),
    ("avg_latency_s",   "Avg Latency (s)"),
    ("avg_rounds",      "Avg Rounds"),
    ("consensus_rate",  "Consensus Rate"),
    ("early_stop_rate", "Early Stop Rate"),
    ("p_vs_debate",     "p (vs debate)"),
]


def format_val(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v) if v is not None else "—"


def print_markdown_table(rows: list[dict]):
    headers = [col[1] for col in COLUMNS]
    col_keys = [col[0] for col in COLUMNS]

    widths = [max(len(h), max(len(format_val(row.get(k, "—"))) for row in rows))
              for h, k in zip(headers, col_keys)]

    def fmt_row(vals):
        return "| " + " | ".join(v.ljust(w) for v, w in zip(vals, widths)) + " |"

    print(fmt_row(headers))
    print("| " + " | ".join("-" * w for w in widths) + " |")
    for row in rows:
        print(fmt_row([format_val(row.get(k, "—")) for k in col_keys]))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze and compare experiment results.")
    parser.add_argument("--results-dir", default=None,
                        help="Path to results/ directory (default: from config.yaml)")
    args = parser.parse_args()

    config = load_config()
    results_dir = Path(args.results_dir or config["logging"]["results_dir"])
    fig_dir = results_dir / "figures"

    print(f"Loading results from {results_dir}/\n")

    debate_data = load_json(results_dir / "debate_summary.json")
    dqa_data    = load_json(results_dir / "baseline_direct_qa.json")
    sc_data     = load_json(results_dir / "baseline_self_consistency.json")

    available = {k: v for k, v in {
        "debate":           debate_data,
        "direct_qa":        dqa_data,
        "self_consistency": sc_data,
    }.items() if v is not None}

    if not available:
        print("No result files found. Run the experiment scripts first.")
        return

    # ── Compute per-method stats ───────────────────────────────────────────────
    stats_rows = {name: extract_stats(data, name) for name, data in available.items()}

    # ── McNemar tests ─────────────────────────────────────────────────────────
    for method in ["direct_qa", "self_consistency"]:
        stats_rows[method]["p_vs_debate"] = "—"

    if "debate" in available:
        debate_results = available["debate"].get("results", [])
        for method in ["direct_qa", "self_consistency"]:
            if method in available:
                other_results = available[method].get("results", [])
                ca, cb = align_by_id(debate_results, other_results)
                p = mcnemar_p(ca, cb)
                stats_rows["debate"]["p_vs_debate"] = "ref"
                stats_rows[method]["p_vs_debate"] = format_val(p) if p is not None else "—"
    else:
        for method in stats_rows:
            stats_rows[method]["p_vs_debate"] = "—"

    if "debate" in stats_rows:
        stats_rows["debate"]["p_vs_debate"] = "ref"

    # ── Print markdown table ───────────────────────────────────────────────────
    ordered = ["debate", "direct_qa", "self_consistency"]
    rows = [stats_rows[m] for m in ordered if m in stats_rows]

    print("\n## Results Comparison\n")
    print_markdown_table(rows)

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = results_dir / "comparison_table.csv"
    with open(csv_path, "w", newline="") as f:
        col_keys = [col[0] for col in COLUMNS]
        writer = csv.DictWriter(f, fieldnames=col_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in col_keys})
    print(f"\nComparison table saved to {csv_path}")

    # ── Figures ────────────────────────────────────────────────────────────────
    print("\nGenerating figures...")

    labels  = [r["method"] for r in rows]
    acc     = [r["accuracy"] for r in rows]
    tokens  = [r["avg_tokens"] for r in rows]
    latency = [r["avg_latency_s"] for r in rows]

    bar_chart(labels, acc,     "Accuracy",           "Accuracy by Method",
              fig_dir / "accuracy_by_method.png",  color="steelblue")
    bar_chart(labels, tokens,  "Avg Tokens / Q",     "Token Usage by Method",
              fig_dir / "tokens_by_method.png",    color="darkorange")
    bar_chart(labels, latency, "Avg Latency (s) / Q","Latency by Method",
              fig_dir / "latency_by_method.png",   color="mediumseagreen")

    if "debate" in available:
        confidence_vs_accuracy_plot(
            available["debate"].get("results", []),
            fig_dir / "confidence_vs_accuracy.png",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
