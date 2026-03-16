"""
analyze_panel_results.py — Compare multi-judge panel vs. single-judge debate.

Expected input files (in results/):
  - debate_summary.json          (from run_debate.py       — single judge)
  - debate_panel_summary.json    (from run_debate_panel.py — panel)

Outputs:
  - Comparison table printed to stdout
  - results/figures/panel_accuracy_comparison.png
  - results/figures/panel_disagreement_analysis.png
  - results/figures/panel_deliberation_effect.png

Usage:
    python experiments/analyze_panel_results.py
"""

import sys
import json
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
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
    if len(correct_a) != len(correct_b):
        return None
    b = sum(1 for a, b_ in zip(correct_a, correct_b) if a and not b_)
    c = sum(1 for a, b_ in zip(correct_a, correct_b) if not a and b_)
    if b + c == 0:
        return None
    # continuity-corrected McNemar
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    from scipy.stats import chi2 as chi2_dist
    return float(chi2_dist.sf(chi2, df=1))


def align_by_id(
    results_a: list[dict], results_b: list[dict]
) -> tuple[list[bool], list[bool]]:
    ids_b = {r["id"]: r for r in results_b}
    aligned_a, aligned_b = [], []
    for r in results_a:
        if r["id"] in ids_b:
            aligned_a.append(bool(r.get("correct", False)))
            aligned_b.append(bool(ids_b[r["id"]].get("correct", False)))
    return aligned_a, aligned_b


def fmt_p(p: float | None) -> str:
    if p is None:
        return "N/A"
    if p < 0.001:
        return "< 0.001 ***"
    if p < 0.01:
        return f"{p:.3f} **"
    if p < 0.05:
        return f"{p:.3f} *"
    return f"{p:.3f}"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    config      = load_config()
    results_dir = Path(config["logging"]["results_dir"])
    fig_dir     = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    single_data = load_json(results_dir / "debate_summary.json")
    panel_data  = load_json(results_dir / "debate_panel_summary.json")

    if not single_data or not panel_data:
        print("ERROR: Both debate_summary.json and debate_panel_summary.json are required.")
        return

    single_results = single_data["results"]
    panel_results  = panel_data["results"]
    single_summary = single_data["summary"]
    panel_summary  = panel_data["summary"]

    # ── 1. Accuracy comparison ─────────────────────────────────────────────────
    print("\n" + "="*65)
    print("PANEL vs. SINGLE-JUDGE ACCURACY COMPARISON")
    print("="*65)

    single_correct = [r.get("correct", False) for r in single_results]
    panel_correct  = [r.get("correct", False) for r in panel_results]

    aligned_single, aligned_panel = align_by_id(single_results, panel_results)
    p_val = mcnemar_p(aligned_single, aligned_panel)

    rows = [
        ("Method",       "Single Judge",                       "Panel (3 judges)"),
        ("Accuracy",     f"{single_summary['accuracy']:.3f}",  f"{panel_summary['accuracy']:.3f}"),
        ("Correct",      str(single_summary['correct']),       str(panel_summary['correct'])),
        ("Parse Fail",   str(single_summary['parse_failures']),str(panel_summary['parse_failures'])),
        ("Avg Tokens/Q", f"{single_summary['avg_tokens_per_q']:,.0f}", f"{panel_summary['avg_tokens_per_q']:,.0f}"),
        ("Avg Latency/Q",f"{single_summary['avg_latency_per_q']:.1f}s", f"{panel_summary['avg_latency_per_q']:.1f}s"),
        ("McNemar p",    "ref",                                fmt_p(p_val)),
    ]

    col_w = [20, 20, 20]
    header = f"{'':20} {'Single Judge':20} {'Panel (3 judges)':20}"
    print(header)
    print("-" * 62)
    for label, single_val, panel_val in rows[1:]:
        print(f"{label:20} {single_val:20} {panel_val:20}")

    # ── 2. Panel disagreement analysis ────────────────────────────────────────
    print("\n" + "="*65)
    print("PANEL DISAGREEMENT ANALYSIS")
    print("="*65)

    n_total         = len(panel_results)
    n_disagreed_r1  = panel_summary.get("r1_disagreements", 0)
    n_deliberated   = panel_summary.get("deliberated", 0)

    # Accuracy split: unanimous R1 vs. disagreed R1
    unanimous_r1_correct  = [
        r.get("correct", False) for r in panel_results
        if r.get("judge", {}).get("panel_unanimous_r1", True)
    ]
    disagreed_r1_correct  = [
        r.get("correct", False) for r in panel_results
        if not r.get("judge", {}).get("panel_unanimous_r1", True)
    ]

    n_unani  = len(unanimous_r1_correct)
    n_disagr = len(disagreed_r1_correct)

    acc_unani  = sum(unanimous_r1_correct)  / n_unani  if n_unani  else 0
    acc_disagr = sum(disagreed_r1_correct)  / n_disagr if n_disagr else 0

    print(f"  R1 unanimous  : {n_unani:3d} questions — accuracy {acc_unani:.3f}")
    print(f"  R1 disagreed  : {n_disagr:3d} questions — accuracy {acc_disagr:.3f}")
    print(f"  Deliberated   : {n_deliberated:3d} questions")

    # ── 3. Deliberation effect ────────────────────────────────────────────────
    deliberated_results = [
        r for r in panel_results
        if r.get("judge", {}).get("panel_deliberated")
    ]

    if deliberated_results:
        print("\n" + "="*65)
        print("DELIBERATION EFFECT (cases where R2 was run)")
        print("="*65)

        # For each deliberated case: what was R1 majority, R2 majority, final correct?
        changed = 0
        wrong_to_right = 0
        right_to_wrong = 0

        for r in deliberated_results:
            judge_data = r.get("judge", {})
            r1 = judge_data.get("panel_r1_results", [])
            r2 = judge_data.get("panel_r2_results", [])
            if not r1 or not r2:
                continue

            r1_verdicts = [x["final_answer"] for x in r1 if x.get("final_answer")]
            r2_verdicts = [x["final_answer"] for x in r2 if x.get("final_answer")]

            r1_majority = Counter(r1_verdicts).most_common(1)[0][0] if r1_verdicts else None
            r2_majority = Counter(r2_verdicts).most_common(1)[0][0] if r2_verdicts else None

            gt = r.get("ground_truth")
            if r1_majority != r2_majority:
                changed += 1
                r1_correct = (r1_majority == gt)
                r2_correct = (r2_majority == gt)
                if not r1_correct and r2_correct:
                    wrong_to_right += 1
                elif r1_correct and not r2_correct:
                    right_to_wrong += 1

        n_delib = len(deliberated_results)
        print(f"  Deliberated cases     : {n_delib}")
        print(f"  Verdict changed in R2 : {changed} ({changed/n_delib:.1%})")
        print(f"  Wrong → Correct       : {wrong_to_right}")
        print(f"  Correct → Wrong       : {right_to_wrong}")

    # ── 4. Disagreement vs. question difficulty ────────────────────────────────
    print("\n" + "="*65)
    print("DISAGREEMENT RATE BY DEBATE LENGTH (difficulty proxy)")
    print("="*65)

    # Group by number of debate rounds (0 = consensus, 1-5 = full debate)
    by_rounds: dict[int, list[bool]] = {}
    for r in panel_results:
        n_rounds = len(r.get("rounds", []))
        disagreed = not r.get("judge", {}).get("panel_unanimous_r1", True)
        by_rounds.setdefault(n_rounds, []).append(disagreed)

    for n_rounds in sorted(by_rounds):
        vals = by_rounds[n_rounds]
        rate = sum(vals) / len(vals) if vals else 0
        label = f"{n_rounds} rounds" if n_rounds > 0 else "0 rounds (consensus)"
        print(f"  {label:25}: {rate:.1%} disagreement  (n={len(vals)})")

    # ── 5. Figures ─────────────────────────────────────────────────────────────

    # Figure 1: Accuracy comparison bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    methods = ["Single Judge", "Panel (3 judges)"]
    accs    = [single_summary["accuracy"], panel_summary["accuracy"]]
    bars    = ax.bar(methods, accs, color=["#4C72B0", "#DD8452"], width=0.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Single Judge vs. Panel Accuracy")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.01,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=11)
    plt.tight_layout()
    fig.savefig(fig_dir / "panel_accuracy_comparison.png", dpi=150)
    plt.close(fig)
    print(f"\nFigure saved: {fig_dir / 'panel_accuracy_comparison.png'}")

    # Figure 2: Disagreement accuracy breakdown
    if n_unani + n_disagr > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = [f"R1 Unanimous\n(n={n_unani})", f"R1 Disagreed\n(n={n_disagr})"]
        accs2  = [acc_unani, acc_disagr]
        colors = ["#55A868", "#C44E52"]
        bars   = ax.bar(labels, accs2, color=colors, width=0.5)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Accuracy")
        ax.set_title("Panel Accuracy: Unanimous vs. Disagreed R1")
        for bar, acc in zip(bars, accs2):
            ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.01,
                    f"{acc:.3f}", ha="center", va="bottom", fontsize=11)
        plt.tight_layout()
        fig.savefig(fig_dir / "panel_disagreement_analysis.png", dpi=150)
        plt.close(fig)
        print(f"Figure saved: {fig_dir / 'panel_disagreement_analysis.png'}")

    # Figure 3: Disagreement rate by rounds
    round_labels = sorted(by_rounds.keys())
    disagr_rates = [sum(by_rounds[k]) / len(by_rounds[k]) for k in round_labels]
    x_labels     = [str(k) for k in round_labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x_labels, disagr_rates, color="#8172B2")
    ax.set_xlabel("Number of Debate Rounds (0 = consensus)")
    ax.set_ylabel("R1 Disagreement Rate")
    ax.set_title("Panel Disagreement Rate by Question Difficulty")
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    fig.savefig(fig_dir / "panel_disagreement_by_rounds.png", dpi=150)
    plt.close(fig)
    print(f"Figure saved: {fig_dir / 'panel_disagreement_by_rounds.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
