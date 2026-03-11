"""
run_debate.py — Run the full debate pipeline on ARC-Challenge questions.

Usage:
    python experiments/run_debate.py
    python experiments/run_debate.py --limit 5       # test on 5 questions
    python experiments/run_debate.py --limit 5 --offset 10
"""

import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config
from src.debate_orchestrator import DebateOrchestrator


def main():
    parser = argparse.ArgumentParser(description="Run LLM debate pipeline.")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Number of questions to run (default: all 200)")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip first N questions")
    args = parser.parse_args()

    config   = load_config()
    data_dir = Path(config["dataset"]["data_dir"])
    data_path = data_dir / "arc_challenge_200.json"

    print(f"Loading dataset from {data_path}...")
    with open(data_path) as f:
        questions = json.load(f)

    # Apply offset and limit
    questions = questions[args.offset:]
    if args.limit:
        questions = questions[:args.limit]

    print(f"Running debate on {len(questions)} questions...\n")

    orchestrator = DebateOrchestrator(config)
    results = []
    correct = 0
    parse_failures = 0

    for q in tqdm(questions, desc="Debate"):
        try:
            result = orchestrator.run(q)
            results.append(result)
            if result["correct"]:
                correct += 1
            if result["verdict"] is None:
                parse_failures += 1
        except Exception as e:
            print(f"\n[ERROR] Question {q['id']}: {e}")
            results.append({
                "id": q["id"], "error": str(e), "correct": False
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    total      = len(results)
    answered   = total - parse_failures
    accuracy   = correct / total if total > 0 else 0

    total_calls   = sum(r.get("usage", {}).get("total_llm_calls", 0) for r in results)
    total_tokens  = sum(r.get("usage", {}).get("total_tokens", 0) for r in results)
    total_latency = sum(r.get("usage", {}).get("total_latency_seconds", 0) for r in results)
    avg_rounds    = sum(len(r.get("rounds", [])) for r in results) / total if total > 0 else 0

    summary = {
        "method":               "debate",
        "total_questions":      total,
        "correct":              correct,
        "accuracy":             round(accuracy, 4),
        "parse_failures":       parse_failures,
        "avg_rounds":           round(avg_rounds, 2),
        "total_llm_calls":      total_calls,
        "total_tokens":         total_tokens,
        "avg_tokens_per_q":     round(total_tokens / total, 1) if total > 0 else 0,
        "total_latency_seconds": round(total_latency, 1),
        "avg_latency_per_q":    round(total_latency / total, 1) if total > 0 else 0,
    }

    print("\n" + "="*60)
    print("DEBATE RESULTS SUMMARY")
    print("="*60)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Save summary
    results_dir = Path(config["logging"]["results_dir"])
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "debate_summary.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
