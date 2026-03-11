"""
run_baselines.py — Run Direct QA and Self-Consistency baselines.

Usage:
    python experiments/run_baselines.py                      # both baselines
    python experiments/run_baselines.py --method direct_qa   # only Direct QA
    python experiments/run_baselines.py --method self_consistency
    python experiments/run_baselines.py --limit 5            # test on 5 questions
"""

import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config
from src.baseline_runner import BaselineRunner


def run_method(runner: BaselineRunner, questions: list, method: str) -> dict:
    """Run one baseline method over all questions and return summary + results."""
    results = []
    correct = 0
    parse_failures = 0

    run_fn = (runner.run_direct_qa if method == "direct_qa"
              else runner.run_self_consistency)

    for q in tqdm(questions, desc=method):
        try:
            result = run_fn(q)
            results.append(result)
            if result["correct"]:
                correct += 1
            if result["answer"] is None:
                parse_failures += 1
        except Exception as e:
            print(f"\n[ERROR] Question {q['id']}: {e}")
            results.append({"id": q["id"], "error": str(e), "correct": False})

    total        = len(results)
    accuracy     = correct / total if total > 0 else 0
    total_tokens  = sum(r.get("usage", {}).get("total_tokens", 0) for r in results)
    total_latency = sum(r.get("usage", {}).get("total_latency_seconds", 0) for r in results)
    total_calls   = sum(r.get("usage", {}).get("total_llm_calls", 0) for r in results)

    summary = {
        "method":                method,
        "total_questions":       total,
        "correct":               correct,
        "accuracy":              round(accuracy, 4),
        "parse_failures":        parse_failures,
        "total_llm_calls":       total_calls,
        "total_tokens":          total_tokens,
        "avg_tokens_per_q":      round(total_tokens / total, 1) if total > 0 else 0,
        "total_latency_seconds": round(total_latency, 1),
        "avg_latency_per_q":     round(total_latency / total, 1) if total > 0 else 0,
    }

    print(f"\n{'='*60}")
    print(f"{method.upper()} RESULTS SUMMARY")
    print("="*60)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return {"summary": summary, "results": results}


def main():
    parser = argparse.ArgumentParser(description="Run baseline experiments.")
    parser.add_argument("--method", choices=["direct_qa", "self_consistency", "both"],
                        default="both", help="Which baseline to run (default: both)")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Number of questions to run (default: all 200)")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip first N questions")
    args = parser.parse_args()

    config    = load_config()
    data_path = Path(config["dataset"]["data_dir"]) / "arc_challenge_200.json"

    print(f"Loading dataset from {data_path}...")
    with open(data_path) as f:
        questions = json.load(f)

    questions = questions[args.offset:]
    if args.limit:
        questions = questions[:args.limit]

    print(f"Running baselines on {len(questions)} questions...\n")

    runner      = BaselineRunner(config)
    results_dir = Path(config["logging"]["results_dir"])
    results_dir.mkdir(exist_ok=True)

    methods = (["direct_qa", "self_consistency"] if args.method == "both"
               else [args.method])

    for method in methods:
        output = run_method(runner, questions, method)
        out_path = results_dir / f"baseline_{method}.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
