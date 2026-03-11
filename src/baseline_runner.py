"""
baseline_runner.py — Direct QA and Self-Consistency baselines.

Both baselines use the 70B model (config: models.baseline).
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from src.utils import (
    load_config, make_client, load_prompt, call_llm,
    format_choices, extract_final_answer,
)


class BaselineRunner:
    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.client, self.model = make_client("baseline", self.config)
        self.n_samples = self.config["self_consistency"]["num_samples"]

        self.log_dir = Path(self.config["logging"]["log_dir"])
        self.log_dir.mkdir(exist_ok=True)

    # ─── Direct QA ────────────────────────────────────────────────────────────

    def run_direct_qa(self, question_dict: dict) -> dict:
        """
        Baseline 1: single CoT call, no debate.

        Returns result dict with answer, correctness, and usage stats.
        """
        question     = question_dict["question"]
        choices      = question_dict["choices"]
        ground_truth = question_dict["answer"]

        prompt = load_prompt(
            "direct_qa",
            question=question,
            choices=format_choices(choices),
        )

        llm = call_llm(self.client, self.model, prompt, self.config)
        answer = extract_final_answer(llm["text"])

        result = {
            "id":            question_dict["id"],
            "question":      question,
            "choices":       choices,
            "ground_truth":  ground_truth,
            "timestamp":     datetime.utcnow().isoformat(),
            "method":        "direct_qa",
            "model":         self.model,
            "response":      llm["text"],
            "answer":        answer,
            "correct":       (answer == ground_truth) if answer else False,
            "usage": {
                "total_llm_calls":         1,
                "total_tokens":            llm["total_tokens"],
                "total_prompt_tokens":     llm["prompt_tokens"],
                "total_completion_tokens": llm["completion_tokens"],
                "total_latency_seconds":   llm["latency_seconds"],
            },
        }

        self._save_log(result, prefix="direct_qa")
        return result

    # ─── Self-Consistency ─────────────────────────────────────────────────────

    def run_self_consistency(self, question_dict: dict) -> dict:
        """
        Baseline 2: N independent calls, majority vote.

        N = config.self_consistency.num_samples
        """
        question     = question_dict["question"]
        choices      = question_dict["choices"]
        ground_truth = question_dict["answer"]

        prompt = load_prompt(
            "direct_qa",
            question=question,
            choices=format_choices(choices),
        )

        samples = []
        acc_usage = {
            "total_llm_calls": 0, "total_tokens": 0,
            "total_prompt_tokens": 0, "total_completion_tokens": 0,
            "total_latency_seconds": 0.0,
        }

        for i in range(self.n_samples):
            llm = call_llm(self.client, self.model, prompt, self.config)
            answer = extract_final_answer(llm["text"])
            samples.append({
                "sample_index":      i,
                "response":          llm["text"],
                "answer":            answer,
                "tokens":            llm["total_tokens"],
                "latency_seconds":   llm["latency_seconds"],
            })
            acc_usage["total_llm_calls"]         += 1
            acc_usage["total_tokens"]            += llm["total_tokens"] or 0
            acc_usage["total_prompt_tokens"]     += llm["prompt_tokens"] or 0
            acc_usage["total_completion_tokens"] += llm["completion_tokens"] or 0
            acc_usage["total_latency_seconds"]   += llm["latency_seconds"] or 0.0

        # Majority vote (exclude None answers)
        valid_answers = [s["answer"] for s in samples if s["answer"]]
        if valid_answers:
            majority_answer = Counter(valid_answers).most_common(1)[0][0]
            vote_counts = dict(Counter(valid_answers))
        else:
            majority_answer = None
            vote_counts = {}

        result = {
            "id":             question_dict["id"],
            "question":       question,
            "choices":        choices,
            "ground_truth":   ground_truth,
            "timestamp":      datetime.utcnow().isoformat(),
            "method":         "self_consistency",
            "model":          self.model,
            "n_samples":      self.n_samples,
            "samples":        samples,
            "vote_counts":    vote_counts,
            "answer":         majority_answer,
            "correct":        (majority_answer == ground_truth) if majority_answer else False,
            "usage":          acc_usage,
        }

        self._save_log(result, prefix="self_consistency")
        return result

    # ─── Logging ──────────────────────────────────────────────────────────────

    def _save_log(self, result: dict, prefix: str):
        filename = f"{prefix}_{result['id']}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        path = self.log_dir / filename
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
