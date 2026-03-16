"""
debate_orchestrator.py — Core 4-phase debate pipeline.

Phases:
  1. Initialization  — both debaters independently pick an initial answer
  2. Multi-Round     — structured debate with early stopping
  3. Judgment        — judge evaluates full transcript
  4. Evaluation      — compare verdict to ground truth, log everything
"""

import json
from datetime import datetime
from pathlib import Path

from src.agents.debater_a import DebaterA
from src.agents.debater_b import DebaterB
from src.agents.judge import Judge
from src.utils import load_config


def _zero_usage() -> dict:
    return {"total_llm_calls": 0, "total_tokens": 0,
            "total_prompt_tokens": 0, "total_completion_tokens": 0,
            "total_latency_seconds": 0.0}


def _add_usage(acc: dict, result: dict):
    """Accumulate token/latency stats from a single call_llm result dict."""
    acc["total_llm_calls"]         += 1
    acc["total_tokens"]            += result.get("total_tokens") or 0
    acc["total_prompt_tokens"]     += result.get("prompt_tokens") or 0
    acc["total_completion_tokens"] += result.get("completion_tokens") or 0
    acc["total_latency_seconds"]   += result.get("latency_seconds") or 0.0


class DebateOrchestrator:
    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self.debater_a = DebaterA(self.config)
        self.debater_b = DebaterB(self.config)
        self.judge = Judge(self.config)

        self.log_dir = Path(self.config["logging"]["log_dir"])
        self.log_dir.mkdir(exist_ok=True)

        debate_cfg = self.config["debate"]
        self.max_rounds = debate_cfg["max_rounds"]
        self.min_rounds = debate_cfg["min_rounds"]
        self.early_stop_consecutive = debate_cfg["early_stop_consecutive"]

    # ─── Public API ───────────────────────────────────────────────────────────

    def run(self, question_dict: dict) -> dict:
        """
        Run the full 4-phase debate pipeline for a single question.

        Args:
            question_dict: {id, question, choices, answer}

        Returns:
            Full result dict (also saved as JSON to logs/)
        """
        qid          = question_dict["id"]
        question     = question_dict["question"]
        choices      = question_dict["choices"]
        ground_truth = question_dict["answer"]

        print(f"\n{'='*60}")
        print(f"Question ID: {qid}")
        print(f"Question: {question[:80]}...")
        print(f"Ground Truth: {ground_truth}")
        print(f"{'='*60}")

        usage = _zero_usage()

        result = {
            "id":           qid,
            "question":     question,
            "choices":      choices,
            "ground_truth": ground_truth,
            "timestamp":    datetime.utcnow().isoformat(),
        }

        # ── Phase 1: Initialization ───────────────────────────────────────────
        print("\n[Phase 1] Getting initial positions...")

        llm_a, ans_a = self.debater_a.get_initial_position(question, choices)
        llm_b, ans_b = self.debater_b.get_initial_position(question, choices)
        _add_usage(usage, llm_a)
        _add_usage(usage, llm_b)

        result["initial_position_a"] = {
            "response": llm_a["text"], "answer": ans_a,
            "tokens": llm_a["total_tokens"], "latency": llm_a["latency_seconds"],
        }
        result["initial_position_b"] = {
            "response": llm_b["text"], "answer": ans_b,
            "tokens": llm_b["total_tokens"], "latency": llm_b["latency_seconds"],
        }

        print(f"  Debater A initial answer: {ans_a}")
        print(f"  Debater B initial answer: {ans_b}")

        # Check for Phase 1 consensus
        if ans_a and ans_b and ans_a == ans_b:
            print(f"  [Consensus reached in Phase 1] Both agree on: {ans_a}")
            result["consensus"]        = True
            result["consensus_answer"] = ans_a
            result["rounds"]           = []
            final_ans_a = ans_a
            final_ans_b = ans_b
        else:
            result["consensus"]        = False
            result["consensus_answer"] = None
            result["rounds"]           = self._run_debate(
                question, choices, ans_a, usage
            )
            final_ans_a = ans_a  # Debater A is always locked to initial position
            final_ans_b = next(
                (r["answer_b"] for r in reversed(result["rounds"]) if r.get("answer_b")),
                ans_b,
            )

        # ── Phase 3: Judgment (always runs per assignment spec) ───────────────
        print("\n[Phase 3] Judge evaluating debate...")

        judge_result = self.judge.evaluate(
            question=question,
            choices=choices,
            initial_a=result["initial_position_a"]["response"],
            initial_b=result["initial_position_b"]["response"],
            rounds=result["rounds"],
            answer_a=final_ans_a,
            answer_b=final_ans_b,
            consensus=result["consensus"],
        )
        _add_usage(usage, {
            "total_tokens":      judge_result["total_tokens"],
            "prompt_tokens":     judge_result["prompt_tokens"],
            "completion_tokens": judge_result["completion_tokens"],
            "latency_seconds":   judge_result["latency_seconds"],
        })

        result["judge"]   = judge_result
        result["verdict"] = judge_result["final_answer"]

        print(f"  Judge verdict: {result['verdict']} (confidence: {judge_result['confidence']})")

        # ── Phase 4: Evaluation ───────────────────────────────────────────────
        verdict = result["verdict"]
        result["correct"] = (verdict == ground_truth) if verdict else False
        print(f"  Verdict: {verdict} | Correct: {result['correct']} (ground truth: {ground_truth})")

        # Attach usage summary
        result["usage"] = usage
        print(f"  LLM calls: {usage['total_llm_calls']} | "
              f"Tokens: {usage['total_tokens']} | "
              f"Latency: {usage['total_latency_seconds']:.1f}s")

        self._save_log(result)
        return result

    # ─── Phase 2 ──────────────────────────────────────────────────────────────

    def _run_debate(
        self,
        question: str,
        choices: dict,
        position_a: str,
        usage: dict,
    ) -> list[dict]:
        """
        Run the multi-round debate loop (Phase 2).

        Returns:
            List of round dicts
        """
        rounds = []
        consecutive_agreements = 0
        position = position_a or list(choices.keys())[0]

        print(f"\n[Phase 2] Starting debate (Debater A defends: {position})")

        for round_num in range(1, self.max_rounds + 1):
            print(f"\n  -- Round {round_num} --")

            # Debater A argues
            llm_a, ans_a = self.debater_a.argue(
                question=question,
                choices=choices,
                position=position,
                rounds=rounds,
            )
            _add_usage(usage, llm_a)
            print(f"    Debater A current answer: {ans_a}")

            # Build partial round so Debater B can see A's argument
            partial_rounds = rounds + [{
                "round":     round_num,
                "debater_a": llm_a["text"],
                "debater_b": "(pending)",
            }]

            # Debater B responds
            llm_b, ans_b = self.debater_b.argue(
                question=question,
                choices=choices,
                rounds=partial_rounds,
            )
            _add_usage(usage, llm_b)
            print(f"    Debater B current answer: {ans_b}")

            round_record = {
                "round":           round_num,
                "debater_a":       llm_a["text"],
                "answer_a":        ans_a,
                "tokens_a":        llm_a["total_tokens"],
                "latency_a":       llm_a["latency_seconds"],
                "debater_b":       llm_b["text"],
                "answer_b":        ans_b,
                "tokens_b":        llm_b["total_tokens"],
                "latency_b":       llm_b["latency_seconds"],
            }
            rounds.append(round_record)

            # Early stopping check
            if ans_a and ans_b and ans_a == ans_b:
                consecutive_agreements += 1
                print(f"    Agreement ({consecutive_agreements}/{self.early_stop_consecutive}): both say {ans_a}")
                if (round_num >= self.min_rounds and
                        consecutive_agreements >= self.early_stop_consecutive):
                    print(f"    Early stopping after round {round_num}.")
                    break
            else:
                consecutive_agreements = 0

        return rounds

    # ─── Logging ──────────────────────────────────────────────────────────────

    def _save_log(self, result: dict):
        """Save result dict as a JSON file in logs/."""
        filename = f"{result['id']}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        path = self.log_dir / filename
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Log saved: {path}")
