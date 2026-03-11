"""
debate_orchestrator.py — Core 4-phase debate pipeline.

Phases:
  1. Initialization  — both debaters independently pick an initial answer
  2. Multi-Round     — structured debate with early stopping
  3. Judgment        — judge evaluates full transcript
  4. Evaluation      — compare verdict to ground truth, log everything
"""

import json
import os
from datetime import datetime
from pathlib import Path

from src.agents.debater_a import DebaterA
from src.agents.debater_b import DebaterB
from src.agents.judge import Judge
from src.utils import load_config


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
        qid       = question_dict["id"]
        question  = question_dict["question"]
        choices   = question_dict["choices"]
        ground_truth = question_dict["answer"]

        print(f"\n{'='*60}")
        print(f"Question ID: {qid}")
        print(f"Question: {question[:80]}...")
        print(f"Ground Truth: {ground_truth}")
        print(f"{'='*60}")

        result = {
            "id":           qid,
            "question":     question,
            "choices":      choices,
            "ground_truth": ground_truth,
            "timestamp":    datetime.utcnow().isoformat(),
            "total_llm_calls": 0,
        }

        # ── Phase 1: Initialization ───────────────────────────────────────────
        print("\n[Phase 1] Getting initial positions...")

        resp_a, ans_a = self.debater_a.get_initial_position(question, choices)
        resp_b, ans_b = self.debater_b.get_initial_position(question, choices)
        result["total_llm_calls"] += 2

        result["initial_position_a"] = {"response": resp_a, "answer": ans_a}
        result["initial_position_b"] = {"response": resp_b, "answer": ans_b}

        print(f"  Debater A initial answer: {ans_a}")
        print(f"  Debater B initial answer: {ans_b}")

        # Check for Phase 1 consensus
        if ans_a and ans_b and ans_a == ans_b:
            print(f"  [Consensus reached in Phase 1] Both agree on: {ans_a}")
            result["consensus"] = True
            result["consensus_answer"] = ans_a
            result["rounds"] = []
        else:
            result["consensus"] = False
            result["consensus_answer"] = None
            result["rounds"] = self._run_debate(
                question, choices, ans_a, result
            )

        # ── Phase 3: Judgment ─────────────────────────────────────────────────
        print("\n[Phase 3] Judge evaluating debate...")

        judge_result = self.judge.evaluate(
            question=question,
            choices=choices,
            initial_a=result["initial_position_a"]["response"],
            initial_b=result["initial_position_b"]["response"],
            rounds=result["rounds"],
        )
        result["total_llm_calls"] += 1
        result["judge"] = judge_result

        verdict = judge_result["final_answer"]
        result["verdict"] = verdict

        print(f"  Judge verdict: {verdict} (confidence: {judge_result['confidence']})")

        # ── Phase 4: Evaluation ───────────────────────────────────────────────
        result["correct"] = (verdict == ground_truth) if verdict else False
        print(f"  Correct: {result['correct']} (ground truth: {ground_truth})")

        # Save log
        self._save_log(result)
        return result

    # ─── Phase 2 ──────────────────────────────────────────────────────────────

    def _run_debate(
        self,
        question: str,
        choices: dict,
        position_a: str,
        result: dict,
    ) -> list[dict]:
        """
        Run the multi-round debate loop (Phase 2).

        Args:
            question: question text
            choices: answer choices dict
            position_a: Debater A's initial answer letter to defend
            result: result dict (mutated to update total_llm_calls)

        Returns:
            List of round dicts
        """
        rounds = []
        consecutive_agreements = 0
        # Fall back to first choice letter if initial answer parse failed
        position = position_a or list(choices.keys())[0]

        print(f"\n[Phase 2] Starting debate (Debater A defends: {position})")

        for round_num in range(1, self.max_rounds + 1):
            print(f"\n  -- Round {round_num} --")

            # Debater A argues
            resp_a, ans_a = self.debater_a.argue(
                question=question,
                choices=choices,
                position=position,
                rounds=rounds,
            )
            result["total_llm_calls"] += 1
            print(f"    Debater A current answer: {ans_a}")

            # Build partial round so Debater B can see A's argument
            partial_rounds = rounds + [{
                "round": round_num,
                "debater_a": resp_a,
                "debater_b": "(pending)",
            }]

            # Debater B responds
            resp_b, ans_b = self.debater_b.argue(
                question=question,
                choices=choices,
                rounds=partial_rounds,
            )
            result["total_llm_calls"] += 1
            print(f"    Debater B current answer: {ans_b}")

            # Record completed round
            round_record = {
                "round":      round_num,
                "debater_a":  resp_a,
                "answer_a":   ans_a,
                "debater_b":  resp_b,
                "answer_b":   ans_b,
            }
            rounds.append(round_record)

            # Early stopping check (only after min_rounds)
            if ans_a and ans_b and ans_a == ans_b:
                consecutive_agreements += 1
                print(f"    Agreement ({consecutive_agreements}/{self.early_stop_consecutive}): both say {ans_a}")
                if (round_num >= self.min_rounds and
                        consecutive_agreements >= self.early_stop_consecutive):
                    print(f"    Early stopping: {self.early_stop_consecutive} consecutive agreements.")
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
