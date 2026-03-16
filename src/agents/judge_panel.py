"""
judge_panel.py — Multi-judge panel with optional deliberation.

Architecture:
  Round 1: N judges independently evaluate the debate transcript.
  Round 2: If judges disagree, each judge sees the other judges' Round 1
           reasoning and reconsiders. Only triggered when deliberate=True.
  Final verdict: majority vote of Round 2 verdicts (or Round 1 if unanimous).

JudgePanel.evaluate() returns the same dict shape as Judge.evaluate() so it
can be passed into DebateOrchestrator as a drop-in replacement.
"""

from collections import Counter

from src.utils import (
    make_client, load_prompt, call_llm,
    format_choices, format_debate_transcript,
    extract_final_answer, extract_confidence,
)


class JudgePanel:
    def __init__(self, config: dict):
        self.config = config
        self.n = config["panel"]["num_judges"]
        self.deliberate = config["panel"]["deliberate"]
        # All panel judges share the same model endpoint
        self.client, self.model = make_client("judge", config)

    # ─── Public API ───────────────────────────────────────────────────────────

    def evaluate(
        self,
        question: str,
        choices: dict,
        initial_a: str,
        initial_b: str,
        rounds: list[dict],
        answer_a: str | None = None,
        answer_b: str | None = None,
        consensus: bool = False,
    ) -> dict:
        """
        Run N-judge panel evaluation with optional deliberation.

        Returns a dict compatible with Judge.evaluate() plus panel-specific keys:
            panel_r1_results    - list of N Round 1 judge result dicts
            panel_r2_results    - list of N Round 2 result dicts, or None
            panel_verdicts      - final verdict from each judge (after R2 if run)
            panel_unanimous_r1  - True if all R1 verdicts agreed
            panel_deliberated   - True if Round 2 was actually run
        """
        transcript = format_debate_transcript(initial_a, initial_b, rounds)

        if consensus:
            consensus_note = (
                f"NOTE: Both debaters independently reached consensus on answer "
                f"{answer_a} before any debate rounds. There are no debate rounds "
                f"to evaluate — assess only the quality of their initial reasoning "
                f"and confirm or override this consensus with your verdict."
            )
        else:
            consensus_note = ""

        # ── Round 1: independent verdicts ─────────────────────────────────────
        r1_results = [
            self._single_evaluate(
                question, choices, transcript, answer_a, answer_b, consensus_note,
                judge_num=i + 1,
            )
            for i in range(self.n)
        ]

        r1_verdicts = [r["final_answer"] for r in r1_results]
        unanimous_r1 = len({v for v in r1_verdicts if v}) <= 1

        # ── Round 2: deliberation on disagreement ──────────────────────────────
        r2_results = None
        if not unanimous_r1 and self.deliberate:
            r2_results = [
                self._deliberate(
                    judge_idx=i,
                    r1_results=r1_results,
                    question=question,
                    choices=choices,
                    transcript=transcript,
                    answer_a=answer_a,
                    answer_b=answer_b,
                    consensus_note=consensus_note,
                )
                for i in range(self.n)
            ]

        final_results = r2_results if r2_results is not None else r1_results
        final_verdicts = [r["final_answer"] for r in final_results]

        # ── Majority vote ──────────────────────────────────────────────────────
        valid = [v for v in final_verdicts if v]
        panel_verdict = Counter(valid).most_common(1)[0][0] if valid else None

        # ── Aggregate usage ────────────────────────────────────────────────────
        all_calls = r1_results + (r2_results or [])
        usage = {
            "prompt_tokens":     sum(r.get("prompt_tokens") or 0 for r in all_calls),
            "completion_tokens": sum(r.get("completion_tokens") or 0 for r in all_calls),
            "total_tokens":      sum(r.get("total_tokens") or 0 for r in all_calls),
            "latency_seconds":   sum(r.get("latency_seconds") or 0.0 for r in all_calls),
        }

        # Confidence: average of non-None scores from final round, rounded
        conf_scores = [r["confidence"] for r in final_results if r.get("confidence")]
        panel_confidence = round(sum(conf_scores) / len(conf_scores)) if conf_scores else None

        return {
            # ── Standard Judge.evaluate() keys ────────────────────────────────
            "raw_response":      "\n\n---\n\n".join(
                                     r.get("raw_response", "") for r in final_results
                                 ),
            "final_answer":      panel_verdict,
            "confidence":        panel_confidence,
            "prompt_tokens":     usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens":      usage["total_tokens"],
            "latency_seconds":   usage["latency_seconds"],
            # ── Panel-specific keys ────────────────────────────────────────────
            "panel_r1_results":   r1_results,
            "panel_r2_results":   r2_results,
            "panel_verdicts":     final_verdicts,
            "panel_unanimous_r1": unanimous_r1,
            "panel_deliberated":  r2_results is not None,
        }

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _single_evaluate(
        self,
        question: str,
        choices: dict,
        transcript: str,
        answer_a: str | None,
        answer_b: str | None,
        consensus_note: str,
        judge_num: int,
    ) -> dict:
        """One independent judge evaluation using the standard judge.txt prompt."""
        prompt = load_prompt(
            "judge",
            question=question,
            choices=format_choices(choices),
            debate_transcript=transcript,
            answer_a=answer_a or "?",
            answer_b=answer_b or "?",
            consensus_note=consensus_note,
        )

        result = call_llm(
            self.client,
            self.model,
            prompt,
            self.config,
            system_prompt=(
                f"You are Judge {judge_num} of {self.n} on an impartial evaluation panel. "
                "Evaluate arguments based on logic, evidence, and reasoning quality. "
                "Follow the output format exactly."
            ),
            max_tokens=self.config["generation"].get("judge_max_tokens", 2048),
        )

        return {
            "judge_num":         judge_num,
            "raw_response":      result["text"],
            "final_answer":      extract_final_answer(result["text"]),
            "confidence":        extract_confidence(result["text"]),
            "prompt_tokens":     result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens":      result["total_tokens"],
            "latency_seconds":   result["latency_seconds"],
        }

    def _deliberate(
        self,
        judge_idx: int,
        r1_results: list[dict],
        question: str,
        choices: dict,
        transcript: str,
        answer_a: str | None,
        answer_b: str | None,
        consensus_note: str,
    ) -> dict:
        """Round 2: this judge sees the other judges' R1 reasoning and reconsiders."""
        judge_num = judge_idx + 1
        peers = [r for i, r in enumerate(r1_results) if i != judge_idx]

        peer_block = ""
        for p in peers:
            verdict = p["final_answer"] or "unclear"
            peer_block += (
                f"--- Judge {p['judge_num']} ---\n"
                f"{p['raw_response']}\n"
                f"Verdict: {verdict}\n\n"
            )

        prompt = load_prompt(
            "judge_deliberation",
            judge_num=judge_num,
            num_judges=self.n,
            question=question,
            choices=format_choices(choices),
            debate_transcript=transcript,
            answer_a=answer_a or "?",
            answer_b=answer_b or "?",
            consensus_note=consensus_note,
            peer_evaluations=peer_block.strip(),
        )

        result = call_llm(
            self.client,
            self.model,
            prompt,
            self.config,
            system_prompt=(
                f"You are Judge {judge_num} of {self.n} on an impartial evaluation panel. "
                "You are now in the deliberation round. "
                "Evaluate arguments based on logic, evidence, and reasoning quality. "
                "Follow the output format exactly."
            ),
            max_tokens=self.config["generation"].get("judge_max_tokens", 2048),
        )

        return {
            "judge_num":         judge_num,
            "raw_response":      result["text"],
            "final_answer":      extract_final_answer(result["text"]),
            "confidence":        extract_confidence(result["text"]),
            "prompt_tokens":     result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens":      result["total_tokens"],
            "latency_seconds":   result["latency_seconds"],
        }
