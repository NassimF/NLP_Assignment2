"""
judge.py — Judge agent (Llama-3.1-70B-Instruct-custom).

Evaluates the full debate transcript and renders a structured verdict.
"""

from src.utils import (
    make_client, load_prompt, call_llm,
    format_choices, format_debate_transcript,
    extract_final_answer, extract_confidence,
)


class Judge:
    def __init__(self, config: dict):
        self.config = config
        self.client, self.model = make_client("judge", config)

    def evaluate(
        self,
        question: str,
        choices: dict,
        initial_a: str,
        initial_b: str,
        rounds: list[dict],
    ) -> dict:
        """
        Phase 3: evaluate the full debate and render a verdict.

        Args:
            question: the question text
            choices: dict of answer choices
            initial_a: Debater A's Phase 1 response text
            initial_b: Debater B's Phase 1 response text
            rounds: list of all debate round dicts

        Returns:
            dict with keys:
                raw_response        - full judge response text
                final_answer        - parsed answer letter (e.g. 'C') or None
                confidence          - int 1-5 or None
                prompt_tokens       - input tokens
                completion_tokens   - output tokens
                total_tokens        - total tokens
                latency_seconds     - API call duration
        """
        transcript = format_debate_transcript(initial_a, initial_b, rounds)

        prompt = load_prompt(
            "judge",
            question=question,
            choices=format_choices(choices),
            debate_transcript=transcript,
        )

        result = call_llm(
            self.client,
            self.model,
            prompt,
            self.config,
            system_prompt=(
                "You are an impartial and rigorous judge. "
                "Evaluate arguments based on logic, evidence, and reasoning quality. "
                "Follow the output format exactly."
            ),
            max_tokens=self.config["generation"].get("judge_max_tokens", 2048),
        )

        return {
            "raw_response":      result["text"],
            "final_answer":      extract_final_answer(result["text"]),
            "confidence":        extract_confidence(result["text"]),
            "prompt_tokens":     result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens":      result["total_tokens"],
            "latency_seconds":   result["latency_seconds"],
        }
