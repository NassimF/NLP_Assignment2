"""
debater_b.py — Opponent agent (Qwen3-8B).

Challenges Debater A's position and argues for an alternative answer.
"""

from src.utils import (
    make_client, load_prompt, call_llm,
    format_choices, format_debate_history,
    extract_current_answer,
)


class DebaterB:
    def __init__(self, config: dict):
        self.config = config
        self.client, self.model = make_client("debater_b", config)

    def get_initial_position(self, question: str, choices: dict) -> tuple[dict, str | None]:
        """
        Phase 1: independently generate an initial answer before debate starts.

        Returns:
            (llm_result dict, answer_letter)
        """
        prompt = load_prompt(
            "initial_position",
            question=question,
            choices=format_choices(choices),
        )
        result = call_llm(self.client, self.model, prompt, self.config)
        answer = extract_current_answer(result["text"])
        return result, answer

    def argue(
        self,
        question: str,
        choices: dict,
        rounds: list[dict],
    ) -> tuple[dict, str | None]:
        """
        Phase 2: generate a counterargument against Debater A's position.

        Args:
            question: the question text
            choices: dict of answer choices
            rounds: list of prior debate round dicts

        Returns:
            (llm_result dict, current_answer_letter)
        """
        prompt = load_prompt(
            "debater_b",
            question=question,
            choices=format_choices(choices),
            debate_history=format_debate_history(rounds),
        )
        result = call_llm(self.client, self.model, prompt, self.config)
        answer = extract_current_answer(result["text"])
        return result, answer
