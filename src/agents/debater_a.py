"""
debater_a.py — Proponent agent (Llama-3.1-8B-Instruct).

Argues IN FAVOR of its assigned position.
"""

from src.utils import (
    make_client, load_prompt, call_llm,
    format_choices, format_debate_history,
    extract_current_answer,
)


class DebaterA:
    def __init__(self, config: dict):
        self.config = config
        self.client, self.model = make_client("debater_a", config)

    def get_initial_position(self, question: str, choices: dict) -> tuple[str, str | None]:
        """
        Phase 1: independently generate an initial answer before debate starts.

        Returns:
            (response_text, answer_letter)
        """
        prompt = load_prompt(
            "initial_position",
            question=question,
            choices=format_choices(choices),
        )
        response = call_llm(self.client, self.model, prompt, self.config)
        answer = extract_current_answer(response)
        return response, answer

    def argue(
        self,
        question: str,
        choices: dict,
        position: str,
        rounds: list[dict],
    ) -> tuple[str, str | None]:
        """
        Phase 2: generate a debate argument defending {position}.

        Args:
            question: the question text
            choices: dict of answer choices
            position: the answer letter this debater is defending (e.g. 'C')
            rounds: list of prior debate round dicts (may be empty for Round 1)

        Returns:
            (response_text, current_answer_letter)
        """
        prompt = load_prompt(
            "debater_a",
            question=question,
            choices=format_choices(choices),
            position=position,
            debate_history=format_debate_history(rounds),
        )
        response = call_llm(self.client, self.model, prompt, self.config)
        answer = extract_current_answer(response)
        return response, answer
