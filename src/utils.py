"""
utils.py — Shared utilities for the LLM Debate Pipeline.

Provides:
  - Config and env loading
  - OpenAI client factory (per role)
  - Prompt template loading and rendering
  - Answer parsing (MY CURRENT ANSWER / FINAL ANSWER)
  - Debate history formatting
  - Choice dict formatting
"""

import os
import re
import time
import yaml
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# ─── Paths ───────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent  # project root
PROMPTS_DIR = ROOT / "prompts"
CONFIG_PATH = ROOT / "config.yaml"


# ─── Config & Env ─────────────────────────────────────────────────────────────

def load_config() -> dict:
    """Load and return config.yaml as a dict."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_env():
    """Load .env file into environment variables."""
    load_dotenv(ROOT / ".env")


# ─── API Client Factory ───────────────────────────────────────────────────────

def make_client(role: str, config: dict) -> tuple[OpenAI, str]:
    """
    Create an OpenAI-compatible client for a given role.

    Args:
        role: one of 'debater_a', 'debater_b', 'judge', 'baseline'
        config: loaded config dict

    Returns:
        (client, model_name)
    """
    load_env()

    # Map role to judge/baseline (same server)
    api_role = role if role in ("debater_a", "debater_b") else "judge"

    url_env = config["api"][f"{api_role}_url_env"]
    key_env = config["api"][f"{api_role}_key_env"]

    base_url = os.getenv(url_env)
    api_key  = os.getenv(key_env)

    if not base_url:
        raise RuntimeError(f"Env var {url_env} is not set. Check your .env file.")
    if not api_key:
        raise RuntimeError(f"Env var {key_env} is not set. Check your .env file.")

    client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
    model  = config["models"][role]

    return client, model


# ─── Prompt Utilities ─────────────────────────────────────────────────────────

def load_prompt(template_name: str, **kwargs) -> str:
    """
    Load a prompt template from prompts/ and fill in {variable} placeholders.

    Args:
        template_name: filename without extension (e.g. 'debater_a')
        **kwargs: variables to substitute into the template

    Returns:
        Rendered prompt string
    """
    path = PROMPTS_DIR / f"{template_name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    template = path.read_text()
    return template.format(**kwargs)


def format_choices(choices: dict) -> str:
    """
    Convert choices dict to a readable string.

    Input:  {"A": "photosynthesis", "B": "respiration", ...}
    Output: "A) photosynthesis\nB) respiration\n..."
    """
    return "\n".join(f"{k}) {v}" for k, v in choices.items())


def format_debate_history(rounds: list[dict]) -> str:
    """
    Format debate history list into a readable transcript string.

    Each round dict has keys: 'round', 'debater_a', 'debater_b'
    Returns empty string if rounds is empty (Phase 1).
    """
    if not rounds:
        return "(No prior debate rounds — this is the opening argument.)"

    lines = []
    for r in rounds:
        lines.append(f"--- Round {r['round']} ---")
        lines.append(f"[Debater A]:\n{r['debater_a']}")
        if r.get("debater_b") and r["debater_b"] != "(pending)":
            lines.append(f"[Debater B]:\n{r['debater_b']}")
        else:
            lines.append("[Debater B]: (has not responded yet this round)")
        lines.append("")
    return "\n".join(lines)


def format_debate_transcript(
    initial_a: str,
    initial_b: str,
    rounds: list[dict]
) -> str:
    """
    Format the full debate transcript for the judge.
    Includes initial positions + all rounds.
    """
    lines = [
        "=== INITIAL POSITIONS ===",
        f"[Debater A (Proponent)]:\n{initial_a}",
        "",
        f"[Debater B (Opponent)]:\n{initial_b}",
        "",
        "=== DEBATE ROUNDS ===",
    ]
    for r in rounds:
        lines.append(f"--- Round {r['round']} ---")
        lines.append(f"[Debater A]:\n{r['debater_a']}")
        lines.append(f"[Debater B]:\n{r['debater_b']}")
        lines.append("")
    return "\n".join(lines)


# ─── Answer Parsing ───────────────────────────────────────────────────────────

def strip_think(text: str) -> str:
    """
    Remove <think>...</think> blocks emitted by reasoning models (e.g. Qwen3).
    """
    if not text:
        return text
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def extract_current_answer(text: str) -> str | None:
    """
    Parse 'MY CURRENT ANSWER: X' from debater responses.
    Returns the letter (e.g. 'C') or None if not found.
    """
    text = strip_think(text)
    match = re.search(r"MY CURRENT ANSWER:\s*([A-E])", text, re.IGNORECASE)
    return match.group(1).upper() if match else None


def extract_final_answer(text: str) -> str | None:
    """
    Parse the final answer letter from judge or direct QA responses.

    Search order:
      1. Look for FINAL ANSWER / Answer pattern in the raw text (before
         stripping <think> blocks) — catches cases where the model puts
         the answer inside a <think> block and only emits a bare letter
         outside it.
      2. Same search in the stripped text.
      3. Fallback: if the entire stripped response is a single A-E letter.

    Handles format variants:
      'FINAL ANSWER: C', 'FINAL ANSWER: [C]',
      '**Final Answer:** C', '**Final Answer**: C',
      '**Answer:** C', '**Final Answer:** [C]'
    """
    _PATTERN = r"(?:final\s+)?answer[*\s]*:[*\s]*\[?([A-E])\]?"

    # 1. Search raw text first (before stripping think blocks)
    match = re.search(_PATTERN, text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 2. Search stripped text
    stripped = strip_think(text)
    match = re.search(_PATTERN, stripped, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 3. Fallback: "the correct answer is X" / "correct answer is X"
    match = re.search(
        r"(?:the\s+)?correct\s+answer\s+is\s+\[?([A-E])\]?",
        stripped, re.IGNORECASE
    )
    if match:
        return match.group(1).upper()

    # 4. Fallback: entire response is a bare letter (model answered
    #    inside <think> and only emitted the letter outside)
    bare = stripped.strip()
    if re.fullmatch(r"[A-E]", bare, re.IGNORECASE):
        return bare.upper()

    return None


def extract_confidence(text: str) -> int | None:
    """
    Parse 'CONFIDENCE: N' (1-5) from judge responses.
    """
    match = re.search(r"CONFIDENCE:\s*([1-5])", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


# ─── LLM Call Wrapper ─────────────────────────────────────────────────────────

def call_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    config: dict,
    system_prompt: str = "You are a helpful assistant participating in a structured debate.",
    max_tokens: int | None = None,
) -> dict:
    """
    Make a single chat completion call and return a result dict with:
      - text             : response text (think blocks stripped)
      - prompt_tokens    : input token count
      - completion_tokens: output token count
      - total_tokens     : total token count
      - latency_seconds  : wall-clock time for the API call
    """
    gen = config["generation"]
    effective_max_tokens = max_tokens if max_tokens is not None else gen["max_tokens"]

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
        temperature=gen["temperature"],
        max_tokens=effective_max_tokens,
        top_p=gen["top_p"],
    )
    latency = time.perf_counter() - t0

    text = response.choices[0].message.content or ""
    text = strip_think(text.strip())

    usage = response.usage
    return {
        "text":              text,
        "prompt_tokens":     usage.prompt_tokens     if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "total_tokens":      usage.total_tokens      if usage else None,
        "latency_seconds":   round(latency, 3),
    }
