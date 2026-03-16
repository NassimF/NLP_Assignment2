# AGENTS.md — AI Assistant Instructions

This file provides context and conventions for AI coding assistants (Claude, Copilot, etc.) working on this project.

## Project Summary

**Research question:** Can a structured adversarial debate between two LLM agents, supervised by an LLM judge, produce more accurate and well-reasoned answers than a single LLM answering directly?

A multi-agent LLM debate pipeline for Commonsense QA (ARC-Challenge, 200 questions).

### Agents
- **Debater A**: Llama-3.1-8B-Instruct — Proponent; argues in favor of its assigned answer
- **Debater B**: Qwen3-8B — Opponent; challenges Debater A and defends the answer it believes is correct
- **Judge**: Llama-3.1-70B-Instruct — Evaluates the full debate transcript and renders a final verdict

All models accessed via OpenAI-compatible API (`openai` Python SDK with custom `base_url`).

### 4-Phase Debate Protocol
1. **Initialization** — Both debaters independently generate an initial position (answer + reasoning). If they agree, skip to Phase 3.
2. **Multi-Round Debate** — N rounds (min 3, max 5) of structured argumentation with CoT. Early stopping if both agents agree for 2 consecutive rounds.
3. **Judgment** — Judge receives full transcript and produces: (a) CoT analysis, (b) strongest/weakest args per debater, (c) final verdict, (d) confidence score (1–5).
4. **Evaluation** — Judge verdict compared to ground truth; full result saved as JSON.

### Baselines
- **Direct QA**: Single CoT call to the 70B model, no debate.
- **Self-Consistency**: 13 independent samples (= total LLM calls in a max-length debate) from the 70B model, majority vote.

## Code Conventions
- Python 3.10+
- All hyperparameters must come from `config.yaml` — never hardcode model names, temperatures, or paths
- API keys loaded from `.env` via `python-dotenv` — never hardcode credentials
- All agent classes live in `src/agents/`; orchestration logic in `src/debate_orchestrator.py`
- Prompt templates stored as `.txt` files in `prompts/` with `{variable}` placeholders
- Every debate run must be saved as a JSON file in `logs/`
- Use `tqdm` for progress bars in experiment scripts

## File Ownership
- `config.yaml` — single source of truth for all settings
- `prompts/` — all prompt text; never embed prompts directly in Python code
- `logs/` — auto-generated, gitignored
- `results/` — auto-generated, gitignored

## Do Not
- Do not commit `.env`
- Do not hardcode any model name, API URL, or key in source files
- Do not modify `REPORT.md` unless explicitly asked — it is the graded blog post
