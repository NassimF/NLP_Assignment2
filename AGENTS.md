# AGENTS.md — AI Assistant Instructions

This file provides context and conventions for AI coding assistants (Claude, Copilot, etc.) working on this project.

## Project Summary
A multi-agent LLM debate pipeline for Commonsense QA (ARC-Challenge).
- **Debater A**: Llama-3.1-8B-Instruct (Proponent)
- **Debater B**: Qwen3-8B (Opponent)
- **Judge**: Llama-3.1-70B-Instruct
- All models accessed via OpenAI-compatible API (`openai` Python SDK with custom `base_url`)

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
