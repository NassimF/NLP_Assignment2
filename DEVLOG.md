# DEVLOG — LLM Debate Assignment

## TODO

### Step 1 — Repository Skeleton
- [x] `git init`, rename branch to `main`
- [x] Create directory structure
- [x] `.gitignore`, `.env.example`, `config.yaml`, `requirements.txt`
- [x] `README.md`, `AGENTS.md`, `DEVLOG.md`
- [x] Initial git commit

### Step 2 — Data Pipeline
- [x] Download ARC-Challenge dataset
- [x] Preprocess and save as JSON

### Step 3 — Prompt Templates
- [x] `prompts/debater_a.txt`
- [x] `prompts/debater_b.txt`
- [x] `prompts/judge.txt`
- [x] `prompts/direct_qa.txt` (baseline)
- [x] `prompts/initial_position.txt` (Phase 1 independent answer)

### Step 4 — Agent Modules
- [x] `src/utils.py` (config loader, API client, prompt loader)
- [x] `src/agents/debater_a.py`
- [x] `src/agents/debater_b.py`
- [x] `src/agents/judge.py`

### Step 5 — Debate Orchestrator
- [x] `src/debate_orchestrator.py` (4-phase pipeline + JSON logging)

### Step 6 — Baseline Runners
- [ ] `experiments/run_baselines.py` (Direct QA + Self-Consistency)
- [ ] `experiments/run_debate.py`

### Step 7 — Evaluation Scripts
- [ ] `experiments/analyze_results.py`

### Step 8 — Web UI
- [ ] `ui/app.py` (Streamlit interface)

### Step 9 — Blog Post
- [ ] `REPORT.md`

### Step 10 — Bonus (optional)
- [ ] Multi-judge panel

---

## Changelog

### 2026-03-11
- Initialized git repository, renamed branch to `main`
- Created full project directory structure
- Added: `.gitignore`, `.env.example`, `config.yaml`, `requirements.txt`, `README.md`, `AGENTS.md`, `DEVLOG.md`
- Added `data/download_data.py` to fetch and preprocess ARC-Challenge from HuggingFace
- Generated `data/arc_challenge_test.json` (1172 questions) and `data/arc_challenge_200.json` (200-question sample, seed=42)
- Added prompt templates: `debater_a.txt`, `debater_b.txt`, `judge.txt`, `direct_qa.txt`, `initial_position.txt`
- Prompts enforce structured CoT output and `MY CURRENT ANSWER: X` format for reliable parsing
- Added `src/utils.py`: config loader, API client factory, prompt renderer, answer parsers, `call_llm` wrapper
- Added `src/agents/debater_a.py`: DebaterA class (Llama-3.1-8B) with `get_initial_position` and `argue` methods
- Added `src/agents/debater_b.py`: DebaterB class (Qwen3-8B) with `get_initial_position` and `argue` methods
- Added `src/agents/judge.py`: Judge class (Llama-3.1-70B) with `evaluate` method returning structured verdict dict
- Note: `strip_think()` in utils handles `<think>` blocks emitted by Qwen3
- Added `src/debate_orchestrator.py`: full 4-phase pipeline with adaptive early stopping and per-question JSON logging
