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
- [x] `src/baseline_runner.py` (BaselineRunner class)
- [x] `experiments/run_baselines.py` (Direct QA + Self-Consistency)
- [x] `experiments/run_debate.py`

### Step 7 — Evaluation Scripts
- [x] `experiments/analyze_results.py`
  - Accuracy per method + parse failure rate
  - Avg LLM calls / tokens / latency per question
  - Debate-only: avg rounds, consensus rate, early-stop rate
  - Confidence vs. accuracy scatter plot (debate)
  - Markdown comparison table to stdout + `results/comparison_table.csv`
  - Bar charts: accuracy, tokens, latency
  - McNemar's test (debate vs. direct_qa, debate vs. self_consistency)

### Step 8 — Web UI
- [ ] `ui/app.py` (Streamlit interface)

### Step 9 — Blog Post
- [ ] `REPORT.md`

### Step 10 — Bonus (optional)
- [ ] Multi-judge panel

---

## Changelog

### 2026-03-16
- Added `experiments/test_connections.py`: verifies API connectivity for all 3 model servers (Debater A, B, Judge) with a minimal call, reports status/latency/sample response
- All 3 servers confirmed OK; Qwen3 and 70B emit `<think>` blocks (handled by strip_think())
- Fixed `config.yaml`: set `self_consistency.num_samples=13` to match total LLM calls in a full debate (2 initial + 2×5 rounds + 1 judge)
- Updated `AGENTS.md`: added research question, 4-phase protocol description, and baseline definitions
- Documented Debater A locked-position design decision in `AGENTS.md`
- Fixed `src/utils.py` `format_debate_history()`: replace `"(pending)"` with `"(has not responded yet this round)"` for cleaner debate transcripts
- Fixed judge parse failures (4/5 in smoke test): two changes made:
  1. Moved `FINAL ANSWER:` before `CONFIDENCE:` in `prompts/judge.txt` — the judge was stopping naturally after `CONFIDENCE:` without outputting `FINAL ANSWER:`, likely due to token budget running out at the end of the structured response. Moving it earlier ensures the critical parse target is always generated first.
  2. Added `judge_max_tokens: 2048` in `config.yaml` and wired it through `call_llm()` — the judge's 5-section structured output (CoT analysis, two argument assessments, verdict, confidence) consistently approached the 1024 token limit, causing truncation. `judge_max_tokens` is a separate config key so debater calls are unaffected.
- Smoke test after fix: 0 parse failures, 5/5 correct (all Phase 1 consensus)

### 2026-03-15
- Added `experiments/analyze_results.py`: loads debate_summary.json + baseline JSONs, prints markdown comparison table, saves CSV, generates 4 figures (accuracy/tokens/latency bar charts + confidence vs. accuracy scatter), runs McNemar's test for statistical significance

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
- Added `src/baseline_runner.py`: BaselineRunner with Direct QA and Self-Consistency methods
- Added `experiments/run_debate.py`: runs debate pipeline with --limit/--offset args, saves summary JSON
- Added `experiments/run_baselines.py`: runs one or both baselines with --method/--limit args, saves summary JSON
