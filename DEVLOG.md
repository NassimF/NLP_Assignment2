# DEVLOG — LLM Debate Assignment

## TODO

### Step 1 — Repository Skeleton
- [x] `git init`, rename branch to `main`
- [x] Create directory structure
- [x] `.gitignore`, `.env.example`, `config.yaml`, `requirements.txt`
- [x] `README.md`, `AGENTS.md`, `DEVLOG.md`
- [ ] Initial git commit

### Step 2 — Data Pipeline
- [ ] Download ARC-Challenge dataset
- [ ] Preprocess and save as JSON

### Step 3 — Prompt Templates
- [ ] `prompts/debater_a.txt`
- [ ] `prompts/debater_b.txt`
- [ ] `prompts/judge.txt`
- [ ] `prompts/direct_qa.txt` (baseline)

### Step 4 — Agent Modules
- [ ] `src/utils.py` (config loader, API client, prompt loader)
- [ ] `src/agents/debater_a.py`
- [ ] `src/agents/debater_b.py`
- [ ] `src/agents/judge.py`

### Step 5 — Debate Orchestrator
- [ ] `src/debate_orchestrator.py` (4-phase pipeline + JSON logging)

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
