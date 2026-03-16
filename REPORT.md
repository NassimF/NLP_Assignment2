# LLM Debate with Judge Pipeline
### CS6263 — LLM & Agentic Systems | Assignment 2
**Author:** Nasim Faridnia
**Date:** March 2026
**Repository:** https://github.com/NassimF/NLP_Assignment2

> **Disclosure:** Claude Code (Anthropic) was used to assist with code generation, debugging, and structure. All written content, design decisions, and analysis are my own.

---

## 1. Methodology

### 1.1 Research Question

Can a structured adversarial debate between two LLM agents, supervised by an LLM judge, produce more accurate and well-reasoned answers than a single LLM answering directly?

### 1.2 Task Domain

We evaluate on **ARC-Challenge** (Clark et al., 2018), a multiple-choice science question benchmark designed to challenge systems that cannot simply use retrieval or co-occurrence. We randomly sampled 200 questions (seed=42) from the test split, well above the 100-question minimum required.

### 1.3 System Architecture

The pipeline implements a 4-phase debate protocol inspired by Irving et al. (2018) and Liang et al. (EMNLP 2024):

#### Phase 1 — Initialization
Both debaters independently generate an initial answer and brief reasoning using the same prompt (`initial_position.txt`), without seeing each other's response. If both select the same answer, consensus is recorded and Phase 2 is skipped.

#### Phase 2 — Multi-Round Debate
When debaters disagree, a structured debate runs for a minimum of 3 and maximum of 5 rounds. In each round:
- **Debater A** argues in defense of its initial answer, with access to the full prior debate history
- **Debater B** responds with a counterargument, also with full history plus Debater A's current-round argument

An adaptive stopping criterion ends the debate early if both agents report the same answer for 2 consecutive rounds (after the minimum 3 rounds).

#### Phase 3 — Judgment
The judge receives the full debate transcript (initial positions + all rounds) and produces a structured evaluation with four required components per the assignment spec:
- **(a)** Chain-of-thought analysis of both debaters' arguments
- **(b)** Strongest and weakest argument from each debater
- **(c)** Final verdict — constrained to select between the two debaters' positions
- **(d)** Confidence score (1–5 scale)

#### Phase 4 — Evaluation
The judge's verdict is compared against the ground-truth answer. All intermediate data (initial positions, per-round arguments, judge reasoning, verdict, ground truth) is saved as JSON for every run.

### 1.4 Model Choices

| Role | Model | Rationale |
|---|---|---|
| Debater A (Proponent) | Llama-3.1-8B-Instruct | Lightweight, instruction-tuned; capable of structured CoT |
| Debater B (Opponent) | Qwen3-8B | Different architecture from Debater A introduces genuine diversity in reasoning paths |
| Judge + Baselines | Llama-3.1-70B-Instruct | Strongest available model; used for both judge and baselines to ensure fair comparison |

**Why the same model for judge and baselines?** The 70B model is the final decision-maker in the debate (the judge). Using it for the baselines isolates the effect of the debate process itself — if the baselines used a weaker model, any accuracy gap could reflect model size rather than the value of debate. This design ensures any difference in accuracy is attributable to the debate pipeline, not the underlying model.

All models are accessed via OpenAI-compatible APIs (vLLM servers) using the `openai` Python SDK.

### 1.5 Configuration and Hyperparameters

All hyperparameters are managed in `config.yaml`:

| Parameter | Value |
|---|---|
| Max debate rounds | 5 |
| Min debate rounds | 3 |
| Early stop consecutive agreements | 2 |
| Temperature | 0.7 |
| Max tokens (debaters) | 1024 |
| Max tokens (judge) | 2048 |
| Max tokens (baselines) | 2048 |
| Self-consistency samples (N) | 13 |
| Dataset size | 200 questions |

**Why N=13 for self-consistency?** The assignment specifies N should match total LLM calls in a full debate: 2 (initial positions) + 2 × 5 (debate rounds) + 1 (judge) = 13. This ensures a fair computational comparison.

---

## 2. Experiments

### 2.1 Experimental Setup

Three methods are compared on the same 200 ARC-Challenge questions:

1. **Debate** — Full 4-phase pipeline (Debater A + Debater B + Judge)
2. **Direct QA** — Single CoT call to the 70B model, no debate
3. **Self-Consistency** — 13 independent CoT samples from the 70B model, majority vote (Wang et al., 2023)

### 2.2 Results

> ⚠️ *Full debate results pending — table will be updated once the 200-question run completes.*

| Method | Accuracy | Parse Fail % | Avg LLM Calls | Avg Tokens/Q | Avg Latency/Q | p (vs Debate) |
|---|---|---|---|---|---|---|
| Debate | TBD | TBD | TBD | TBD | TBD | ref |
| Direct QA | 0.500 | 4.5% | 1 | 918 | 1.3s | TBD |
| Self-Consistency | 0.535 | 0% | 13 | 11,920 | 16.5s | TBD |

*Statistical significance tested using McNemar's test on paired per-question correctness.*

### 2.3 Figures

> ⚠️ *Figures will be inserted here after running `experiments/analyze_results.py`.*

- Accuracy by method (bar chart)
- Token usage by method (bar chart)
- Latency by method (bar chart)
- Judge confidence vs. accuracy (scatter plot, debate only)

---

## 3. Analysis

### 3.1 Qualitative Transcript Analysis

> ⚠️ *This section will be completed after the full debate run, with 3–5 selected transcripts from `logs/`.*

#### Known Failure Case: Persuasive but Wrong (AKDE&ED_2008_8_48)

In this question (ground truth: A), Debater A incorrectly defended answer C across all 5 rounds, while Debater B correctly defended A. Despite Debater B making the factually correct argument, the judge sided with Debater A's position (C) and gave a confidence score of 4/5.

This illustrates a known limitation of LLM-as-judge systems: **persuasiveness ≠ correctness**. The judge evaluates argument quality and rhetorical coherence, not factual accuracy. A well-structured but incorrect argument can outperform a correct but poorly-argued position. This aligns with theoretical concerns raised by Irving et al. (2018) — debate is only reliable when the judge can independently verify claims, which is not guaranteed for a judge operating purely on language.

### 3.2 Connection to Theoretical Predictions

Irving et al. (2018) predict that debate improves outcomes when: (a) one debater has the correct answer, and (b) the judge can distinguish strong from weak arguments. Our results will test this empirically — cases where both debaters are wrong (neither chose the correct answer) represent a fundamental ceiling for the debate system.

---

## 4. Prompt Engineering

### 4.1 Debater Prompts

Both debaters receive the same `initial_position.txt` in Phase 1 — a neutral prompt asking for step-by-step reasoning and a final answer in `MY CURRENT ANSWER: X` format.

In Phase 2, the prompts diverge by role:

**Debater A** (`debater_a.txt`) is explicitly assigned a position to defend (`YOUR ASSIGNED POSITION: {position}`). The prompt ends with `MY CURRENT ANSWER: {position}` hardcoded — Debater A cannot change its reported answer. This enforces the strict proponent role: Debater A commits to its initial answer for the entire debate.

**Debater B** (`debater_b.txt`) is the free-form opponent. It is instructed to identify flaws in Debater A's reasoning and defend whatever answer it believes is correct. `MY CURRENT ANSWER:` is left open-ended, allowing Debater B to switch answers across rounds.

**Why lock Debater A?** If both agents could freely change their answer, the debate loses its adversarial structure — they would converge independently, equivalent to running two Direct QA calls. Locking Debater A creates genuine pressure: convergence only happens when Debater B is persuaded. This also makes the early stopping criterion well-defined and non-trivial.

### 4.2 Judge Prompt

The judge prompt (`judge.txt`) required the most iteration. Key design decisions:

**Structured output enforcement:** The judge must produce 6 clearly labeled sections. The final answer tag (`FINAL ANSWER: C`) was placed before the confidence score after discovering the model consistently stopped after `CONFIDENCE:` without outputting `FINAL ANSWER:` — a token budget exhaustion issue.

**Constraining to debaters' positions:** An early version of the judge prompt allowed it to select any answer, leading to cases where the judge picked a third option neither debater defended. The assignment specifies the judge should determine "which debater was more persuasive" — implying selection between the two positions argued. The prompt was updated to explicitly state both debaters' final answers and require the verdict to be one of those two.

**Consensus case handling:** When both debaters agree in Phase 1, an initial implementation short-circuited the judge entirely. Re-reading the assignment ("skip to Phase 3" not "skip Phase 3") revealed Phase 3 is always required. The judge prompt for consensus cases includes a note acknowledging no debate rounds occurred and instructs the judge to evaluate the quality of the initial reasoning instead.

**Plain text formatting:** The 70B model frequently used markdown formatting in its responses (`**Final Answer:** C`, `**Answer:** [B]`) rather than the required plain text. Explicit instructions were added: *"Write all required tags as plain text exactly as shown — no markdown, no bold, no brackets around the letter."*

### 4.3 Parse Failure Iteration

The most significant prompt engineering challenge was getting the 70B model to reliably output parseable answers. Four rounds of fixes were required:

| Round | Fix Applied | Parse Failures | Accuracy |
|---|---|---|---|
| Initial | — | 109 / 200 (54.5%) | 0.28 |
| 1 | Handle `FINAL ANSWER: [C]` bracket variant | 83 / 200 (41.5%) | 0.41 |
| 2 | Handle `**Final Answer:**` bold markdown variants | 68 / 200 (34.0%) | 0.38 |
| 3 | Search raw text before stripping `<think>` blocks; bare letter fallback | 45 / 200 (22.5%) | 0.45 |
| 4 | `baseline_max_tokens=2048`; `"correct answer is X"` fallback pattern | **9 / 200 (4.5%)** | **0.50** |

**Root causes identified:**
1. The model wrapped its entire reasoning (including `FINAL ANSWER:`) inside `<think>` blocks — `strip_think()` removed it, leaving only a bare letter
2. The model used markdown bold formatting despite explicit instructions
3. Token truncation at 1024 tokens cut off responses before the answer tag
4. Non-standard answer phrasings (`"correct answer is B"`, bare single letter)

The final parser uses a multi-fallback strategy: search raw text → search stripped text → match "correct answer is X" → match bare single letter. Remaining 4.5% failures are genuine edge cases (mid-reasoning truncation, indecisive responses) that cannot be resolved without model-level changes.

---

## Appendix: Full Prompt Templates

### A. Initial Position (`prompts/initial_position.txt`)

```
You are an AI agent participating in a debate. Before the debate begins, you must independently form your initial position on the following multiple-choice question.

QUESTION:
{question}

ANSWER CHOICES:
{choices}

INSTRUCTIONS:
1. Think step by step to reason through the question.
2. Select the answer you believe is most likely correct.
3. Provide a brief (2-3 sentence) justification for your choice.
4. End your response with the following line exactly:
MY CURRENT ANSWER: [single letter only, e.g. C]

Your response:
```

### B. Debater A (`prompts/debater_a.txt`)

```
You are Debater A, the Proponent in an academic debate. Your role is to argue IN FAVOR of the answer assigned to you for the following multiple-choice question.

QUESTION:
{question}

ANSWER CHOICES:
{choices}

YOUR ASSIGNED POSITION: You are defending answer {position}.

DEBATE HISTORY SO FAR:
{debate_history}

INSTRUCTIONS:
1. Begin with a brief restatement of your position.
2. Think step by step (chain-of-thought): lay out your reasoning clearly before stating your argument.
3. Present at least two distinct pieces of evidence or logical reasoning in support of your answer.
4. If the debate history is not empty, directly address and rebut the most recent argument made by Debater B. Identify the flaw in their reasoning.
5. Be concise but precise. Do not repeat arguments already made in previous rounds.
6. End your response with the following line exactly:
MY CURRENT ANSWER: {position}

Your response:
```

### C. Debater B (`prompts/debater_b.txt`)

```
You are Debater B, the Opponent in an academic debate. Your role is to challenge the position argued by Debater A and defend the answer you believe is most correct.

QUESTION:
{question}

ANSWER CHOICES:
{choices}

DEBATE HISTORY SO FAR:
{debate_history}

INSTRUCTIONS:
1. Read Debater A's most recent argument carefully.
2. Think step by step (chain-of-thought): identify the weakest point in Debater A's reasoning before constructing your counterargument.
3. Present at least two distinct counterarguments or pieces of evidence that contradict Debater A's position.
4. Clearly state which answer YOU believe is correct and why.
5. Be concise but precise. Do not repeat arguments already made in previous rounds.
6. End your response with the following line exactly:
MY CURRENT ANSWER: [the single letter of the answer you are defending, e.g. A]

Your response:
```

### D. Judge (`prompts/judge.txt`)

```
You are an impartial Judge evaluating a structured debate between two AI agents on a multiple-choice question. Your task is to carefully analyze the debate and render a final verdict.

QUESTION:
{question}

ANSWER CHOICES:
{choices}

FULL DEBATE TRANSCRIPT:
{debate_transcript}

DEBATER POSITIONS:
- Debater A is defending answer: {answer_a}
- Debater B is defending answer: {answer_b}

{consensus_note}
INSTRUCTIONS:
Provide your evaluation in the following structured format. Do not skip any section.
IMPORTANT: Write all required tags (FINAL ANSWER, CONFIDENCE) as plain text exactly as shown — no markdown, no bold, no brackets around the letter.
IMPORTANT: Your FINAL ANSWER must be either {answer_a} or {answer_b}. Do not select any other answer.

### 1. Chain-of-Thought Analysis
Think step by step. Evaluate the quality of each debater's reasoning in their initial positions and any subsequent rounds. What were the key arguments made? How sound was the logic?

### 2. Debater A — Argument Assessment
- Strongest argument: [identify the single most compelling point made by Debater A]
- Weakest argument: [identify the least convincing or most flawed point made by Debater A]

### 3. Debater B — Argument Assessment
- Strongest argument: [identify the single most compelling point made by Debater B]
- Weakest argument: [identify the least convincing or most flawed point made by Debater B]

### 4. Verdict
Based on the quality of reasoning presented, select the answer defended by the more persuasive debater: [Debater A / Debater B]

The correct answer is: [single letter only, e.g. C]

### 5. Final Verdict
Write the answer letter as plain text exactly as shown — no brackets, no bold, no markdown:
FINAL ANSWER: C

### 6. Confidence Score
On a scale of 1 to 5, how confident are you in this verdict?
(1 = very uncertain, 3 = moderately confident, 5 = very confident)

CONFIDENCE: [1-5]
```

### E. Direct QA (`prompts/direct_qa.txt`)

```
You are an expert at answering multiple-choice questions. Answer the following question using careful step-by-step reasoning.

QUESTION:
{question}

ANSWER CHOICES:
{choices}

INSTRUCTIONS:
1. Think step by step before committing to an answer.
2. Consider each answer choice and explain why it is correct or incorrect.
3. Select the single best answer.
4. End your response with the following line exactly as shown — no markdown, no brackets, no bold:
FINAL ANSWER: C

Replace C with your chosen letter. Do not write "FINAL ANSWER: [C]" or "**FINAL ANSWER: C**". Write only the plain text line above.

Your response:
```
