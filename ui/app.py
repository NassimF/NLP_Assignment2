"""
app.py — Streamlit UI for the LLM Debate Pipeline.

Three modes:
  1. Run Debate     — pick a question from the dataset and run a live debate
  2. Browse Logs    — explore past debate logs saved in logs/
  3. Results        — comparison table and figures from analyze_results.py
"""

import sys
import json
import glob
import csv
from pathlib import Path

import streamlit as st

# Allow imports from project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LLM Debate Pipeline",
    page_icon="⚖️",
    layout="wide",
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_data
def load_dataset():
    config = load_config()
    path = ROOT / config["dataset"]["data_dir"] / "arc_challenge_200.json"
    with open(path) as f:
        return json.load(f)


def load_log(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def verdict_badge(correct: bool) -> str:
    return "✅ Correct" if correct else "❌ Incorrect"


def confidence_bar(score) -> str:
    if score is None:
        return "N/A"
    filled = "🟦" * int(score)
    empty  = "⬜" * (5 - int(score))
    return f"{filled}{empty}  ({score}/5)"


def format_choices(choices: dict) -> str:
    return "\n".join(f"**{k})** {v}" for k, v in choices.items())


def render_debate_log(log: dict):
    """Render a full debate log in the UI."""

    # ── Header ────────────────────────────────────────────────────────────────
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"Question ID: `{log['id']}`")
        st.markdown(f"**{log['question']}**")
        st.markdown(format_choices(log["choices"]))
    with col2:
        st.metric("Ground Truth", log["ground_truth"])
        st.metric("Verdict", log.get("verdict") or "None")
        result_label = verdict_badge(log.get("correct", False))
        st.markdown(f"### {result_label}")

    st.divider()

    # ── Phase 1: Initial positions ─────────────────────────────────────────────
    st.markdown("### Phase 1 — Initial Positions")
    c1, c2 = st.columns(2)
    with c1:
        ans_a = log.get("initial_position_a", {}).get("answer", "?")
        st.markdown(f"**🔵 Debater A (Llama-3.1-8B)** — Answer: `{ans_a}`")
        with st.expander("Show reasoning"):
            st.markdown(log.get("initial_position_a", {}).get("response", "—"))
    with c2:
        ans_b = log.get("initial_position_b", {}).get("answer", "?")
        st.markdown(f"**🔴 Debater B (Qwen3-8B)** — Answer: `{ans_b}`")
        with st.expander("Show reasoning"):
            st.markdown(log.get("initial_position_b", {}).get("response", "—"))

    if log.get("consensus"):
        st.success(f"⚡ Phase 1 consensus reached — both agree on **{log['consensus_answer']}**. Skipping debate rounds.")

    # ── Phase 2: Debate rounds ─────────────────────────────────────────────────
    rounds = log.get("rounds", [])
    if rounds:
        st.markdown(f"### Phase 2 — Debate Rounds ({len(rounds)} round{'s' if len(rounds) > 1 else ''})")
        for r in rounds:
            with st.expander(f"Round {r['round']}  —  A: `{r.get('answer_a','?')}` vs B: `{r.get('answer_b','?')}`"):
                rc1, rc2 = st.columns(2)
                with rc1:
                    st.markdown(f"**🔵 Debater A** (current answer: `{r.get('answer_a','?')}`)")
                    st.markdown(r.get("debater_a", "—"))
                with rc2:
                    st.markdown(f"**🔴 Debater B** (current answer: `{r.get('answer_b','?')}`)")
                    st.markdown(r.get("debater_b", "—"))

    # ── Phase 3: Judge verdict ─────────────────────────────────────────────────
    st.markdown("### Phase 3 — Judge Verdict")
    judge = log.get("judge", {})
    jc1, jc2 = st.columns([2, 1])
    with jc1:
        with st.expander("Show full judge reasoning", expanded=True):
            st.markdown(judge.get("raw_response", "—"))
    with jc2:
        st.metric("Final Answer", judge.get("final_answer") or "None")
        st.markdown(f"**Confidence:** {confidence_bar(judge.get('confidence'))}")
        st.metric("Tokens used", judge.get("total_tokens", "—"))
        st.metric("Latency", f"{judge.get('latency_seconds', 0):.1f}s")

    # ── Usage summary ──────────────────────────────────────────────────────────
    st.divider()
    usage = log.get("usage", {})
    u1, u2, u3 = st.columns(3)
    u1.metric("Total LLM Calls", usage.get("total_llm_calls", "—"))
    u2.metric("Total Tokens", usage.get("total_tokens", "—"))
    u3.metric("Total Latency", f"{usage.get('total_latency_seconds', 0):.1f}s")


# ─── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("⚖️ LLM Debate Pipeline")
st.sidebar.markdown("**CS6263 — Assignment 2**")
st.sidebar.divider()

mode = st.sidebar.radio("Mode", ["🔴 Run Debate", "📂 Browse Logs", "📊 Results"])

# ─── Mode 1: Run Debate ───────────────────────────────────────────────────────

if mode == "🔴 Run Debate":
    st.title("Run a Live Debate")
    st.markdown("Select a question from the ARC-Challenge dataset and run the full 4-phase debate pipeline.")

    try:
        questions = load_dataset()
    except Exception as e:
        st.error(f"Could not load dataset: {e}")
        st.stop()

    # Question selector
    q_options = {f"[{q['id']}] {q['question'][:80]}...": q for q in questions}
    selected_label = st.selectbox("Select a question", list(q_options.keys()))
    selected_q = q_options[selected_label]

    st.markdown("**Answer choices:**")
    st.markdown(format_choices(selected_q["choices"]))
    st.markdown(f"*(Ground truth hidden until after the debate)*")

    st.divider()

    if st.button("▶️ Run Debate", type="primary"):
        try:
            from src.utils import load_config
            from src.debate_orchestrator import DebateOrchestrator

            config = load_config()

            with st.spinner("Running debate pipeline..."):
                progress = st.empty()

                progress.info("Phase 1: Getting initial positions...")
                orchestrator = DebateOrchestrator(config)
                result = orchestrator.run(selected_q)

            st.success("Debate complete!")
            render_debate_log(result)

        except Exception as e:
            st.error(f"Error running debate: {e}")
            st.exception(e)

# ─── Mode 2: Browse Logs ──────────────────────────────────────────────────────

elif mode == "📂 Browse Logs":
    st.title("Browse Debate Logs")

    log_dir = ROOT / "logs"
    debate_logs = sorted(
        glob.glob(str(log_dir / "*.json")),
        key=lambda p: Path(p).stat().st_mtime,
        reverse=True,
    )
    # Filter to debate logs only (exclude baseline logs)
    debate_logs = [p for p in debate_logs
                   if not Path(p).name.startswith(("direct_qa_", "self_consistency_"))]

    if not debate_logs:
        st.warning("No debate logs found. Run a debate first.")
        st.stop()

    # Summary stats
    total = len(debate_logs)
    logs_data = [load_log(p) for p in debate_logs]
    correct = sum(1 for d in logs_data if d.get("correct", False))
    consensus = sum(1 for d in logs_data if d.get("consensus", False))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Debates", total)
    c2.metric("Accuracy", f"{correct/total:.1%}" if total else "—")
    c3.metric("Consensus Rate", f"{consensus/total:.1%}" if total else "—")
    c4.metric("Avg Rounds", f"{sum(len(d.get('rounds',[])) for d in logs_data)/total:.1f}" if total else "—")

    st.divider()

    # Filter controls
    col1, col2 = st.columns(2)
    with col1:
        filter_correct = st.selectbox("Filter by result", ["All", "Correct only", "Incorrect only"])
    with col2:
        filter_consensus = st.selectbox("Filter by consensus", ["All", "Consensus", "Full debate"])

    filtered = logs_data
    if filter_correct == "Correct only":
        filtered = [d for d in filtered if d.get("correct")]
    elif filter_correct == "Incorrect only":
        filtered = [d for d in filtered if not d.get("correct")]
    if filter_consensus == "Consensus":
        filtered = [d for d in filtered if d.get("consensus")]
    elif filter_consensus == "Full debate":
        filtered = [d for d in filtered if not d.get("consensus")]

    if not filtered:
        st.info("No logs match the current filters.")
        st.stop()

    # Log selector
    options = {
        f"{'✅' if d.get('correct') else '❌'} [{d['id']}] {d['question'][:70]}...": d
        for d in filtered
    }
    selected_label = st.selectbox(f"Select a debate ({len(filtered)} available)", list(options.keys()))
    selected_log = options[selected_label]

    st.divider()
    render_debate_log(selected_log)

# ─── Mode 3: Results ──────────────────────────────────────────────────────────

elif mode == "📊 Results":
    st.title("Experiment Results")
    st.markdown("Comparison of debate pipeline vs. baselines on ARC-Challenge (200 questions).")

    config      = load_config()
    results_dir = ROOT / config["logging"]["results_dir"]
    csv_path    = results_dir / "comparison_table.csv"
    fig_dir     = results_dir / "figures"

    # ── Comparison table ──────────────────────────────────────────────────────
    st.markdown("### Comparison Table")
    if csv_path.exists():
        rows = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        st.dataframe(rows, use_container_width=True)
    else:
        st.warning(f"No comparison table found at `{csv_path}`. Run `experiments/analyze_results.py` first.")

    st.divider()

    # ── Figures ───────────────────────────────────────────────────────────────
    st.markdown("### Figures")

    figures = {
        "Accuracy by Method":  fig_dir / "accuracy_by_method.png",
        "Token Usage by Method": fig_dir / "tokens_by_method.png",
        "Latency by Method":   fig_dir / "latency_by_method.png",
        "Confidence vs. Accuracy (Debate)": fig_dir / "confidence_vs_accuracy.png",
    }

    available = {k: v for k, v in figures.items() if v.exists()}

    if not available:
        st.warning(f"No figures found in `{fig_dir}`. Run `experiments/analyze_results.py` first.")
    else:
        cols = st.columns(2)
        for i, (title, path) in enumerate(available.items()):
            with cols[i % 2]:
                st.markdown(f"**{title}**")
                st.image(str(path))

    # ── McNemar p-values note ─────────────────────────────────────────────────
    if csv_path.exists() and rows:
        st.divider()
        st.markdown("### Statistical Significance")
        st.markdown("McNemar's test p-values (debate vs. each baseline) are included in the comparison table above under the `p_vs_debate` column.")
        st.caption("p < 0.05 indicates a statistically significant difference in accuracy between the debate system and that baseline.")
