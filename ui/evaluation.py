"""Evaluation metrics: offline QAEvalChain suite results + live operational metrics from logs."""
import pandas as pd
import plotly.express as px
import streamlit as st

from hca import db
from hca.evaluation import MODULES, Evaluator, tool_metrics
from ui.common import df

METRIC_LABELS = {
    "history_accuracy": "History summary accuracy", "search_accuracy": "Search answer accuracy",
    "search_groundedness": "Search groundedness", "search_citation_rate": "Citation rate",
    "planner_tool_f1": "Planner tool F1", "planner_exact_match": "Planner exact plan",
    "planner_entity_accuracy": "Planner entity accuracy", "booking_success_rate": "Booking success",
    "grader_calibration": "Grader calibration", "tool_success_rate": "Tool success (all-time)",
}


def _run_panel() -> None:
    with st.expander("▶️ Run evaluation suite", expanded=not db.query("SELECT id FROM eval_runs LIMIT 1")):
        st.caption("Generates predictions with the agent model, then grades them in one batch with the independent "
                   "grader model (QAEvalChain / ContextQAEvalChain). A full run takes several minutes on a laptop.")
        modules = st.multiselect("Modules", MODULES, default=list(MODULES))
        if st.button("Run evaluation", type="primary") and modules:
            with st.status("Evaluating ...", expanded=True) as s:
                result = Evaluator(on_progress=s.write).run(tuple(modules))
                s.update(label=f"Evaluation #{result['eval_run_id']} complete", state="complete")
            st.rerun()


def _latest() -> None:
    runs = db.query("SELECT * FROM eval_runs ORDER BY id DESC")
    if not runs:
        st.info("No evaluation runs yet.")
        return
    run_labels = {f"#{r['id']} · {r['created_at'][:16].replace('T', ' ')} · {', '.join(r['metrics'].get('modules', []))}": r
                  for r in runs}
    run = run_labels[st.selectbox("Evaluation run", list(run_labels))]
    m = run["metrics"]
    st.caption(f"Agent model `{m.get('agent_model', '?')}` · grader `{m.get('grader_model', '?')}`")
    shown = [(k, v) for k, v in METRIC_LABELS.items() if m.get(k) is not None]
    for i in range(0, len(shown), 5):
        cols = st.columns(5)
        for col, (k, label) in zip(cols, shown[i:i + 5]):
            col.metric(label, f"{m[k]:.0%}")
    if m.get("agent_avg_latency_s"):
        st.caption(f"Average end-to-end agent latency (booking cases): {m['agent_avg_latency_s']:.1f}s")

    results = db.query("SELECT * FROM eval_results WHERE eval_run_id=?", (run["id"],))
    for module in [x for x in (*MODULES, "grader") if any(r["module"] == x for r in results)]:
        rows = [r for r in results if r["module"] == module]
        with st.expander(f"{module} — {sum(r['grade'] == 'CORRECT' for r in rows)}/{len(rows)} correct"):
            st.dataframe(df(rows, ["case_id", "grade", "score", "query", "expected", "predicted", "details"]),
                         hide_index=True, width="stretch")

    if len(runs) > 1:
        trend = pd.DataFrame([{"run": r["id"], "metric": METRIC_LABELS[k], "value": r["metrics"][k]}
                              for r in runs for k in METRIC_LABELS if r["metrics"].get(k) is not None])
        st.plotly_chart(px.line(trend, x="run", y="value", color="metric", markers=True, title="Metrics across runs"),
                        width="stretch")


def _live() -> None:
    st.subheader("Live operational metrics")
    tm = tool_metrics()
    if not tm["tool_calls"]:
        st.info("No tool calls logged yet.")
        return
    per_tool = pd.DataFrame([{"tool": k, **v} for k, v in tm["per_tool"].items()])
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(per_tool, x="tool", y="success_rate", text="calls", range_y=[0, 1.05],
                           title="Tool success rate (label = calls)"), width="stretch")
    c2.plotly_chart(px.bar(per_tool, x="tool", y="avg_ms", title="Average tool latency (ms)", log_y=True),
                    width="stretch")
    runs = pd.DataFrame(db.query("SELECT source, success, latency_ms, created_at FROM agent_runs"))
    if not runs.empty:
        by_source = runs.groupby("source").agg(runs=("success", "size"), success_rate=("success", "mean"),
                                               avg_latency_s=("latency_ms", lambda x: x.mean() / 1000)).reset_index()
        c1, c2 = st.columns(2)
        c1.dataframe(by_source, hide_index=True, width="stretch")
        c2.plotly_chart(px.histogram(runs, x=runs["latency_ms"] / 1000, nbins=20, title="Agent run latency (s)"),
                        width="stretch")
    book = db.query_one("SELECT COUNT(*) n, AVG(success) rate FROM tool_logs WHERE tool='book_appointment' AND run_id!='eval'")
    if book["n"]:
        st.metric("Live booking success rate", f"{book['rate']:.0%}", help=f"{book['n']} booking attempts")


def page() -> None:
    st.title("📊 Evaluation & monitoring")
    _run_panel()
    _latest()
    st.divider()
    _live()
