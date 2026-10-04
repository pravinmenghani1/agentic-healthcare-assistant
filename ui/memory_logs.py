"""Agent memory traces, planning breakdowns and tool-usage logs."""
import pandas as pd
import plotly.express as px
import streamlit as st

from hca import db, memory
from ui.common import df, render_plan


def _runs() -> None:
    runs = db.query("SELECT r.*, p.name patient FROM agent_runs r LEFT JOIN patients p ON p.id=r.patient_id "
                    "ORDER BY r.created_at DESC LIMIT 200")
    if not runs:
        st.info("No agent runs yet — ask the assistant something.")
        return
    st.dataframe(df(runs, ["created_at", "id", "source", "query", "patient", "success", "latency_ms"]),
                 hide_index=True, width="stretch")
    labels = {f"{r['created_at'][11:19]} · {r['id']} · {r['query'][:70]}": r for r in runs}
    run = labels[st.selectbox("Inspect run", list(labels))]
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Planning breakdown**")
        render_plan(run["plan"], run["trace"])
    with c2:
        st.markdown("**Trace timeline**")
        tr = pd.DataFrame([{"node": t["node"], "ms": t["ms"]} for t in run["trace"]])
        st.plotly_chart(px.bar(tr, x="ms", y="node", orientation="h").update_yaxes(autorange="reversed"),
                        width="stretch")
    st.markdown("**Node details**")
    for t in run["trace"]:
        with st.expander(f"{t['node']} · {t['ms']} ms"):
            st.json(t["detail"], expanded=False)
    st.markdown("**Tool calls in this run**")
    st.dataframe(df(db.query("SELECT * FROM tool_logs WHERE run_id=? ORDER BY id", (run["id"],)),
                    ["created_at", "tool", "success", "latency_ms", "error", "args", "output_preview"]),
                 hide_index=True, width="stretch")
    st.markdown("**Memory written by this run**")
    st.dataframe(df(db.query("SELECT * FROM memory_events WHERE run_id=?", (run["id"],)),
                    ["created_at", "patient_id", "kind", "content"]), hide_index=True, width="stretch")
    with st.expander("Final response"):
        st.markdown(run["response"])


def _tool_logs() -> None:
    tools = [r["tool"] for r in db.query("SELECT DISTINCT tool FROM tool_logs ORDER BY tool")]
    c1, c2, c3 = st.columns(3)
    sel = c1.multiselect("Tools", tools)
    outcome = c2.selectbox("Outcome", ["all", "success", "failure"])
    source = c3.selectbox("Source", ["all", "agent runs", "ui", "eval"])
    sql, params = "SELECT * FROM tool_logs WHERE 1=1", []
    if sel:
        sql += f" AND tool IN ({','.join('?' * len(sel))})"
        params += sel
    if outcome != "all":
        sql += " AND success=?"
        params.append(int(outcome == "success"))
    if source == "ui":
        sql += " AND run_id='ui'"
    elif source == "eval":
        sql += " AND run_id='eval'"
    elif source == "agent runs":
        sql += " AND run_id NOT IN ('ui','eval')"
    rows = db.query(sql + " ORDER BY id DESC LIMIT 500", tuple(params))
    st.dataframe(df(rows, ["created_at", "run_id", "tool", "success", "latency_ms", "error", "args", "output_preview"]),
                 hide_index=True, width="stretch")
    if rows:
        frame = pd.DataFrame(rows)
        frame["outcome"] = frame["success"].map({1: "success", 0: "failure"})
        st.plotly_chart(px.histogram(frame, x="tool", color="outcome", barmode="stack", title="Tool calls by outcome",
                                     color_discrete_map={"success": "#54A24B", "failure": "#E45756"}), width="stretch")


def _memory() -> None:
    st.markdown("**Long-term memory events (all patients)**")
    st.dataframe(df(db.query("SELECT m.*, p.name patient FROM memory_events m LEFT JOIN patients p ON p.id=m.patient_id "
                             "ORDER BY m.id DESC LIMIT 300"), ["created_at", "patient", "kind", "content", "run_id"]),
                 hide_index=True, width="stretch")
    store_name = st.radio("Vector store", ["patient_memory", "medical_kb"], horizontal=True)
    vs = memory.store(store_name)
    st.caption(f"{len(vs)} vectors in FAISS index `{store_name}`")
    q = st.text_input("Semantic query", placeholder="blood pressure medication")
    if q:
        hits = vs.search(q, k=8)
        st.dataframe(pd.DataFrame([{"distance": round(float(s), 3), **{k: v for k, v in d.metadata.items()
                                                                        if k in ("name", "kind", "title", "date")},
                                    "text": d.page_content[:300]} for d, s in hits]), hide_index=True, width="stretch")


def page() -> None:
    st.title("🧠 Memory & logs")
    t1, t2, t3 = st.tabs(["Agent runs & plans", "Tool usage log", "Memory"])
    with t1:
        _runs()
    with t2:
        _tool_logs()
    with t3:
        _memory()
