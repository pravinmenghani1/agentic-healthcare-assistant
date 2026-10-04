"""Shared Streamlit helpers."""
import json

import pandas as pd
import streamlit as st

from hca import db, llm, seed


@st.cache_resource
def _init_db() -> bool:
    db.init_db()
    return True


def ensure_ready() -> bool:
    """Make sure the DB exists and is seeded; offer seeding otherwise. Returns False if the page should stop."""
    _init_db()
    if db.query_one("SELECT COUNT(*) AS n FROM patients")["n"]:
        return True
    st.warning("The database is empty. Seed it from the dataset (records.xlsx + clinical PDFs) to get started.")
    use_llm = st.checkbox("Enrich patient profiles with the LLM (slower)", value=True)
    if st.button("Initialize data", type="primary"):
        with st.status("Seeding ...", expanded=True) as s:
            seed.run(use_llm=use_llm, log=s.write)
            s.update(label="Seeded", state="complete")
        st.rerun()
    return False


def sidebar_status() -> None:
    status = llm.ollama_status()
    with st.sidebar:
        if status["ok"]:
            st.caption(f"🟢 Ollama · agent `{llm.config.LLM_MODEL}` · grader `{llm.config.EVAL_MODEL}` · "
                       f"embeddings `{llm.config.EMBED_MODEL}`")
        else:
            st.error(status.get("error") or f"Missing Ollama models: {', '.join(status['missing'])}. "
                     f"Run `ollama pull <model>`.")


def patient_options(include_none: bool = False) -> dict[str, int | None]:
    opts = {f"{p['name']} (#{p['id']}, {p.get('age')} y)": p["id"] for p in db.query("SELECT * FROM patients ORDER BY name")}
    return ({"— none —": None} | opts) if include_none else opts


def doctor_options() -> dict[str, int]:
    return {f"{d['name']} · {d['specialty']}": d["id"] for d in db.query("SELECT * FROM doctors ORDER BY specialty, name")}


def as_list(value) -> str:
    return ", ".join(value) if isinstance(value, list) and value else "—"


def df(rows: list[dict], columns: list[str] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if columns and not frame.empty:
        frame = frame[[c for c in columns if c in frame.columns]]
    for c in frame.columns:
        if frame[c].map(lambda v: isinstance(v, (list, dict))).any():
            frame[c] = frame[c].map(lambda v: json.dumps(v, default=str) if isinstance(v, (list, dict)) else v)
    return frame


def render_plan(plan: dict, trace: list[dict] | None = None, results: list[dict] | None = None) -> None:
    """Planning breakdown: extracted entities, sub-goals with tool + outcome, repairs, and node timings."""
    entities = {k: v for k, v in plan.items() if k not in ("steps", "repairs") and v not in (None, [], "")}
    st.markdown(f"**Intent:** {plan.get('intent', '—')}")
    st.json(entities, expanded=False)
    outcome = {}
    if results:
        outcome = {r["tool"]: ("✅" if r["ok"] else ("⏸️ skipped" if r["result"].get("skipped") else "❌")) for r in results}
    elif trace:
        for t in trace:
            if t["node"].startswith("execute:"):
                outcome[t["node"].split(":", 1)[1]] = "✅" if t["detail"].get("ok") else "❌"
    st.dataframe(pd.DataFrame([{"#": i + 1, "tool": s["tool"], "sub-goal": s["goal"], "outcome": outcome.get(s["tool"], "")}
                               for i, s in enumerate(plan.get("steps", []))]), hide_index=True, width="stretch")
    if plan.get("repairs"):
        st.caption("Plan guard-rails applied: " + " · ".join(plan["repairs"]))
    if trace:
        st.dataframe(pd.DataFrame([{"node": t["node"], "ms": t["ms"], "at": t.get("at")} for t in trace]),
                     hide_index=True, width="stretch")
