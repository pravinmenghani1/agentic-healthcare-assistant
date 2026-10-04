"""Scenario lab: run predefined or custom multi-turn scenarios and check the agent's behaviour."""
import streamlit as st

from hca import db, memory
from hca.agent.graph import run_agent
from ui.common import render_plan

SCENARIOS = {
    "Sample scenario (problem statement)": {
        "turns": ["My 70-year-old father has chronic kidney disease. I want to book a nephrologist for him. Also, can "
                  "you summarize latest treatment methods?", "His name is Ramesh Kulkarni."],
        "expect": {"identify_patient", "get_medical_history", "find_doctor_slots", "book_appointment", "medical_search"},
    },
    "History + booking with date": {
        "turns": ["Show David Thompson's medical history and book him with an endocrinologist next Friday afternoon"],
        "expect": {"identify_patient", "get_medical_history", "find_doctor_slots", "book_appointment"},
    },
    "Attendant updates a record": {
        "turns": ["Add to Anjali Mehra's record: cough resolved, prescribed cetirizine 10mg for 5 days, allergic to penicillin"],
        "expect": {"identify_patient", "add_medical_record"},
    },
    "Unknown patient": {
        "turns": ["Book a cardiologist for John Doe tomorrow"],
        "expect": {"identify_patient", "find_doctor_slots", "book_appointment"},
    },
    "Medical information only": {
        "turns": ["What are the symptoms of migraine?"],
        "expect": {"medical_search"},
    },
    "Cancel then list": {
        "turns": ["Cancel Anjali Mehra's appointment", "Which appointments does Anjali Mehra have?"],
        "expect": {"identify_patient", "cancel_appointment", "list_appointments"},
    },
}


def _rollback(run_ids: list[str]) -> None:
    for rid in run_ids:
        for a in db.query("SELECT * FROM appointments WHERE run_id=?", (rid,)):
            db.execute("UPDATE slots SET status='open' WHERE id=?", (a["slot_id"],))
            db.execute("DELETE FROM appointments WHERE id=?", (a["id"],))
        memory.forget_run(rid)


def page() -> None:
    st.title("🧪 Scenario lab")
    st.caption("Run multi-turn scenarios end-to-end and compare the tools the agent used with the expected ones.")
    name = st.selectbox("Scenario", [*SCENARIOS, "Custom"])
    if name == "Custom":
        text = st.text_area("One user turn per line", height=120)
        turns, expect = [t for t in text.splitlines() if t.strip()], set()
    else:
        turns, expect = SCENARIOS[name]["turns"], SCENARIOS[name]["expect"]
        for i, t in enumerate(turns, 1):
            st.markdown(f"**Turn {i}:** {t}")
    rollback = st.checkbox("Roll back bookings and memory written by this scenario afterwards", value=True)
    if not st.button("Run scenario", type="primary", disabled=not turns):
        return

    history, pending, used, run_ids = [], None, set(), []
    for i, q in enumerate(turns, 1):
        with st.status(f"Turn {i}: {q[:80]}", expanded=False) as s:
            state = run_agent(q, chat_history=history, pending_request=pending, source="scenario",
                              on_event=lambda e: s.update(label=f"Turn {i}: {e['node']}"
                                                          + (f" · {e['step']['tool']}" if "step" in e else "")))
            s.update(label=f"Turn {i} done", state="complete")
        pending = state.get("pending_request")
        run_ids.append(state["run_id"])
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": state["response"]}]
        used |= {r["tool"] for r in state["results"] if r["ok"]}
        st.markdown(f"#### Turn {i}")
        st.markdown(state["response"])
        with st.expander("Plan & trace"):
            render_plan({**state["plan"], "repairs": state.get("plan_repairs")}, state["trace"], state["results"])

    if expect:
        missing = expect - used
        (st.success if not missing else st.warning)(
            f"Expected tools succeeded: {sorted(expect & used)}" + (f" · not completed: {sorted(missing)}" if missing else ""))
    if rollback:
        _rollback(run_ids)
        st.caption("Scenario bookings and memory rolled back.")
