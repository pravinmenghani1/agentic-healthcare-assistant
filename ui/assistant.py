"""Chat with the agent; shows live planning/execution progress and the plan breakdown for each answer."""
import streamlit as st

from hca.agent.graph import run_agent
from ui.common import patient_options, render_plan

EXAMPLES = [
    "My 70-year-old father has chronic kidney disease. I want to book a nephrologist for him. Also, can you summarize latest treatment methods?",
    "Show David Thompson's medical history and book him with an endocrinologist next Friday afternoon",
    "Add to Anjali Mehra's record: cough resolved, prescribed cetirizine 10mg for 5 days, allergic to penicillin",
    "What are the symptoms of migraine?",
    "Which appointments does David Thompson have coming up?",
]
NODE_LABELS = {"load_context": "🧠 Loading memory & context", "plan": "🧭 Planning sub-goals",
               "respond": "💬 Composing reply"}


def _run(query: str) -> None:
    ss = st.session_state
    ss.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.status("Thinking ...", expanded=True) as status:
            def on_event(e: dict) -> None:
                if e["status"] == "running":
                    label = f"🔧 {e['step']['tool']} — {e['step']['goal']}" if "step" in e else NODE_LABELS[e["node"]]
                    status.update(label=label)
                elif e["node"] == "plan":
                    status.write("🧭 Plan: " + " → ".join(s["tool"] for s in e["detail"]["plan"]["steps"]))
                elif "step" in e:
                    status.write(f"{'✅' if e['ok'] else '❌'} {e['step']['tool']}")
            state = run_agent(query, chat_history=ss.messages[:-1], active_patient_id=ss.get("active_patient_id"),
                              pending_request=ss.get("pending_request"), on_event=on_event)
            status.update(label="Done", state="complete", expanded=False)
        st.markdown(state["response"])
        with st.expander("🧭 Plan & trace"):
            render_plan({**state["plan"], "repairs": state.get("plan_repairs")}, state["trace"], state["results"])
    ss.pending_request = state.get("pending_request")
    ss.messages.append({"role": "assistant", "content": state["response"], "run_id": state["run_id"],
                        "plan": {**state["plan"], "repairs": state.get("plan_repairs")}, "trace": state["trace"],
                        "results": state["results"]})


def page() -> None:
    ss = st.session_state
    ss.setdefault("messages", [])
    st.title("🤖 Healthcare Assistant")
    st.caption("Books appointments, manages records, summarizes histories and searches MedlinePlus / WHO — "
               "running entirely on local Ollama models.")
    with st.sidebar:
        opts = patient_options(include_none=True)
        choice = st.selectbox("Active patient (optional context)", list(opts), key="active_patient_label")
        ss.active_patient_id = opts[choice]
        if ss.get("pending_request"):
            st.info(f"Awaiting details for: _{ss.pending_request[:120]}_")
        if st.button("🧹 New conversation"):
            ss.messages, ss.pending_request = [], None
            st.rerun()

    for m in ss.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("plan"):
                with st.expander("🧭 Plan & trace"):
                    render_plan(m["plan"], m["trace"], m["results"])

    if not ss.messages:
        st.markdown("**Try one of these:**")
        for i, ex in enumerate(EXAMPLES):
            if st.button(ex, key=f"ex{i}"):
                ss.queued = ex
                st.rerun()

    query = st.chat_input("Ask the assistant ...") or ss.pop("queued", None)
    if query:
        _run(query)
