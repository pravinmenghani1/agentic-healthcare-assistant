"""Agentic Healthcare Assistant — Streamlit dashboard.

    streamlit run app.py
"""
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import streamlit as st  # noqa: E402

if not hasattr(st, "navigation") or not hasattr(st, "fragment"):
    st.error(f"This app needs Streamlit ≥ 1.40 but is running {st.__version__} from a different Python environment. "
             "Start it with the project's virtual environment:  `./run.sh`  or  `.venv/bin/streamlit run app.py`")
    st.stop()

from ui import appointments, assistant, doctors, evaluation, medical_info, memory_logs, patients, scenarios  # noqa: E402
from ui.common import ensure_ready, sidebar_status  # noqa: E402

st.set_page_config(page_title="Agentic Healthcare Assistant", page_icon="🏥", layout="wide")

pages = st.navigation({
    "Assistant": [st.Page(assistant.page, title="Assistant", icon="🤖", url_path="assistant", default=True),
                  st.Page(scenarios.page, title="Scenario lab", icon="🧪", url_path="scenarios")],
    "Operations": [st.Page(patients.page, title="Patients", icon="🧑", url_path="patients"),
                   st.Page(doctors.page, title="Doctors", icon="🩺", url_path="doctors"),
                   st.Page(appointments.page, title="Appointments", icon="📅", url_path="appointments"),
                   st.Page(medical_info.page, title="Medical info", icon="📚", url_path="medical-info")],
    "LLMOps": [st.Page(evaluation.page, title="Evaluation", icon="📊", url_path="evaluation"),
               st.Page(memory_logs.page, title="Memory & logs", icon="🧠", url_path="memory-logs")],
})
sidebar_status()
if ensure_ready():
    pages.run()
