#!/usr/bin/env bash
# Launch the dashboard with the project's virtual environment (not a global/Anaconda Streamlit).
cd "$(dirname "$0")"
exec .venv/bin/streamlit run app.py "$@"
