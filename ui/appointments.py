"""Real-time appointment tracking (auto-refreshing) and manual booking."""
from datetime import date, datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from hca import db
from hca.tools import appointments as appt
from ui.common import patient_options


@st.fragment(run_every="5s")
def _live() -> None:
    rows = appt.list_appointments()
    now = datetime.now().isoformat()
    upcoming = [r for r in rows if r["status"] == "booked" and r["start"] >= now]
    c = st.columns(5)
    c[0].metric("Upcoming", len(upcoming))
    c[1].metric("Today", sum(r["start"][:10] == date.today().isoformat() for r in upcoming))
    c[2].metric("Booked by agent", sum(r["booked_by"] == "agent" for r in rows if r["status"] == "booked"))
    c[3].metric("Cancelled", sum(r["status"] == "cancelled" for r in rows))
    c[4].metric("Last update", datetime.now().strftime("%H:%M:%S"))
    if not rows:
        st.info("No appointments yet.")
        return
    frame = pd.DataFrame(rows)
    status = st.session_state.get("appt_status_filter", ["booked"])
    view = frame[frame["status"].isin(status)] if status else frame
    st.dataframe(view[["appointment_id", "start", "patient", "doctor", "specialty", "location", "status", "booked_by",
                       "reason", "created_at"]].sort_values("start"), hide_index=True, width="stretch")
    if not view.empty:
        tl = view.assign(start=pd.to_datetime(view["start"]))
        tl["end"] = tl["start"] + pd.Timedelta(minutes=30)
        fig = px.timeline(tl, x_start="start", x_end="end", y="doctor", color="specialty", hover_data=["patient", "reason"],
                          title="Appointment timeline")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="stretch")
    st.markdown("**Recent activity**")
    st.dataframe(pd.DataFrame(db.query(
        "SELECT a.updated_at, a.status, p.name patient, d.name doctor, a.booked_by FROM appointments a "
        "JOIN patients p ON p.id=a.patient_id JOIN doctors d ON d.id=a.doctor_id ORDER BY a.updated_at DESC LIMIT 8")),
        hide_index=True, width="stretch")


def page() -> None:
    st.title("📅 Appointments")
    st.caption("Live view — refreshes every 5 seconds, so bookings made by the agent in another tab appear here.")
    st.multiselect("Status filter", ["booked", "cancelled", "completed"], default=["booked"], key="appt_status_filter")
    _live()

    with st.expander("➕ Book manually"):
        popts = patient_options()
        pid = popts[st.selectbox("Patient", list(popts), key="mb_patient")]
        spec = st.selectbox("Specialty", appt.list_specialties(), key="mb_spec")
        day = st.date_input("From date", date.today(), key="mb_day")
        res = appt.find_slots(spec, date_from=day.isoformat(), limit=8)
        if not res["ok"]:
            st.warning(res["error"])
        else:
            labels = {f"{s['display']} — {s['doctor']} ({s['location']})": s["slot_id"] for s in res["slots"]}
            slot = st.selectbox("Slot", list(labels), key="mb_slot")
            reason = st.text_input("Reason", key="mb_reason")
            if st.button("Book", type="primary"):
                out = appt.book_appointment(pid, labels[slot], reason, booked_by="ui")
                (st.success if out["ok"] else st.error)(out.get("appointment") or out.get("error"))
