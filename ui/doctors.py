"""Doctor view: daily schedule, utilisation, and patient context for upcoming appointments."""
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from hca import db
from hca.tools import appointments as appt
from hca.tools import records
from ui.common import as_list, doctor_options


def page() -> None:
    st.title("🩺 Doctors")
    opts = doctor_options()
    did = opts[st.selectbox("Doctor", list(opts))]
    d = db.query_one("SELECT * FROM doctors WHERE id=?", (did,))
    st.caption(f"{d['specialty']} · {d['location']} · {d['bio']}")

    stats = db.query_one("SELECT COUNT(*) total, SUM(status='booked') booked FROM slots WHERE doctor_id=? AND start_ts>=?",
                         (did, date.today().isoformat()))
    upcoming = appt.list_appointments(doctor_id=did, status="booked", upcoming_only=True)
    c = st.columns(3)
    c[0].metric("Upcoming slots", stats["total"] or 0)
    c[1].metric("Utilisation", f"{(stats['booked'] or 0) / max(1, stats['total'] or 0):.0%}")
    c[2].metric("Appointments via this system", len(upcoming))

    left, right = st.columns([1, 1])
    with left:
        day = st.date_input("Day", date.today() + timedelta(days=1))
        sched = appt.doctor_day_schedule(did, day)
        if not sched:
            st.info("No clinic hours that day.")
        else:
            st.dataframe(pd.DataFrame([{
                "time": s["start_ts"][11:16],
                "status": "🟢 open" if s["status"] == "open" else ("🔵 " + s["patient"] if s["patient"] else "⚫ booked (external)"),
                "reason": s["reason"] or ""} for s in sched]), hide_index=True, width="stretch", height=520)
    with right:
        st.markdown("**Upcoming patients (with EHR context)**")
        if not upcoming:
            st.info("No patients booked through the assistant yet.")
        for a in upcoming:
            p = records.get_patient(a["patient_id"])
            with st.expander(f"{a['start'][:16].replace('T', ' ')} · {a['patient']}"):
                st.markdown(f"**Reason:** {a['reason'] or '—'}  \n**Conditions:** {as_list(p['conditions'])}  \n"
                            f"**Medications:** {as_list(p['medications'])}  \n**Allergies:** {as_list(p['allergies'])}")
                if p.get("summary"):
                    st.caption(p["summary"])

    week = db.query("SELECT date(start_ts) day, SUM(status='booked') booked, SUM(status='open') open FROM slots "
                    "WHERE doctor_id=? AND start_ts>=? GROUP BY 1 ORDER BY 1 LIMIT 14", (did, date.today().isoformat()))
    if week:
        frame = pd.DataFrame(week).melt(id_vars="day", var_name="status", value_name="slots")
        fig = px.bar(frame, x="day", y="slots", color="status", title="Capacity, next 14 days",
                     color_discrete_map={"booked": "#4C78A8", "open": "#9ECAE9"})
        st.plotly_chart(fig, width="stretch")
