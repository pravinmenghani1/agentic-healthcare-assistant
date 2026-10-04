"""Patient view: EHR browsing, attendant record entry (structured / free text / PDF), AI history summary, memory."""
import pandas as pd
import streamlit as st

from hca import db, memory
from hca.tools import appointments as appt
from hca.tools import records
from ui.common import as_list, df, patient_options


def _overview() -> None:
    rows = records.list_patients()
    st.dataframe(pd.DataFrame([{"id": p["id"], "name": p["name"], "age": p["age"], "gender": p["gender"],
                                "conditions": as_list(p["conditions"]), "medications": len(p["medications"] or []),
                                "address": p["address"]} for p in rows]), hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1.expander("🔎 Semantic patient search (FAISS)"):
        q = st.text_input("Describe the patient", placeholder="woman with chest pain")
        if q:
            res = records.find_patient(description=q)
            if res["ok"]:
                st.success(f"Best match: **{res['patient']['name']}** (distance {res['score']}) · "
                           f"candidates: {', '.join(res['candidates'])}")
            else:
                st.warning(res["error"])
    with c2.expander("➕ Register a new patient"):
        with st.form("register", clear_on_submit=True):
            name = st.text_input("Full name*")
            a, g = st.columns(2)
            age = a.number_input("Age", 0, 120, 40)
            gender = g.selectbox("Gender", ["Female", "Male", "Other"])
            phone = st.text_input("Phone")
            address = st.text_input("Address")
            conds = st.text_input("Known conditions (comma separated)")
            if st.form_submit_button("Register") and name:
                res = records.register_patient(name, int(age), gender, phone or None, None, address or None,
                                               [c.strip() for c in conds.split(",") if c.strip()])
                st.success(("Registered " if res["created"] else "Already exists: ") + res["patient"]["name"])


def _profile(p: dict) -> None:
    c = st.columns(4)
    c[0].metric("Age", p["age"] or "—")
    c[1].metric("Gender", p["gender"] or "—")
    c[2].metric("Records", db.query_one("SELECT COUNT(*) n FROM medical_records WHERE patient_id=?", (p["id"],))["n"])
    c[3].metric("Upcoming appts", len(appt.list_appointments(patient_id=p["id"], status="booked", upcoming_only=True)))
    st.markdown(f"**Conditions:** {as_list(p['conditions'])}  \n**Medications:** {as_list(p['medications'])}  \n"
                f"**Allergies:** {as_list(p['allergies'])}  \n**DOB:** {p.get('dob') or '—'} · **Phone:** "
                f"{p.get('phone') or '—'} · **Email:** {p.get('email') or '—'}  \n**Address:** {p.get('address') or '—'}")
    if p.get("summary"):
        st.info(p["summary"])
    with st.expander("✏️ Update demographics / structured fields"):
        with st.form(f"upd{p['id']}"):
            cols = st.columns(2)
            age = cols[0].number_input("Age", 0, 120, int(p["age"] or 0))
            phone = cols[1].text_input("Phone", p.get("phone") or "")
            address = st.text_input("Address", p.get("address") or "")
            conditions = st.text_area("Conditions (one per line)", "\n".join(p["conditions"] or []))
            meds = st.text_area("Medications (one per line)", "\n".join(p["medications"] or []))
            allergies = st.text_input("Allergies (comma separated)", ", ".join(p["allergies"] or []))
            if st.form_submit_button("Save"):
                records.update_patient(p["id"], age=int(age), phone=phone, address=address,
                                       conditions=[x.strip() for x in conditions.splitlines() if x.strip()],
                                       medications=[x.strip() for x in meds.splitlines() if x.strip()],
                                       allergies=[x.strip() for x in allergies.split(",") if x.strip()])
                st.success("Saved and re-indexed in memory")
                st.rerun()


def _records(p: dict) -> None:
    for r in records.get_records(p["id"]):
        dx = ", ".join(f"{d['name']} ({d.get('code') or '—'})" for d in r["diagnoses"] or []) or "—"
        with st.expander(f"{r['visit_date'] or 'undated'} · {r['record_type']} · {r['source']} · Dx: {dx}"):
            for label in ("subjective", "objective", "assessment", "plan"):
                if r.get(label):
                    st.markdown(f"**{label.title()}:** {r[label]}")
            if r.get("medications"):
                st.markdown(f"**Medications:** {as_list(r['medications'])}")
            if r["record_type"] == "document" or not any(r.get(k) for k in ("subjective", "plan")):
                st.text(r.get("raw_text", "")[:5000])


def _add_record(p: dict) -> None:
    mode = st.radio("Entry type", ["Free-text note (LLM structures it)", "Structured SOAP form", "Upload clinical PDF"],
                    horizontal=True)
    if mode.startswith("Free"):
        with st.form("note", clear_on_submit=True):
            note = st.text_area("Attendant note", height=150, placeholder="Seen today for ... BP 150/95 ... started "
                                                                          "amlodipine 5mg OD ... allergic to sulfa")
            if st.form_submit_button("Add to record") and note.strip():
                with st.spinner("Structuring note with the LLM ..."):
                    res = records.add_record(p["id"], note)
                st.success(f"Added record #{res['record_id']}")
                st.json(res["structured"])
    elif mode.startswith("Structured"):
        with st.form("soap", clear_on_submit=True):
            visit_date = st.date_input("Visit date")
            s, o = st.text_area("Subjective"), st.text_area("Objective")
            a, pl = st.text_area("Assessment"), st.text_area("Plan")
            dx = st.text_input("Diagnoses (name[:ICD10], comma separated)", placeholder="Hypertension:I10")
            meds = st.text_input("Medications (comma separated)")
            if st.form_submit_button("Add to record"):
                diagnoses = [{"condition": d.split(":")[0].strip(), "code": (d.split(":")[1].strip() if ":" in d else None)}
                             for d in dx.split(",") if d.strip()]
                res = records.add_record(p["id"], "\n".join([s, o, a, pl]), structured={
                    "visit_date": visit_date.isoformat(), "subjective": s, "objective": o, "assessment": a, "plan": pl,
                    "diagnoses": diagnoses, "medications": [m.strip() for m in meds.split(",") if m.strip()]})
                st.success(f"Added record #{res['record_id']}")
    else:
        up = st.file_uploader("Clinical PDF (H&P note / visit summary)", type="pdf")
        if up and st.button("Ingest PDF"):
            res = records.ingest_pdf_document(up.getvalue(), filename=up.name, patient_id=p["id"])
            (st.success if res["ok"] else st.error)(str(res))


def _summary(p: dict) -> None:
    focus = st.text_input("Focus (optional)", placeholder="e.g. kidney function, diabetes control")
    if st.button("🧾 Generate AI history summary", type="primary"):
        with st.spinner("Retrieving records + memory and summarizing ..."):
            res = records.summarize_history(p["id"], focus or None)
        st.markdown(res["markdown"])
        st.caption(f"Used {res['records_used']} records and {res['memories_used']} memory items.")


def _appointments(p: dict) -> None:
    rows = appt.list_appointments(patient_id=p["id"])
    if not rows:
        st.info("No appointments.")
    for a in rows:
        c1, c2 = st.columns([5, 1])
        c1.markdown(f"**{a['start'][:16].replace('T', ' ')}** · {a['doctor']} ({a['specialty']}) · {a['status']} · "
                    f"_{a['reason'] or ''}_")
        if a["status"] == "booked" and c2.button("Cancel", key=f"cx{a['appointment_id']}"):
            appt.cancel_appointment(a["appointment_id"])
            st.rerun()


def _memory(p: dict) -> None:
    st.markdown("**Long-term memory events**")
    st.dataframe(df(memory.recent_events(p["id"], 50), ["created_at", "kind", "content", "run_id"]),
                 hide_index=True, width="stretch")
    q = st.text_input("Semantic recall over this patient's memory", placeholder="kidney")
    if q:
        hits = memory.recall(q, p["id"], k=6)
        st.dataframe(df(hits, ["score", "kind", "date", "text"]), hide_index=True, width="stretch")


def page() -> None:
    st.title("🧑 Patients")
    _overview()
    opts = patient_options()
    label = st.selectbox("Patient", list(opts))
    p = records.get_patient(opts[label])
    st.subheader(p["name"])
    tabs = st.tabs(["Profile", "Records", "Add / update record", "AI summary", "Appointments", "Memory"])
    for tab, fn in zip(tabs, (_profile, _records, _add_record, _summary, _appointments, _memory)):
        with tab:
            fn(p)
