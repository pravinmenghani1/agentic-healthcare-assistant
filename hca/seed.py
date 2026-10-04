"""Build the local state: SQLite EHR + doctor schedules from the dataset, and the FAISS patient memory."""
import random
from datetime import date, datetime, time, timedelta

from . import config, db, memory
from .ingest import load_records_xlsx
from .tools import records

# Synthetic doctor directory (the dataset has no doctor data). Patients live in India and the US.
DOCTORS = [
    ("Dr. Priya Raman", "Nephrologist", "Chennai Kidney & Dialysis Centre", "Chronic kidney disease, dialysis planning"),
    ("Dr. Arjun Menon", "Nephrologist", "Apollo Clinic, Bangalore", "Hypertensive and diabetic kidney disease"),
    ("Dr. Sarah Collins", "General Physician", "Bridport Family Medicine, Tigard OR", "Primary care, preventive health"),
    ("Dr. Rakesh Verma", "General Physician", "City Care Clinic, Pune", "Acute infections, routine check-ups"),
    ("Dr. Vikram Shah", "Cardiologist", "Heart Institute, Chennai", "Hypertension, heart failure"),
    ("Dr. Neha Kapoor", "Endocrinologist", "Diabetes & Thyroid Centre, Bangalore", "Type 2 diabetes, thyroid"),
    ("Dr. Rohan Iyer", "Pulmonologist", "Lung Care Hospital, Pune", "Asthma, COPD, respiratory infections"),
    ("Dr. Emily Carter", "Gynecologist", "Women's Health Clinic, Portland OR", "PCOS, cervical screening"),
    ("Dr. Sanjay Gupta", "Neurologist", "NeuroCare, Delhi", "Migraine, epilepsy, stroke"),
    ("Dr. Michael Brooks", "Orthopedist", "OrthoPlus, Tigard OR", "Musculoskeletal pain, joints"),
    ("Kavya Nair, RD", "Nutritionist", "Wellness Hub, Bangalore", "Diabetic and renal diets"),
]
DAY_SLOTS = [time(h, m) for h in (9, 10, 11, 12, 14, 15, 16) for m in (0, 30)]


def reset_state() -> None:
    config.DB_PATH.unlink(missing_ok=True)
    for suffix in ("-wal", "-shm"):
        config.DB_PATH.with_name(config.DB_PATH.name + suffix).unlink(missing_ok=True)
    memory.patient_store().reset()
    memory.knowledge_store().reset()
    db.init_db()


def seed_doctors(days: int = config.SCHEDULE_DAYS, busy_ratio: float = 0.35, seed: int = 7) -> int:
    rng = random.Random(seed)
    start = date.today()
    n = 0
    with db.session() as conn:
        for name, specialty, location, bio in DOCTORS:
            did = conn.execute("INSERT INTO doctors (name, specialty, location, bio) VALUES (?,?,?,?)",
                               (name, specialty, location, bio)).lastrowid
            for offset in range(days):
                day = start + timedelta(days=offset)
                if day.weekday() >= 5 and rng.random() < 0.8:  # most doctors are off on weekends
                    continue
                for t in DAY_SLOTS:
                    st = datetime.combine(day, t)
                    if st < datetime.now():
                        continue
                    # Some slots are already taken by patients outside this system.
                    status = "booked" if rng.random() < busy_ratio else "open"
                    conn.execute("INSERT INTO slots (doctor_id, start_ts, end_ts, status) VALUES (?,?,?,?)",
                                 (did, st.isoformat(timespec="seconds"),
                                  (st + timedelta(minutes=30)).isoformat(timespec="seconds"), status))
                    n += 1
    return n


def seed_patients(use_llm: bool = True, log=print) -> None:
    for p in load_records_xlsx(config.DATA_DIR / "records.xlsx"):
        records.register_patient(p["name"], p["age"], p["gender"], p["phone"], p["email"], p["address"],
                                 summary=p["summary"])
        log(f"  patient: {p['name']}")
    for pdf in sorted(config.DATA_DIR.glob("*.pdf")):
        res = records.ingest_pdf_document(pdf, created_by="seed")
        log(f"  {pdf.name}: {res}")
    if use_llm:
        for p in records.list_patients():
            log(f"  enriching profile with LLM: {p['name']}")
            records.enrich_profile(p["id"])


def seed_appointments() -> None:
    """Follow-ups already ordered in the dataset (David: nutritionist; Anjali: GP re-check)."""
    from .tools import appointments as appt
    for name, specialty, reason in (("David Thompson", "Nutritionist", "Diabetes diet follow-up (ordered 2024-01-10)"),
                                    ("Anjali Mehra", "General Physician", "Follow-up for persistent cough")):
        p = db.query_one("SELECT id, address FROM patients WHERE name=?", (name,))
        slots = appt.find_slots(specialty, near=p["address"] if p else None, limit=1)
        if p and slots["ok"]:
            appt.book_appointment(p["id"], slots["slots"][0]["slot_id"], reason, booked_by="seed")


def run(use_llm: bool = True, log=print) -> None:
    log("Resetting state ...")
    reset_state()
    log(f"Doctors & slots: {seed_doctors()} slots generated")
    log("Patients & records ...")
    seed_patients(use_llm, log)
    seed_appointments()
    log(f"FAISS patient memory: {memory.rebuild_patient_index()} vectors")
    db.execute("DELETE FROM tool_logs")  # seeding activity is not agent performance
    log("Done.")
